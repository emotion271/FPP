"""Models for fashion Net."""
import logging
import threading

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import random

import polyvore.config as cfg
import polyvore.param
import utils
from polyvore import debugger

from . import backbones as B
from . import basemodel as M

NAMED_MODELS = utils.get_named_function(B)
LOGGER = logging.getLogger(__name__)


def contrastive_loss(margin, im, s):
    size, dim = im.shape
    scores = im.matmul(s.t()) / dim
    diag = scores.diag()
    zeros = torch.zeros_like(scores)
    # shape #item x #item
    # sum along the row to get the VSE loss from each image
    cost_im = torch.max(zeros, margin - diag.view(-1, 1) + scores)
    # sum along the column to get the VSE loss from each sentence
    cost_s = torch.max(zeros, margin - diag.view(1, -1) + scores)
    # to fit parallel, only compute the average for each item
    vse_loss = cost_im.sum(dim=1) + cost_s.sum(dim=0) - 2 * margin
    # for data parallel, reshape to (size, 1)
    return vse_loss / (size - 1)


def soft_margin_loss(x):
    target = torch.ones_like(x)
    return F.soft_margin_loss(x, target, reduction="none")


@utils.singleton
class RankMetric(threading.Thread):
    def __init__(self, num_users, args=(), kwargs=None):
        from queue import Queue

        threading.Thread.__init__(self, args=(), kwargs=None)
        self.daemon = True
        self.queue = Queue()
        self.deamon = True
        self.num_users = num_users
        self._scores = [[[] for _ in range(self.num_users)] for _ in range(4)]

    def reset(self):
        self._scores = [[[] for _ in range(self.num_users)] for _ in range(4)]

    def put(self, data):
        self.queue.put(data)

    def process(self, data):
        with threading.Lock():
            uidx, scores = data
            for u in uidx:
                for n, score in enumerate(scores):
                    for s in score:
                        self._scores[n][u].append(s)

    def run(self):
        print(threading.currentThread().getName(), "RankMetric")
        while True:
            data = self.queue.get()
            if data is None:  # If you send `None`, the thread will exit.
                return
            self.process(data)

    def rank(self):
        auc = utils.metrics.calc_AUC(self._scores[0], self._scores[1])
        binary_auc = utils.metrics.calc_AUC(self._scores[2], self._scores[3])
        ndcg = utils.metrics.calc_NDCG(self._scores[0], self._scores[1])
        binary_ndcg = utils.metrics.calc_NDCG(self._scores[2], self._scores[3])
        return dict(auc=auc, binary_auc=binary_auc, ndcg=ndcg, binary_ndcg=binary_ndcg)


class FashionNet(nn.Module):
    """Base class for fashion net."""

    def __init__(self, param):
        """See NetParam for details."""
        super().__init__()
        self.param = param
        self.scale = 1.0
        self.S = nn.Parameter(torch.Tensor(1, 5, param.dim))
        nn.init.xavier_uniform_(self.S)
        # user embedding layer
        #self.user_eb = M.UserEb(param)
        self.user_embedding = M.UserEncoder(param)
        self.disen_user_outfit = M.UserOutfitDisen(param, self.S)
        # self.disen_user = M.UserDisen(param, self.S)
        # self.disen_outfit = M.OutfitDisen(param, self.S)
        # feature extractor
        if self.param.use_visual:
            self.features = NAMED_MODELS[param.backbone](tailed=True)
        # single encoder or multi-encoders
        num_encoder = 1 if param.single else cfg.NumCate
        # hashing codes
        if self.param.use_visual:
            feat_dim = self.features.dim
            self.encoder_v = nn.ModuleList(
                [M.ImgEncoder(feat_dim, param) for _ in range(num_encoder)]
            )
            #self.atten_v = M.ItemAttention(param)
        if self.param.use_semantic:
            feat_dim = 2400
            self.encoder_t = nn.ModuleList(
                [M.TxtEncoder(feat_dim, param) for _ in range(num_encoder)]
            )
            #self.atten_t = M.ItemAttention(param)
        # matching block
        if self.param.hash_types == polyvore.param.NO_WEIGHTED_HASH:
            # use learnable scale
            self.core = M.LearnableScale(1)
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_BOTH:
            # two weighted hashing for both user-item and item-item
            self.core = nn.ModuleList([M.CoreMat(param.dim), M.CoreMat(param.dim)])
        else:
            # single weighted hashing for user-item or item-item
            self.core = M.CoreMat(param.dim)
        self.match = M.Match(param)
        if self.param.use_semantic and self.param.use_visual:
            self.loss_weight = dict(rank_loss=1.0, binary_loss=None, vse_loss=0.1)
        else:
            self.loss_weight = dict(rank_loss=1.0, binary_loss=None)
        self.configure_trace()
        self.rank_metric = RankMetric(self.param.num_users)
        if not self.rank_metric.is_alive():
            self.rank_metric.start()

    def gather(self, results):
        losses, accuracy = results
        loss = 0.0
        gathered_loss = {}
        for name, value in losses.items():
            weight = self.loss_weight[name]
            value = value.mean()
            # save the scale
            gathered_loss[name] = value.item()
            if weight:
                loss += value * weight
        # save overall loss
        gathered_loss["loss"] = loss.item()
        gathered_accuracy = {k: v.sum().item() / v.numel() for k, v in accuracy.items()}
        return gathered_loss, gathered_accuracy

    def get_user_binary_code(self, device="cpu"):
        uidx = np.arange(self.param.num_users).reshape(-1, 1)
        uidx = torch.from_numpy(uidx).to(device)
        one_hot = utils.one_hot(uidx, self.param.num_users)
        user = self.user_embedding(one_hot)
        return self.sign(user).cpu().numpy()

    def get_matching_weight(self):
        if self.param.hash_types == polyvore.param.WEIGHTED_HASH_BOTH:
            weights_u = self.core[0].weight.data.cpu().numpy()
            weights_i = self.core[1].weight.data.cpu().numpy()
            w = 1.0
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_I:
            weights_u = []
            weights_i = self.core.weight.data.cpu().numpy()
            w = 1.0
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_U:
            weights_u = self.core.weight.data.cpu().numpy()
            weights_i = []
            w = 1.0
        else:
            weights_u = []
            weights_i = []
            w = self.core.weight.data.cpu().numpy()
        return weights_i, weights_u, w

    def register_figure(self, tracer, title):
        for key, trace_dict in self.tracer.items():
            tracer.register_figure(
                title=title, xlabel="iteration", ylabel=key, trace_dict=trace_dict
            )

    def configure_trace(self):
        self.tracer = dict()
        self.tracer["loss"] = {
            "train.loss": "Train Loss(*)",
            "train.binary_loss": "Train Loss",
            "test.loss": "Test Loss(*)",
            "test.binary_loss": "Test Loss",
        }
        if self.param.use_semantic and self.param.use_visual:
            # in this case, the overall loss dose not equal to rank loss
            vse = {
                "train.vse_loss": "Train VSE Loss",
                "test.vse_loss": "Test VSE Loss",
                "train.rank_loss": "Train Rank Loss",
                "test.rank_loss": "Test Rank Loss",
            }
            self.tracer["loss"].update(vse)
        self.tracer["accuracy"] = {
            "train.accuracy": "Train(*)",
            "train.binary_accuracy": "Train",
            "test.accuracy": "Test(*)",
            "test.binary_accuracy": "Test",
        }
        self.tracer["rank"] = {
            "test.auc": "AUC(*)",
            "test.binary_auc": "AUC",
            "test.ndcg": "NDCG(*)",
            "test.binary_ndcg": "NDCG",
        }

    def __repr__(self):
        return super().__repr__() + "\n" + self.param.__repr__()

    def set_scale(self, value):
        """Set scale of TanH layer."""
        if not self.param.scale_tanh:
            return
        self.scale = value
        LOGGER.info("Set the scale to %.3f", value)
        self.user_embedding.set_scale(value)
        if self.param.use_visual:
            for encoder in self.encoder_v:
                encoder.set_scale(value)
        if self.param.use_semantic:
            for encoder in self.encoder_t:
                encoder.set_scale(value)

    def scores(self, ulatent, ilatents, his, scale=10.0):
        """For simplicity, we remove top-top pair on variable-length outfit and
           use duplicated top when necessary.
        """
        size = 5
        indx, indy = np.triu_indices(size, k=1)
        if size == 4:
            # remove top-top comparison
            indx = indx[1:]
            indy = indy[1:]
        # N x size x D
        #latents = torch.stack(ilatents, dim=1)
        #print('latents:', latents.size())
        #latents, i_norm = self.disen_graph(latents)
        #print('graph_latent:', latents)
        latents = ilatents
        #ulatent = ulatent.unsqueeze(1).expand(-1, size, -1)
        #print('u_latent', ulatent.size())
        # # N x size x D
        x = latents * ulatent
        x_his = his * ulatent
        # N x Comb x D
        y = latents[:, indx] * latents[:, indy]
        # shape [N]
        if self.param.hash_types == polyvore.param.WEIGHTED_HASH_BOTH:
            x = self.core[0](x)
            x_his = self.core[0](x_his)
            score_u = self.match(x, x_his).mean(dim=(1, 2))
            score_i = self.core[1](y).mean(dim=(1, 2))
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_U:
            #score_u = self.core(x).mean(dim=1)
            #score_i = y.mean(dim=(1, 2))
            #x = self.core(x)
            score_u = self.match(x).mean(dim=(1, 2))
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_I:
            score_u = x.mean(dim=(1, 2))
            score_i = self.core(x).mean(dim=(1, 2))
        elif self.param.hash_types == polyvore.param.NO_WEIGHTED_HASH:
            score_u = self.core(x.mean(dim=(1, 2)))
            score_i = y.mean(dim=(1, 2))
        else:
            score_u = x.mean(dim=(1, 2))
            score_i = y.mean(dim=(1, 2))
        # scale score to [-2 * scale, 2 * scale]
        if self.param.zero_iterm:
            score = score_u * (scale * 2.0)
        elif self.param.zero_uterm:
            score = score_i * (scale * 2.0)
        else:
            score = (score_u + score_i) * scale
            # score = score_u*scale
        # shape N x 1
        return score.view(-1, 1)

    def sign(self, x):
        """Return hash code of x.

        if x is {1,-1} binary, then h(x) = sign(x)
        if x is {1,0} binary, then h(x) = (sign(x - 0.5) + 1)/2
        """
        if self.param.binary01:
            return ((x.detach() - 0.5).sign() + 1) / 2.0
        return x.detach().sign()

    def latent_code(self, items, encoder):
        """Return latent codes."""
        latent_code = []
        size = len(items)
        if self.param.single:
            cate = [0] * size
        else:
            cate = self.param.cate_map
        latent_code = [encoder[c](x) for c, x in zip(cate, items)]
        # shape Length * N x D
        return latent_code

    def _pairwise_output(self, lcus, pos_feat, neg_feat, his_v, encoder):
        lcpi = self.latent_code(pos_feat, encoder)
        lcni = self.latent_code(neg_feat, encoder)
        # score with relaxed features

        lcpi_s = torch.stack(lcpi, dim=1)
        lcni_s = torch.stack(lcni, dim=1)

        '''
        lcus_k = self.disen_user(lcus)
        lcpi_k = self.disen_outfit(lcpi_s)
        lcni_k = self.disen_outfit(lcni_s)
        his_vk = [self.disen_outfit(out_v) for out_v in his_v]
        his_k = torch.stack(his_vk, dim=1)
        his = torch.mean(his_k, dim=1)
        '''

        lcus_k, lcpi_k, neg_k, his = self.disen_user_outfit(lcus, lcpi_s, lcni_s, his_v)

        '''
        batch_size = lcus_k.size()[0]

        NEG = 5
        for i in range(NEG):
            rand = int((random.random() + i) * batch_size / NEG)
            neg_k = torch.cat([neg_k, torch.narrow(lcpi_k, 0, rand, batch_size-rand),
                             torch.narrow(lcpi_k, 0, 0, rand)], 0)

        pos_k = torch.repeat_interleave(lcpi_k, NEG+1, 0)
        user_k = torch.repeat_interleave(lcus_k, NEG+1, 0)
        his = torch.repeat_interleave(his, NEG+1, 0)
        '''

        user_k = lcus_k
        pos_k = lcpi_k

        pscore = self.scores(user_k, pos_k, his)
        nscore = self.scores(user_k, neg_k, his)

        # pscore, lpi = self.scores(lcus_k, lcpi_k, his)
        # nscore, lni = self.scores(lcus_k, lcni_k, his)

        # score with binary codes
        '''
        bcus = self.sign(lcus)
        bcpi = [self.sign(h) for h in lcpi]
        bcni = [self.sign(h) for h in lcni]
        bpscore, _, _ = self.scores(bcus, bcpi)
        bnscore, _, _ = self.scores(bcus, bcni)
        '''

        # stack latent codes
        # (N x Num) x D
        if self.param.variable_length:
            # only use second top item since to balance top category
            latents = torch.stack(lcpi[1:] + lcni[1:], dim=1)
        else:
            latents = torch.stack(lcpi + lcni, dim=1)
        latents = latents.view(-1, self.param.dim)
        return (pscore, nscore, pscore, nscore), latents

    def semantic_output(self, lcus, pos_feat, neg_feat):
        scores, latents = self._pairwise_output(
            lcus, pos_feat, neg_feat, self.encoder_t
        )
        debugger.put("item.s", latents)
        return scores, latents

    def visual_output(self, lcus, pos_img, neg_img, his_v):
        # extract visual features
        pos_feat = [self.features(x) for x in pos_img]
        neg_feat = [self.features(x) for x in neg_img]
        scores, latents = self._pairwise_output(
            lcus, pos_feat, neg_feat, his_v, self.encoder_v
        )
        debugger.put("item.v", latents)
        return scores, latents

    def debug(self):
        LOGGER.debug("Scale value: %.3f", self.scale)
        debugger.log("user")
        if self.param.use_visual:
            debugger.log("item.v")
        if self.param.use_semantic:
            debugger.log("item.s")
        #if isinstance(self.core, nn.ModuleList):
            #for module in self.core:
                #module.debug()

    def reg(self, s):
        #print(s.size())
        g = s.matmul(s.transpose(1, 2))
        #print(g.size())
        reg = g.diagonal(dim1=1, dim2=2).sum() - torch.logdet(g).sum()
        return reg

    def forward(self, *inputs):
        """Forward according to setting."""
        # pair-wise output
        posi, nega, user_his, uidx = inputs
        #print(posi)
        one_hot = utils.one_hot(uidx, self.param.num_users)
        # score latent codes
        lcus = self.user_embedding(one_hot).unsqueeze(1)
        #print(lcus_eb.size())
        #lcus_k = self.disen_user(lcus_eb)
        #print(lcus_k.size())
        #lcus = torch.mean(lcus_k, dim=1)
        #print(lcus.size())
        debugger.put("user", lcus)
        loss = dict()
        accuracy = dict()
        if self.param.use_semantic and self.param.use_visual:
            '''
            his_v = []
            his_t = []
            for out in user_his:
                out_v, out_t = out
                #print(out_v)
                out_v_feat = [self.features(x) for x in out_v]
                #print(out_v_feat)
                out_v_latent = torch.stack(self.latent_code(out_v_feat, self.encoder_v), dim=1)  # B*3*dim
                out_t_latent = torch.stack(self.latent_code(out_t, self.encoder_t), dim=1)
                his_v.append(out_v_latent)
                his_t.append(out_t_latent)

            his_v_latent = torch.stack(his_v, dim=1)
            his_t_latent = torch.stack(his_t, dim=1)
            # his_latent = torch.cat([his_v_latent, his_t_latent], dim=-1)
            #print(his_v_latent.size())
            his_v_eb = torch.mean(torch.sum(his_v_latent, dim=2), dim=1)
            his_t_eb = torch.mean(torch.sum(his_t_latent, dim=2), dim=1)
            #print(his_v_eb.size())
            lcus_v = self.user_embedding(torch.cat([lcus, his_v_eb], dim=-1))
            lcus_t = self.user_embedding(torch.cat([lcus, his_t_eb], dim=-1))
            debugger.put("user", lcus_v)
            #print(lcus_v.size())
            '''
            pos_v, pos_s = posi
            neg_v, neg_s = nega
            score_v, latent_v = self.visual_output(lcus, pos_v, neg_v)
            score_s, latent_s = self.semantic_output(lcus, pos_s, neg_s)
            scores = [0.5 * (v + s) for v, s in zip(score_v, score_s)]
            # visual-semantic similarity
            vse_loss = contrastive_loss(self.param.margin, latent_v, latent_s)
            loss.update(vse_loss=vse_loss)
        elif self.param.use_visual:
            his_v = []
            for out in user_his:
                out_v = out
                # print(out_v)
                out_v_feat = [self.features(x) for x in out_v]
                # print(out_v_feat)
                out_v_latent = torch.stack(self.latent_code(out_v_feat, self.encoder_v), dim=1)  # B*3*dim
                his_v.append(out_v_latent)
            scores, _ = self.visual_output(lcus, posi, nega, his_v)
        elif self.param.use_semantic:
            scores, _ = self.semantic_output(lcus, posi, nega)
        else:
            raise ValueError
        data = (uidx.tolist(), [s.tolist() for s in scores])
        self.rank_metric.put(data)
        diff = scores[0] - scores[1]
        #print('score_pos:', scores[0])
        #print('score_neg:', scores[1])
        binary_diff = scores[2] - scores[3]
        reg = self.reg(self.S)
        rank_loss = soft_margin_loss(diff) + reg*0.01
        binary_loss = soft_margin_loss(binary_diff)
        acc = torch.gt(diff.data, 0)
        binary_acc = torch.gt(binary_diff.data, 0)
        loss.update(rank_loss=rank_loss, binary_loss=binary_loss)
        accuracy.update(accuracy=acc, binary_accuracy=binary_acc)
        return loss, accuracy

    def num_gropus(self):
        """Size of sub-modules."""
        return len(self._modules)

    def active_all_param(self):
        """Active all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_all_param(self):
        """Active all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_user_param(self):
        """Freeze user's latent codes."""
        self.active_all_param()
        for param in self.user_embedding.parameters():
            param.requires_grad = False

    def freeze_item_param(self):
        """Freeze item's latent codes."""
        self.freeze_all_param()
        for param in self.user_embedding.parameters():
            param.requires_grad = True

    def init_weights(self):
        """Initialize net weights with pre-trained model.

        Each sub-module should has its own same methods.
        """
        for model in self.children():
            if isinstance(model, nn.ModuleList):
                for m in model:
                    m.init_weights()
            else:
                model.init_weights()


class FashionNetDeploy(FashionNet):
    """Fashion Net.

    Fashion Net has three parts:
    1. features: Feature extractor for all items.
    2. encoder: Learn latent code for each category.
    3. user embedding: Map user to latent codes.
    """

    def __init__(self, param):
        """Initialize FashionNet.

        Parameters:
        num: number of users.
        dim: Dimension for user latent code.

        """
        super().__init__(param)

    def binary_v(self, items):
        feat = [self.features(x) for x in items]
        latent = self.latent_code(feat, self.encoder_v)
        binary_codes = [self.sign(h) for h in latent]
        return binary_codes

    def binary_t(self, items):
        latent = self.latent_code(items, self.encoder_t)
        binary_codes = [self.sign(h) for h in latent]
        return binary_codes

    def _single_output(self, lcus, feat, his_v, encoder):
        lcpi = self.latent_code(feat, encoder)
        # score with relaxed features
        lcpi_s = torch.stack(lcpi, dim=1)
        lcus_k, lcpi_k, neg_k, his = self.disen_user_outfit(lcus, lcpi_s, lcpi_s, his_v)
        #cus_k, lcpi_k = self.disen_user_outfit(lcus, lcpi_s)
        #lcpi_k, _ = self.disen_outfit(lcpi_s)
        score = self.scores(lcus_k, lcpi_k, his)
        # score with binary codes
        #bcus = self.sign(lcus)
        #bcpi = [self.sign(h) for h in lcpi]
        #b_score, _, _ = self.scores(bcus, bcpi)
        return score, score

    def semantic_output(self, lcus, feat):
        return self._single_output(lcus, feat, self.encoder_t)

    def visual_output(self, lcus, img, his_v):
        # extract visual features
        feat = [self.features(x) for x in img]
        return self._single_output(lcus, feat, his_v, self.encoder_v)

    def forward(self, *inputs):
        """Forward.

        Return the scores for items.
        """
        items, user_his, uidx = inputs
        one_hot = utils.one_hot(uidx, self.param.num_users)
        # compute latent codes
        lcus = self.user_embedding(one_hot).unsqueeze(1)
        # print(lcus_eb.size())
        #lcus_k = self.disen_user(lcus_eb)
        if self.param.use_semantic and self.param.use_visual:

            his_v = []
            his_t = []
            for out in user_his:
                out_v, out_t = out
                out_v_feat = [self.features(x) for x in out_v]
                out_v_latent = torch.stack(self.latent_code(out_v_feat, self.encoder_v), dim=1)  # B*3*dim
                out_t_latent = torch.stack(self.latent_code(out_t, self.encoder_t), dim=1)
                his_v.append(out_v_latent)
                his_t.append(out_t_latent)

            his_v_latent = torch.stack(his_v, dim=1)
            his_t_latent = torch.stack(his_t, dim=1)
            # his_latent = torch.cat([his_v_latent, his_t_latent], dim=-1)
            # print(his_v_latent.size())
            his_v_eb = torch.mean(torch.sum(his_v_latent, dim=2), dim=1)
            his_t_eb = torch.mean(torch.sum(his_t_latent, dim=2), dim=1)
            # print(his_v_eb.size())
            lcus_v = self.user_embedding(torch.cat([lcus, his_v_eb], dim=-1))
            lcus_t = self.user_embedding(torch.cat([lcus, his_t_eb], dim=-1))

            score_v = self.visual_output(lcus_v, items[0])
            score_s = self.semantic_output(lcus_t, items[1])
            score = [0.5 * (v + s) for v, s in zip(score_v, score_s)]
        elif self.param.use_visual:
            his_v = []
            for out in user_his:
                out_v = out
                # print(out_v)
                out_v_feat = [self.features(x) for x in out_v]
                # print(out_v_feat)
                out_v_latent = torch.stack(self.latent_code(out_v_feat, self.encoder_v), dim=1)  # B*3*dim
                his_v.append(out_v_latent)
            score = self.visual_output(lcus, items, his_v)
        elif self.param.use_semantic:
            score = self.semantic_output(lcus, items)
        else:
            raise ValueError
        return score

    def compute_codes(self, items):
        """Forward.

        Return the scores for items.
        """
        # compute latent codes
        if self.param.use_semantic and self.param.use_visual:
            items_v = self.binary_v(items[0])
            items_t = self.binary_t(items[1])
            return items_v, items_t
        elif self.param.use_visual:
            items_v = self.binary_v(items)
            return items_v
        elif self.param.use_semantic:
            items_t = self.binary_t(items)
            return items_t
        else:
            raise ValueError

