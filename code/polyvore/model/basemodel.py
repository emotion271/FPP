"""Basic model utils."""
import logging

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

import polyvore.config as cfg
from polyvore import debugger
from . import attentionblock as A

NUM_ENCODER = cfg.NumCate
LOGGER = logging.getLogger(__name__)


def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=2)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 2)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if m.weight is not None:
            m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


class LatentCode(nn.Module):
    """Basic class for learning latent code."""

    def __init__(self, param):
        """Latent code.

        Parameters:
        -----------
        See option.param.NetParam
        """
        super().__init__()
        self.param = param
        self.register_buffer("scale", torch.ones(1))

    def debug(self):
        raise NotImplementedError

    def set_scale(self, value):
        """Set the scale of TanH layer."""
        self.scale.fill_(value)

    def feat(self, x):
        """Compute the feature of all images."""
        raise NotImplementedError

    def forward(self, x):
        """Forward a feature from DeepContent."""
        x = self.feat(x)
        if self.param.without_binary:
            return x
        if self.param.scale_tanh:
            x = torch.mul(x, self.scale)
        if self.param.binary01:
            return 0.5 * (torch.tanh(x) + 1)
        # shape N x D
        return torch.tanh(x).view(-1, self.param.dim)


class TxtEncoder(LatentCode):
    def __init__(self, in_feature, param):
        super().__init__(param)
        self.encoder = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, param.dim, bias=False),
        )

    def debug(self):
        debugger.log("item.s")

    def feat(self, x):
        return self.encoder(x)

    def init_weights(self):
        """Initialize weights for encoder with pre-trained model."""
        nn.init.normal_(self.encoder[0].weight.data, std=0.01)
        nn.init.constant_(self.encoder[0].bias.data, 0)
        nn.init.normal_(self.encoder[-1].weight.data, std=0.01)


class ImgEncoder(LatentCode):
    """Module for encoder to learn the latent codes."""

    def __init__(self, in_feature, param):
        """Initialize an encoder.

        Parameter
        ---------
        in_feature: feature dimension for image features
        param: see option.param.NetParam for details

        """
        super().__init__(param)
        half = in_feature // 2
        self.encoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_feature, half),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(half, param.dim, bias=False),
        )

    def debug(self):
        debugger.log("item.v")

    def feat(self, x):
        return self.encoder(x)

    def init_weights(self):
        """Initialize weights for encoder with pre-trained model."""
        nn.init.normal_(self.encoder[1].weight.data, std=0.01)
        nn.init.constant_(self.encoder[1].bias.data, 0)
        nn.init.normal_(self.encoder[-1].weight.data, std=0.01)


class UserEncoder(LatentCode):
    """User embedding layer."""

    def __init__(self, param):
        """User embedding.

        Parameters:
        ----------
        param: see option.NetParam for details

        """
        super().__init__(param)
        if param.share_user:
            self.encoder = nn.Sequential(
                nn.Linear(param.num_users, 128),
                nn.Softmax(dim=1),
                nn.Linear(128, param.dim, bias=False),
            )
        else:
            self.encoder = nn.Linear(param.num_users, param.dim, bias=False)

    def debug(self):
        debugger.log("user")

    def feat(self, x):
        return self.encoder(x)

    def init_weights(self):
        """Initialize weights for user encoder."""
        if self.param.share_user:
            nn.init.normal_(self.encoder[0].weight.data, std=0.01)
            nn.init.constant_(self.encoder[0].bias.data, 0.0)
            nn.init.normal_(self.encoder[-1].weight.data, std=0.01)
        else:
            nn.init.normal_(self.encoder.weight.data, std=0.01)


class CoreMat(nn.Module):
    """Weighted hamming similarity."""

    def __init__(self, dim):
        """Weights for this layer that is drawn from N(mu, std)."""
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.init_weights()

    def debug(self):
        weight = self.weight.data.view(-1).tolist()
        msg, args = debugger.code_to_str(weight)
        LOGGER.debug("Core Mat:" + msg, *args)

    def init_weights(self):
        """Initialize weights."""
        self.weight.data.fill_(1.0)

    def forward(self, x):
        """Forward."""
        return torch.mul(x, self.weight)

    def __repr__(self):
        """Format string for module CoreMat."""
        return self.__class__.__name__ + "(dim=" + str(self.dim) + ")"


class LearnableScale(nn.Module):
    def __init__(self, init=1.0):
        super(LearnableScale, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1).fill_(init))

    def debug(self):
        LOGGER.debug("Core Mat: %.3f", self.weight.item())

    def forward(self, inputs):
        return self.weight * inputs

    def init_weights(self):
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


'''
class ItemAttention(nn.Module):
    def __init__(self, param):
        super(ItemAttention, self).__init__()
        self.dim = param.dim
        self.att = nn.Sequential(nn.Linear(self.dim*2, self.dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(self.dim, 1),
                                 nn.Softmax(dim=1),
                                 )

    def forward(self, his_outfit, user_eb):
        his = his_outfit.reshape(-1, 3, self.dim)
        user_eb_tile = user_eb.repeat(1, 3*his_outfit.size()[1]).reshape(-1, 3, self.dim)
        query = torch.cat([his, user_eb_tile], -1)
        atten = self.att(query).reshape([-1, 1, 3])
        score = atten.reshape(-1, his_outfit.size()[1], 1, 3)

        output = torch.matmul(score, his_outfit)
        return output.squeeze(2)
'''


class UserEb(nn.Module):
    def __init__(self, param):
        super(UserEb, self).__init__()
        self.eb = nn.Linear(param.num_users, param.dim, bias=False)

    def forward(self, x):
        return self.eb(x)

    def init_weights(self):
        nn.init.normal_(self.eb.weight.data, std=1)


class ConditionalSimNet(nn.Module):
    def __init__(self):
        super(ConditionalSimNet, self).__init__()
        self.learned_mask = True
        self.hidden_dim = 128
        self.n_conditions = 5
        self.prein = False

        if self.learned_mask:
            if self.prein:
                self.masks = nn.Embedding(self.n_conditions, self.hidden_dim)
                mask_array = np.zeros([self.n_conditions, self.hidden_dim])
                mask_array.fill(0.1)
                mask_len = int(self.hidden_dim / self.n_conditions)
                for i in range(self.n_conditions):
                    mask_array[i, i * mask_len:(i + 1) * mask_len] = 1

                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                self.masks = torch.nn.Embedding(self.n_conditions, self.hidden_dim)
                self.masks.weight.data.normal_(0.9, 0.7)
        else:
            self.masks = torch.nn.Embedding(self.n_conditions, self.hidden_dim)
            mask_array = np.zeros([self.n_conditions, self.hidden_dim])
            mask_len = int(128 / self.n_conditions)
            for i in range(self.n_conditions):
                mask_array[i, i * mask_len:(i + 1) * mask_len] = 1
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)
        self.max_num = 3
        self.alpha = 0.2
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_conditions),
            # nn.Linear(self.hidden_dim * 2, self.n_conditions),
            nn.Softmax(dim=-1)
        )

        self.ego = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )

        self.mess = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )
        '''
        self.score = nn.Sequential(

            nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim // 2, 1, bias=False),
            nn.Sigmoid()

        )
        '''
        self.apply(init_weight)

    def _get_outfit_graph_feat(self, img_embedding, masked_embedding):

        atten_input = self._prepare_attentional_mechanism_input(img_embedding)

        e = self.attention(atten_input).contiguous().view(-1, self.max_num * self.max_num, self.n_conditions).unsqueeze(
            3)
        attention = e
        e = e.repeat(1, 1, 1, self.hidden_dim)

        ones = torch.ones(3, 3).cuda(0)
        ones_diag = torch.diag(torch.ones(3)).cuda(0)
        adj = ones - ones_diag
        masked_embedding_in_chunks = masked_embedding.repeat_interleave(self.max_num, dim=1)

        masked_embedding_alternating = masked_embedding.repeat(1, self.max_num, 1, 1)
        relation = masked_embedding_in_chunks * masked_embedding_alternating  ## [batch, max_num*max_num, n_condition, dim]
        # relation = masked_embedding_alternating ## [batch, max_num*max_num, n_condition, dim]
        relation = relation * e

        relation = torch.sum(relation, dim=2)
        adj = adj.view(-1, self.max_num * self.max_num, 1).repeat(1, 1, self.hidden_dim)
        relation = relation * adj
        relation = relation.view(-1, self.max_num, self.max_num, self.hidden_dim)
        relation = torch.sum(relation, dim=2)
        #seq_len = seq_len.repeat_interleave(self.hidden_dim * self.max_num, dim=0).view(-1, self.max_num, self.hidden_dim)

        relation = torch.div(relation, 3.0)
        com_mess = self.mess(relation)
        ego_mess = self.ego(img_embedding)

        new_code = (ego_mess + com_mess).view(-1, self.max_num, self.hidden_dim)
        #new_code = new_code * mask

        return new_code

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh_repeated_in_chunks = Wh.repeat_interleave(self.max_num, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, self.max_num, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(-1, self.max_num, self.max_num, 2 * self.hidden_dim)

    def _compatibility_score(self, ilatents, mask):  # [b,4,dim]
        # ilatents:[batch, max_num, 4096]#512
        b = ilatents.size()[0]
        m = ilatents.size()[1]
        d = ilatents.size()[2]

        iscore = self.score(ilatents)  # [b,4,8]

        y = torch.sum(iscore, dim=2)  # [b,4]
        # score = torch.mean(y, dim=1)
        score = torch.sum(y * mask, dim=1)
        # score = torch.div(y)
        return score

    def forward(self, input):

        #print('input:', input)
        features = F.normalize(input, p=2, dim=-1)
        #img_embedding = features * mask_512  # as input and comparison [batch, max_num, 512]
        img_embedding = features
        masked_embedding = img_embedding.unsqueeze(2).repeat(1, 1, self.n_conditions, 1)
        mask_weight = F.relu(self.masks.weight)
        mask_weigh_norm = mask_weight.norm(1) / self.n_conditions * 0.0005
        mask_weight = mask_weight.contiguous().view(-1, 1, self.n_conditions, self.hidden_dim)
        masked_embedding = masked_embedding * mask_weight
        #print('mask_embedding:', mask_weight)
        code = self._get_outfit_graph_feat(img_embedding, masked_embedding)
        #score = self._compatibility_score(I, mask)
        #score = torch.div(score, seq_len)
        return code, mask_weigh_norm

    def init_weights(self):
        pass


class UserDisen(nn.Module):
    def __init__(self, param, s):
        super(UserDisen, self).__init__()
        self.num_heads = 4
        self.num_seeds = 5
        self.ln = False
        self.S = s
        self.eb = nn.Sequential(
            A.NPMA(param.dim, self.num_heads, self.num_seeds, self.S, ln=self.ln),
            A.SAB(param.dim, param.dim, self.num_heads, ln=self.ln),
        )

    def forward(self, X):
        return self.eb(X)

    def init_weights(self):
        pass


class OutfitDisen(nn.Module):
    def __init__(self, param, s):
        super(OutfitDisen, self).__init__()
        self.dim = param.dim
        self.num_heads = 4
        self.num_inds = 16
        self.num_seeds = 5
        self.ln = False
        #self.S = nn.Parameter(torch.Tensor(1, self.num_seeds, self.dim))
        #nn.init.normal_(self.S)
        self.S = s
        self.enc = nn.Sequential(
            A.ISAB(self.dim, self.dim, self.num_heads, self.num_inds, ln=self.ln),
            A.ISAB(self.dim, self.dim, self.num_heads, self.num_inds, ln=self.ln),
        )
        self.dec = nn.Sequential(
            A.NPMA(self.dim, self.num_heads, self.num_seeds, self.S, ln=self.ln),
            A.SAB(self.dim, self.dim, self.num_heads, ln=self.ln),
        )

    def forward(self, X):
        return self.dec(self.enc(X))

    def init_weights(self):
        pass


class UserOutfitDisen(nn.Module):
    def __init__(self, param, s):
        super(UserOutfitDisen, self).__init__()
        self.userdisen = UserDisen(param, s)
        self.outfitdisen = OutfitDisen(param, s)

    def forward(self, lcus, out_p, out_n, his_v):
        lcus_k = self.userdisen(lcus)
        out_pk = self.outfitdisen(out_p)
        out_nk = self.outfitdisen(out_n)
        his_vk = [self.outfitdisen(out_v) for out_v in his_v]
        his_k = torch.stack(his_vk, dim=1)
        his = torch.mean(his_k, dim=1)
        return lcus_k, out_pk, out_nk, his

    def init_weights(self):
        pass


class Match(nn.Module):
    def __init__(self, param):
        super(Match, self).__init__()
        self.dim = param.dim
        self.num_heads = 4
        self.ln = False
        self.eb = A.MAB(self.dim, self.dim, self.dim, self.num_heads, ln=self.ln)

    def forward(self, X, Y):
        return self.eb(X, Y)

    def init_weights(self):
        pass

