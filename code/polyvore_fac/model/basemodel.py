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
            nn.init.normal_(self.encoder.weight.data, std=1)


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
        nn.init.normal_(self.eb.weight.data, std=0.01)


class FactorBlock(nn.Module):
    def __init__(self, param):
        super(FactorBlock, self).__init__()
        self.dim = param.dim
        #self.num_heads = 4
        #self.num_inds = 16
        #self.num_seeds = 1
        #self.ln = True

        self.trans_w = nn.Linear(param.dim, param.dim, bias=False)
        '''
        self.enc = nn.Sequential(
            A.ISAB(self.dim, self.dim, self.num_heads, self.num_inds, ln=self.ln),
            A.ISAB(self.dim, self.dim, self.num_heads, self.num_inds, ln=self.ln),
        )
        self.pma = A.PMA(self.dim, self.num_heads, self.num_seeds, ln=self.ln)
        self.sab = nn.Identity()
        '''

    def forward(self, X):
        item_f = self.trans_w(X)
        #item_f_enc = self.enc(item_f)
        #out_f = item_f_enc.mean(axis=1)
        #out_f = self.sab(self.pma(item_f_enc)).squeeze(1)
        #print(out_f.size())
        #assert (0)
        return item_f


class UserDisen(nn.Module):
    def __init__(self, param, ln=False):
        super(UserDisen, self).__init__()
        self.dim = param.dim
        self.num_heads = 4
        self.num_seeds = param.num_seeds
        self.ln = False
        self.branches = self._make_branch(self.num_seeds, param)

    @staticmethod
    def _make_branch(num_seeds, param):
        branches = nn.ModuleList()
        for branch in range(num_seeds):
            branches.append(FactorBlock(param))
        return branches

    def forward(self, X):
        user_f = []
        for branch in self.branches:
            user = branch(X)
            user_f.append(user)

        x = torch.cat(user_f, dim=1)
        #print(x.size())
        mat_feats = F.normalize(x, p=2, dim=-1)
        diversity_mat = torch.bmm(mat_feats, mat_feats.transpose(1, 2))
        eye_mat = torch.eye(diversity_mat.size(-1)).unsqueeze(0).repeat(diversity_mat.size(0), 1, 1).to(
            diversity_mat.device)
        diversity_loss = torch.pow(eye_mat - diversity_mat, exponent=2).view(
            -1, self.num_seeds*self.num_seeds).sum(dim=-1, keepdim=True)
        #print("diversity:", diversity_loss.size())

        return x, diversity_loss

    def init_weights(self):
        pass


class OutfitDisen(nn.Module):
    def __init__(self, param):
        super(OutfitDisen, self).__init__()
        self.num_seeds = param.num_seeds
        self.branches = self._make_branch(self.num_seeds, param)
        self.dim = param.dim
        self.num_heads = 4
        self.num_inds = 16
        self.seedvec = 1
        self.ln = True

        self.enc = nn.Sequential(
            A.ISAB(self.dim, self.dim, self.num_heads, self.num_inds, ln=self.ln),
            A.ISAB(self.dim, self.dim, self.num_heads, self.num_inds, ln=self.ln),
        )
        self.pma = A.PMA(self.dim, self.num_heads, self.seedvec, ln=self.ln)
        self.sab = nn.Identity()

    @staticmethod
    def _make_branch(num_seeds, param):
        branches = nn.ModuleList()
        for branch in range(num_seeds):
            branches.append(FactorBlock(param))
        return branches

    def forward(self, X):
        item_f, out_f = [], []
        for branch in self.branches:
            item = branch(X)
            item_f.append(item)

        mat_feats = F.normalize(torch.stack(item_f, dim=2), p=2, dim=-1)
        diversity_mat = torch.matmul(mat_feats, mat_feats.transpose(2, 3))
        eye_mat = torch.eye(diversity_mat.size(-1)).unsqueeze(0).unsqueeze(0).repeat(
            diversity_mat.size(0),diversity_mat.size(1),1,1).to(diversity_mat.device)
        diversity_loss = torch.pow(eye_mat-diversity_mat, exponent=2).view(
            -1, eye_mat.size(1)*self.num_seeds*self.num_seeds).sum(dim=-1, keepdim=True)

        for item in item_f:
            out = self.sab(self.pma(self.enc(item))).squeeze(1)
            out_f.append(out)

        x = torch.stack(out_f, dim=1)

        #item_feat = F.normalize(torch.stack(item_f, dim=1), p=2, dim=-1)

        item_feat = torch.stack(item_f, dim=1)

        size = item_feat.size(2)
        indx, indy = np.triu_indices(size, k=1)

        if size == 4:
            indx = indx[1:]
            indy = indy[1:]

        feat_com = item_feat[:,:,indx] * item_feat[:,:,indy]

        return x, feat_com, diversity_loss

    def init_weights(self):
        pass

'''
class Match(nn.Module):
    def __init__(self, param):
        super(Match, self).__init__()
        self.dim = param.dim
        num_heads = 4
        self.att1 = A.MATT(self.dim, self.dim, self.dim, num_heads)
        self.att2 = A.MATT(self.dim, self.dim, self.dim, num_heads)
        self.att3 = A.MATT(self.dim, self.dim, self.dim, num_heads)

    def forward(self, out, user, his_v):
        user = user.contiguous().view(-1, user.size(1), 128)
        out = out.contiguous().view(-1, out.size(1), 128)
        out_f = self.att1(user, out)
        #print(out)
        #print(user)
        his_f = self.att2(user, his_v)
        out_user = out_f * user
        his_user = his_f * user
        x = self.att3(his_user, out_user)
        #print(user.size())
        #print(out.size())
        #print(out_f.size())
        #print(his_f.size())
        #print(x.size())
        #assert (0)
        return x

    def init_weights(self):
        pass
'''
