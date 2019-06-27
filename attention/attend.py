import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


class AttentionVector(nn.Module):
    """
    Attention vector for a single player
    x is (n_samples, n_actions1, n_actions2, 1)
    """
    def __init__(self, is_2nd_pl=False, is_last=False, is_cuda=False):
        super(AttentionVector, self).__init__()
        self.is_2nd_pl = is_2nd_pl
        self.is_last = is_last

        self.linear_compare = nn.Linear(2, 1)
        self.sharp1 = nn.Parameter(torch.ones(1))
        self.sharp2 = nn.Parameter(torch.ones(1))
        self.threshold1 = nn.Parameter(torch.zeros(1)) # b=0 allows allow vals to pass
        self.threshold2 = nn.Parameter(torch.zeros(1)) # b=0 allows allow vals to pass

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        if is_cuda:
            self.eps = Variable(torch.Tensor([1e-6]).float().cuda(), requires_grad=False)
        else:
            self.eps = Variable(torch.Tensor([1e-6]).float(), requires_grad=False)


    def compare_param(self, x, compare_func):
        N, a1, a2 = x.size()
        # Here w.shape = (f_in, f_out)
        x_copy = x.clone()
        # Apply function pairwise
        x_out_test = x_copy.repeat(1, x.size()[1], 1)
        x_out_test = x_out_test.view((N, a1, a1, a2))
        expanded = x[:, :, None, :].expand(N, a1, a1, a2)
        concatenated = torch.cat((x_out_test.unsqueeze(-1), expanded.unsqueeze(-1)), dim=-1)
        x_out_test = compare_func(concatenated).squeeze(-1)

        # Apply activation
        x_out_test = self.sigmoid(x_out_test)
        # Pool
        x_out_test = torch.mean(x_out_test, dim=3)

        # Rescale
        x_out_test = self.max_normalize(x_out_test, dim=2)
        return x_out_test

    def max_normalize(self, x, dim=None):
        ''' Normalization by dividing with the maximum of each action '''
        max_val = torch.max(x, dim=dim)[0]
        max_normalize = torch.add(max_val, self.eps)
        x = torch.div(x, max_normalize.unsqueeze(-1))
        return x

    # accepts (N, a1, a1, a2, 1) shape, can change to f soon
    def compute_player_mask(self, x):
        x = torch.mean(x, dim=2)  #TODO: Consider a richer function class
        # here (exchangeable layer)

        x = self.relu(x * self.sharp2 + self.threshold2)

        x = self.softmax(x)
        if not self.is_last:
            x = self.max_normalize(x, dim=1)

        return x

    def forward(self, x):
        N, n, m, f = x.size()
        if self.is_2nd_pl:
            pl2 = x[..., 1].clone()
            pl2 = pl2.permute(0, 2, 1)
            pl2 = self.compare_param(pl2, self.linear_compare)
            pl2 = self.compute_player_mask(pl2)
            return pl2
        else:
            pl1 = x[..., 0].clone()
            pl1 = self.compare_param(pl1, self.linear_compare)
            pl1 = self.compute_player_mask(pl1)
            return pl1


class AttentionModule(nn.Module):
    """ Module that computes two hidden units, i.e. for both players, across layers """

    def __init__(self, is_simult=False, module_layers=1, is_cuda=False):
        super(AttentionModule, self).__init__()
        self.is_simult = is_simult
        self.module_layers = module_layers
        self.attend_vecs1 = nn.ModuleList([AttentionVector(is_2nd_pl=False, is_cuda=is_cuda)
                                           for i in range(self.module_layers)])
        self.attend_vecs2 = nn.ModuleList([AttentionVector(is_2nd_pl=True, is_cuda=is_cuda)
                                           for i in range(self.module_layers)])

    def forward(self, x, all_masks, att_vecs, training=True):
        for attend_vec1, attend_vec2, i in zip(self.attend_vecs1,\
                                       self.attend_vecs2,\
                                       range(self.module_layers)):
            if self.is_simult:
                vec1 = attend_vec1(x)
                vec2 = attend_vec2(x)
                vec1 = vec1.unsqueeze(-1)
                vec2 = vec2.unsqueeze(1)
                mask = torch.bmm(vec1, vec2)
                x = x * mask.unsqueeze(-1)
            else:
                vec1 = attend_vec1(x)
                x = x * vec1.unsqueeze(-1).unsqueeze(-1)
                vec2 = attend_vec2(x)
                x = x * vec2.unsqueeze(1).unsqueeze(-1)

                vec1 = vec1.unsqueeze(-1)
                vec2 = vec2.unsqueeze(1)
                mask = torch.bmm(vec1, vec2)   # only approximation

        return x, mask, vec1, vec2


class AttentionNet(nn.Module):
    """
    A network of only attention modules.
    """
    def __init__(self, hid_layers=1, hid_units=2, is_simult=False,
                 is_fc_first=True, is_fc_hid=False, with_last=True,
                 is_cuda=False, drop_p=0.):
        super(AttentionNet, self).__init__()
        assert hid_units % 2 == 0 and hid_layers >= 0
        assert drop_p >= 0. and drop_p < 1.
        self.hid_layers = hid_layers
        self.hid_units = hid_units
        self.is_simult = is_simult
        self.is_fc_first = is_fc_first
        self.is_fc_hid = is_fc_hid
        self.is_cuda = is_cuda
        self.with_last = with_last
        self.drop_p = drop_p

        if is_fc_first:
            self.fc = nn.Linear(2, self.hid_units)
            self.relu = nn.ReLU()
            self.sigm = nn.Sigmoid()
            self.lrelu = nn.LeakyReLU(negative_slope=0.01)
        if drop_p > 0.:
            self.hid_dropout_layers = nn.ModuleList([nn.Dropout2d(self.drop_p)
                                                    for l in range(self.hid_layers)])
            self.dropout_layer_in = nn.Dropout2d(self.drop_p)
        self.att_modules = nn.ModuleList([nn.ModuleList(
                                          [AttentionModule(is_cuda=is_cuda)
                                                for i in range(self.hid_units/2)])
                                          for l in range(self.hid_layers)])
        if is_fc_hid:
            self.hid_fc = nn.ModuleList([nn.Linear(self.hid_units, self.hid_units)
                                         for l in range(self.hid_layers)])
        if self.with_last:
            self.out_attend_vec = AttentionVector(is_2nd_pl=False, is_last=True,\
                                                     is_cuda=is_cuda)
            self.softmax = nn.Softmax(dim=1)

        # only in eval() mode
        self.all_masks = []
        self.att_vecs = {0: [], 1: []}
        self.out_att_vec = []


    def forward(self, x):
        x = x.transpose(1, 3)
        x = x.transpose(1, 2)
        N, a1, a2, f = x.size()
        if not self.training:
            self.all_masks = []
            self.att_vecs = {0: [], 1: []}
            self.out_att_vec = []

        if self.is_fc_first:
            x = self.fc(x)
            x = self.relu(x)
            if self.drop_p > 0.:
                x = x.permute(0, 3, 1, 2)
                x = self.dropout_layer_in(x).permute(0, 2, 3, 1)

        for l in range(self.hid_layers):
            x, mask, att_vec1, att_vec2 =\
                    self.att_modules[l][0](x, self.all_masks, self.att_vecs,\
                                            training=self.training)
            if self.is_fc_hid:
                x = self.hid_fc[l](x)
                x = self.relu(x)
            if not self.training:
                self.all_masks.append(mask)
                self.att_vecs[0].append(att_vec1)
                self.att_vecs[1].append(att_vec2)

#        for l in range(self.hid_layers):
#            x_cloned = x.clone()
#            for i in range(0, self.hid_units, 2):    # i represents a feature
#                masked_unit = self.att_modules[l][i/2](x[..., i:i+2], self.all_masks,\
#                                              self.att_vecs, training=self.training)
#                x_cloned[..., i:i+2] = masked_unit
#            x = x_cloned.clone()
#            if self.is_fc_hid:
#                x = self.hid_fc[l](x)
#                x = self.relu(x)
#                if self.drop_p > 0.:
#                    x = x.permute(0, 3, 1, 2)
#                    x = self.hid_dropout_layers[l](x).permute(0, 2, 3, 1)

        if self.with_last:
            x = self.out_attend_vec(x)
        out = x
        if not self.training and self.with_last:
            self.out_att_vec = out.clone()

        return out
