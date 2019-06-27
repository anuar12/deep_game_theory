from __future__ import print_function, absolute_import

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from functools import reduce
from operator import mul


def xavier_weight(nin, nout=None, rng=None, dtype="float32"):
    """
    Xavier init
    """
    rng = numpy.random.RandomState(rng)
    if nout is None:
        nout = nin
    r = numpy.sqrt(6.) / numpy.sqrt(nin + nout)
    W = rng.rand(nin, nout) * 2 * r - r
    return numpy.array(W, dtype=dtype)


def normal_weight(nin, nout=None, rng=None, std=0.01, dtype="float32"):
    """
    Normal init
    """
    rng = numpy.random.RandomState(rng)
    W = rng.randn(nin, nout) * std
    return numpy.array(W, dtype=dtype)


class ActionPool(nn.Module):
    '''
    Basic pooling operations. 
    '''
    def __init__(self, axis, function = "mean", expand=True):
        super(ActionPool, self).__init__()
        self.expand = expand
        self._function_name = function
        self._axis_name = axis
        
        if isinstance(axis, str):
            self.dim = {"row":0, "column":1, "both":None}[axis]
        else:
            self.dim = axis
        
        if function == "max":
            self.function = lambda x, dim: torch.max(x, dim=dim)[0]
        elif function == "sum":
            self.function = lambda x, dim: torch.sum(x, dim=dim)    
        elif function == "mean":
            self.function = torch.mean
        else:
            raise ValueError("Unrecognised function: %s" % function)

    def forward(self, input):
        if self.dim is None:
            n, p, i, j = input.size()
            input = input.contiguous()
            reshaped = input.view(n, p, i*j)
            # get the output shapes right
            output = self.function(reshaped, dim=2)
            if self.expand:
                output = output.unsqueeze(2).unsqueeze(2)
        else:
            output = self.function(input, dim=self.dim+2)
            if self.expand:
                output = output.unsqueeze(self.dim+2)
        if self.expand:
            return output.expand_as(input)
        else:
            return output


class MatrixLinear(nn.Linear):
    '''
    Matrix-based linear feed-forward layer. Think of it as the
    matrix analog to a feed-forward layer in an MLP.
    '''
    def forward(self, input):
        n, p, i, j = input.size()
        x = input.permute(0, 2, 3, 1)
        state = input.view((n*i*j, p))
        output = super(MatrixLinear, self).forward(state)
        output = output.view((n, i, j, self.out_features)).permute(0, 3, 1, 2)
        return output


class MatrixLayer(nn.Module):
    '''
    Set layers are linear layers with pooling. Pooling operations
    are defined above.
    '''
    def __init__(self, in_features, out_features, pooling = "max", 
                 axes=["row", "column", "both"], name="", debug=False,
                 dropout=0.):
        super(MatrixLayer, self).__init__()

        # build list of pooling functions
        pool = []
        for axis in axes:
            pool.append(ActionPool(axis, pooling, expand=True))
        self.pool = pool
        self.dropout = dropout
        self.linear = MatrixLinear(in_features * (1+len(pool)), out_features)
        if dropout > 0.:
            self.dropoutlayer = nn.Dropout2d(dropout)
        self.name = name
        self.debug = debug

    def forward(self, input):
        pooled = [p(input) for p in self.pool]
        state = torch.cat([input] + pooled, dim=1)
        if self.dropout > 0.:
            state = self.dropoutlayer(state)
        if self.debug:
            print(self.name, input.size(), state.size())
        return self.linear(state)


class TensorSoftmax(nn.Module):
    '''
    Weighted sum of softmax outputs
    '''
    def __init__(self):
        super(TensorSoftmax, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, input):
        shape = input.size()
        out = self.softmax(input.view(reduce(mul, shape[0:-1]), shape[-1]))
        return out.view(shape)


def project_simplex(x):
    """
    Project an arbitary vector onto the simplex.
    See [Wang & Carreira-Perpin 2013] for a description and references.
    """
    n = x.size()[0]
    mu = torch.sort(x, 0, descending=True)[0]
    
    sm = 0
    for j in xrange(1, n+1):
        sm += mu[j - 1]
        t = mu[j - 1] - (1./(j)) * (sm - 1)
        if t > 0:
            row = j
            sm_row = sm
    theta = (1. / row) * (sm_row - 1)
    y = x - theta
    return torch.clamp(y, min=0.)


def project_parameters(parameters):
    for p in parameters:
        p.data = project_simplex(p.data)


class MixtureSoftmax(nn.Module):
    '''
    '''
    def __init__(self, in_features):
        super(MixtureSoftmax, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_weights()
        
    def reset_weights(self):
        # ensure the weights initialize randomly and sum to one.
        self.weight.data.normal_(10, 1.)
        self.weight.data /= self.weight.data.sum()
        
    def forward(self, input):
        return torch.sum(input * self.weight[None,:,None], dim=1)
    
    def project_parameters(self):
        for p in self.parameters():
            p.data = project_simplex(p.data)


class ARLayer(nn.Module):
    '''
    '''
    def __init__(self, in_features):
        super(ARLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, 1))
        self.reset_weights()
        
    def reset_weights(self):
        # ensure the weights initialize randomly and sum to one.
        self.weight.data.normal_(10, 1.)
        self.weight.data /= self.weight.data.sum()
        
    def forward(self, input):
        raise NotImplementedError()


class FeatureLayers(nn.Module):
    def __init__(self, input_dim, units, functions="mean", activation=F.relu,
                 output_pool="mean", axes=["row", "column", "both"],
                 dropout=0.):
        super(FeatureLayers, self).__init__()
        if isinstance(functions, str):
            functions = [functions] * len(units)
        units = [input_dim] + units
        layers = []
        self.activation = activation
        for i in xrange(1, len(units)):
            layers.append(MatrixLayer(units[i-1], units[i], functions[i-1], 
                                      axes=axes, name="Layer %d" % i, debug=False,
                                      dropout=dropout))
        
        self.preactivations = ActionPool("column", output_pool, expand=False)
        self.softmax = TensorSoftmax()
        
        self.layers = nn.ModuleList(layers)

    def forward(self, input):
        state = input
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            state = layer(state)
            if i == last:
                break
            state = self.activation(state)
        pre = self.preactivations(state)
        return self.softmax(pre)

    def reset_weights(self):
        for p in self.parameters():
            shape = p.data.size()
            if len(shape) > 1:
                p.data = torch.from_numpy(normal_weight(shape[0], shape[1]))

