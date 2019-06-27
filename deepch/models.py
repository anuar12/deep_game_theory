from __future__ import print_function, absolute_import

import torch.nn as nn

from deepch.layers import MixtureSoftmax

class DeepCH(nn.Module):
    '''
    DeepCH - haven't implemented the action response layer yet...
    '''
    def __init__(self, att_layers, feature_layers, units, reset_weights=True):
        super(DeepCH, self).__init__()
        self.att_layers = att_layers
        self.feature_layers = feature_layers
        self.mixture = MixtureSoftmax(units)
        if reset_weights:
            self.reset_weights()
        
    def forward(self, input):
        state = input
        if self.att_layers:
            state = self.att_layers(input)
            state = state.transpose(1, 3)
            state = state.transpose(2, 3)
        #self.project_parameters()
        return self.mixture(self.feature_layers(state))
        
    def project_parameters(self):
        self.mixture.project_parameters()
    
    def reset_weights(self):
        self.feature_layers.reset_weights()
        self.mixture.reset_weights()
