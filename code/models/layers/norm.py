"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, seq_len, d_model):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm([seq_len, d_model])


    def forward(self, x):
        out = self.norm(x)

        return out

class BatchNorm(nn.Module):
    def __init__(self, seq_len, d_model):
        super(BatchNorm, self).__init__()
        self.norm = nn.BatchNorm1d(d_model)


    def forward(self, x):
        out = self.norm(x.transpose(1, 2)).transpose(1, 2)

        return out

