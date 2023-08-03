

from torch import nn
import torch
from sklearn.decomposition import PCA
from blocks.encoder_layer_ICCP import EncoderLayer


class Transformer_Layer(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(seq_len=args.d_seq,
                                                  d_model=args.d_model,
                                                  ffn_hidden=args.ffn_hidden,
                                                  n_head=args.n_head,
                                                  drop_prob=args.dropout)
                                     for _ in range(args.e_layers)])
        
        self.head = nn.Linear(args.d_model * args.d_seq, args.d_light_in)
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.head(x)
        
        
        return x
    
class Classification_MLP_Layer(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_in)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.Linear = nn.Linear(d_in, d_in)
    def forward(self, x):
        x = self.Linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.norm(x)
        return x
        
class Classification_MLP(nn.Module):
    def __init__(self, n_layers, d_in):
        super().__init__()
        self.layers = nn.ModuleList([Classification_MLP_Layer(d_in=d_in)
                                     for _ in range(n_layers)])
        self.sigmoid = nn.Sigmoid()
        self.Linear = nn.Linear(d_in, 2)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.Linear(x)
        x = self.sigmoid(x)
        return x
        
class Lightweight_MLP(nn.Module):
    def __init__(self,  d_in, d_tar):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.sigmoid = nn.Sigmoid()
        self.Linear = nn.Linear(d_in, d_tar)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.Linear(x)
        x = self.sigmoid(x)
        return x
    

    
        
