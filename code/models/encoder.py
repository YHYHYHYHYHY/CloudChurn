"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import torch

from blocks.encoder_layer import EncoderLayer
from embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=args.d_model,
                                        max_len=args.seq_len,
                                        enc_size=args.enc_size,
                                        drop_prob=args.dropout,
                                        device='cpu')

        self.layers = nn.ModuleList([EncoderLayer(seq_len=args.seq_len,
                                                  d_model=args.d_model,
                                                  ffn_hidden=args.ffn_hidden,
                                                  n_head=args.n_head,
                                                  drop_prob=args.dropout)
                                     for _ in range(args.e_layers)])
        
        self.head = nn.Linear(args.d_model * args.seq_len, 2)
    def forward(self, x):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.head(x)
        
        
        return x
    
