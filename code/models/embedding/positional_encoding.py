import torch
from torch import nn
class Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(Positional_Encoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):

        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :]



class Learnable_Positional_Encoding(nn.Module):

    def __init__(self, d_model, seq_len):
        super(Learnable_Positional_Encoding, self).__init__()
        self.emb = nn.Parameter(torch.randn((seq_len, d_model)))

    def forward(self, x):

        return self.emb

