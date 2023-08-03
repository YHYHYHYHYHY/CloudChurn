from torch import nn
class Token_Embedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(Token_Embedding, self).__init__()
        self.Linear1 = nn.Linear(vocab_size, d_model)
    def forward(self, x):
        out = self.Linear1(x)
        return out
