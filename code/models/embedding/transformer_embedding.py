from torch import nn
from models.embedding.positional_encoding import Positional_Encoding, Learnable_Positional_Encoding
from models.embedding.token_embeddings import Token_Embedding
class TransformerEmbedding(nn.Module):
    

    def __init__(self, enc_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = Token_Embedding(enc_size, d_model)
        #self.pos_emb = Positional_Encoding(d_model, max_len, device)
        self.pos_emb = Learnable_Positional_Encoding(d_model, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
