import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForwardLayer(nn.Module):
    def __init__(self, d_in, d_model):
        super(FeedForwardLayer, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.relu = nn.ReLU()
        self.Linear1 = nn.Linear(self.d_in, self.d_model)
        self.Linear2 = nn.Linear(self.d_model, self.d_in)
    def forward(self, x):
        _x = x.clone()
        x = self.relu(self.Linear1(x))
        x = self.relu(self.Linear2(x))
        x = x + _x
        return x

class Clf_page(nn.Module):
    def __init__(self, configs):
        super(Clf_page, self).__init__()
        self.d_in = configs.d_in
        self.d_emb = configs.d_emb
        self.d_model = configs.d_model
        self.d_tar = configs.d_tar
        self.embedding = nn.Linear(self.d_in, self.d_emb)
        self.Linear = nn.Linear(self.d_emb * 2, self.d_tar)
        self.layer_num = 3
        
        self.layers = nn.ModuleList([FeedForwardLayer(d_in=self.d_emb, d_model=self.d_model)
                                     for _ in range(self.layer_num)])
        
        self.relu = nn.ReLU()

    


    def forward(self, x):
        x = x.contiguous().view(x.shape[0], 2, -1)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.Linear(x)
        
        return x
    
    
    
class Clf_order(nn.Module):
    def __init__(self, configs):
        super(Clf_order, self).__init__()
        self.d_in = configs.d_in
        self.d_emb = configs.d_emb
        self.d_model = configs.d_model
        self.d_tar = configs.d_tar
        self.Linear = nn.Linear(self.d_emb, self.d_tar)
        self.layer_num = 3
        
        self.layers = nn.ModuleList([FeedForwardLayer(d_in=self.d_emb, d_model=self.d_model)
                                     for _ in range(self.layer_num)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    


    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.Linear(x)
        return x
        
        

class Clf_1(nn.Module):
    def __init__(self, configs):
        super(Clf_1, self).__init__()
        self.clf_order = Clf_order(configs.args_1)
        self.clf_page = Clf_page(configs.args_2)
        self.w1 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.softmax = nn.Softmax(dim=1)

    


    def forward(self, data):
        order, page = data
        prob1 = self.softmax(self.clf_order(order))
        prob2 = self.softmax(self.clf_page(page))
        x = self.w1 * prob1 + self.w2 * prob2
        
        return x
        
        
class Clf_2(nn.Module):
    def __init__(self, configs):
        super(Clf_2, self).__init__()
        configs.args_1.d_tar = 2
        configs.args_2.d_tar = 2
        self.clf_order = Clf_order(configs.args_1)
        self.clf_page = Clf_page(configs.args_2)
        self.Linear = nn.Linear(configs.args_1.d_tar + configs.args_2.d_tar, 2)
        self.sigmoid = nn.Sigmoid()

    


    def forward(self, data):
        order, page = data
        vec1 = self.clf_order(order)
        vec2 = self.clf_page(page)
        x = torch.cat((vec1, vec2), 1)
        x = self.Linear(x)
        
        return x