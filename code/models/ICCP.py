
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import math
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score

from sklearn.decomposition import PCA
from Components import Transformer_Layer, Classification_MLP, Lightweight_MLP
import numpy as np




class MyDataset(Dataset):
    def __init__(self, data_reduced, seq, label):
        self.data_reduced = data_reduced
        self.seq = seq
        self.label = label
        
    def __getitem__(self, index):
        return self.data_reduced[index], self.seq[index], self.label[index]
    
    def __len__(self):
        return self.label.shape[0]

class ICCP(nn.Module):
    def  __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_seq = configs.d_seq
        self.d_reduced = configs.n_components
        self.d_emb = configs.d_emb
        self.d_light_tar = configs.d_light_tar
        self.emb1 = nn.Linear(self.seq_len, self.d_emb)
        self.emb2 = nn.Linear(self.seq_len, self.d_emb)
        self.emb3 = nn.Linear(self.seq_len, self.d_emb)
        self.emb_Transformer = nn.Linear(3*self.d_emb, configs.d_model)
        
        
        
        self.Lightweight_MLP = Lightweight_MLP(d_in=configs.d_light_in, d_tar=self.d_light_tar)
        self.Transformer_Layer = Transformer_Layer(configs)
        self.Classification_MLP = Classification_MLP(n_layers=3, d_in=self.d_reduced + self.d_light_tar)
        
    def forward(self, x_reduced, seq):
        seq = seq.transpose(1, 2)
        random_vec = torch.randn((seq.shape[0], seq.shape[1], self.seq_len))
        seq = seq + random_vec
        seq_vec = self.emb1(seq)
        pos_vec = self.emb2(seq)
        learnable_vec = self.emb3(seq)
        embed_vec = torch.cat((seq_vec, pos_vec, learnable_vec), dim=2)
        embed_vec = self.emb_Transformer(embed_vec)
        
        seq_out = self.Transformer_Layer(embed_vec)
        seq_out = self.Lightweight_MLP(seq_out)
        vec = torch.cat((x_reduced, seq_out), dim=1)
        out = self.Classification_MLP(vec)
        return out
    
class Exp():
    def __init__(self, configs):
        self.ICCP = ICCP(configs)
        self.pca = PCA(n_components=configs.n_components)
        
        self.val_rate = configs.val_rate
        self.batch_size = configs.batch_size
        self.lr = configs.lr
        self.epochs = configs.epochs
        self.verbose = configs.verbose
        
    def train(self, port, seq, label):
        data = np.hstack((port, seq.reshape(seq.shape[0], -1)))
        data_reduced = self.pca.fit_transform(data)
        
        data_reduced = torch.tensor(data_reduced, dtype=torch.float32)
        seq = torch.tensor(seq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        val_rate = self.val_rate
        val_size = int(val_rate * len(label))
        data_reduced_val = data_reduced[:val_size]
        data_reduced = data_reduced[val_size:]
        seq_val = seq[:val_size]
        seq = seq[val_size:]
        label_val = label[:val_size]
        label = label[val_size:]
        
        dataset = MyDataset(data_reduced, seq, label)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, drop_last=False, shuffle=True)
        dataset_val = MyDataset(data_reduced_val, seq_val, label_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, drop_last=False, shuffle=True)
        
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        optimizer = optim.Adam(self.ICCP.parameters(), lr=self.lr)
        
        size = label.shape[0]
        val_size = label_val.shape[0]
        for epoch in range(self.epochs):
            self.ICCP.train()
            train_loss, correct = 0, 0
            for batch, (X_reduced, seq, y) in enumerate(dataloader):
                pred = self.ICCP(X_reduced, seq)
                    
                loss = loss_func(pred, y) 
                train_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss /= size
            correct /= size
            if self.verbose:
                tqdm.write(f"Epoch: {epoch + 1}, : Accuracy: {correct:>8f}, Avg loss: {train_loss:>8f}", end='           ') 
            
            # ----- Validation -------------------
            self.ICCP.eval()
            val_loss, correct = 0, 0
            for batch, (X_reduced, seq, y) in enumerate(dataloader_val):
                pred = self.ICCP(X_reduced, seq)
                    
                loss = loss_func(pred, y)
                val_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= val_size
            correct /= val_size
            if self.verbose:
                tqdm.write(f"Val:  Accuracy: {correct:>8f}, Avg loss: {val_loss:>8f}") 
        
    def predict(self, port, seq, label):
        data = np.hstack((port, seq.reshape(seq.shape[0], -1)))
        data_reduced = self.pca.transform(data)
        
        data_reduced = torch.tensor(data_reduced, dtype=torch.float32)
        seq = torch.tensor(seq, dtype=torch.float32)
        
        pred = self.ICCP(data_reduced, seq)
        pred = torch.argmax(pred, dim=1)
        return pred.detach().numpy()
        #print(classification_report(label, pred.detach().numpy(), digits=3))
    
    def get_metric(self, port, seq, label):
        data = np.hstack((port, seq.reshape(seq.shape[0], -1)))
        data_reduced = self.pca.transform(data)
        
        data_reduced = torch.tensor(data_reduced, dtype=torch.float32)
        seq = torch.tensor(seq, dtype=torch.float32)
        
        pred = self.ICCP(data_reduced, seq)
        pred = torch.argmax(pred, dim=1)
        
        f1 = f1_score(label, pred)
        auc = roc_auc_score(label, pred)
        
        return f1, auc