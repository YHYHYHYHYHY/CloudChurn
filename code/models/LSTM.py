
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

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return self.label.shape[0]

class LSTM(nn.Module):

    def __init__(self, input_size=100, hidden_size=200, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.Linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        hidden_cell = (torch.randn((1, 1, self.hidden_size)), torch.randn((1, 1, self.hidden_size)))
        lstm_out, (h_n, h_c) = self.lstm(x, hidden_cell)
        out = self.Linear(lstm_out.view(lstm_out.shape[0], -1))
        out = self.sigmoid(out)
        return out

class LSTMExp():
    def __init__(self):
        self.batch_size = 64
        self.epochs = 30
        self.lr = 1e-3
        self.verbose = False
    def fit(self, data, label):
        
        input_size = data.shape[1]
        self.lstm = LSTM(input_size=input_size, hidden_size=200)
        
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        dataset = MyDataset(data, label)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, drop_last=False, shuffle=True)
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        optimizer = optim.Adam(self.lstm.parameters(), lr=self.lr)
        size = len(dataset)
        for epoch in range(self.epochs):
            self.lstm.train()
            train_loss, correct = 0, 0
            for batch, (X, y) in enumerate(dataloader):
                pred = self.lstm(X)
                    
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
                
    def predict(self, data):
        data = torch.tensor(data, dtype=torch.float32)
        
        pred = self.lstm(data)
        pred = torch.argmax(pred, dim=1)
        return pred
            