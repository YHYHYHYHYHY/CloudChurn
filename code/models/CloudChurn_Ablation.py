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

from clf import Clf_page, Clf_order, Clf_1, Clf_2
from encoder import Encoder







class SingleDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return self.label.shape[0]

class ClfDataset(Dataset):
    def __init__(self, order, page, label):
        self.order = order
        self.page = page
        self.label = label
        
        
    def __getitem__(self, index):
        return self.order[index], self.page[index], self.label[index]
    
    def __len__(self):
        return self.label.shape[0]

class MetricDataset(Dataset):
    def __init__(self, data, id, label):
        self.data = data
        self.id = id
        self.label = label
        
    def __getitem__(self, index):
        return self.data[index], self.id[index], self.label[index]
    
    def __len__(self):
        return self.label.shape[0]
    
class TotalDataset(Dataset):
    def __init__(self, order, page, metric, label):
        self.order = order
        self.page = page
        self.metric = metric
        self.label = label
        
        
    def __getitem__(self, index):
        return self.order[index], self.page[index], self.metric[index], self.label[index]
    
    def __len__(self):
        return self.label.shape[0]

class CloudChurn():
    def __init__(
        self,
        configs
    ):
        self.Clf_2 = Clf_2(configs.args_train_2)
        self.Encoder = Encoder(configs.args_T)
        self.Clf_order = Clf_order(configs.args_1)
        self.Clf_page = Clf_page(configs.args_2)
        
        self.args_1 = configs.args_1
        self.args_2 = configs.args_2
        self.args_train_1 = configs.args_train_1
        self.args_train_2 = configs.args_train_2
        self.args_T = configs.args_T
        self.args_finetune = configs.args_finetune
        
        self.verbose = configs.verbose
        self.weight_decay = configs.weight_decay
        
        self.head = nn.Linear(4, 2)
    
    

        
        
    def Clf_train_1(self, order, page, label):
        val_rate = self.args_train_1.val_rate
        val_size = int(val_rate * len(label))
        order_val = order[:val_size]
        order = order[val_size:]
        page_val = page[:val_size]
        page = page[val_size:]
        label_val = label[:val_size]
        label = label[val_size:]
        
        dataset = ClfDataset(order, page, label)
        dataloader = DataLoader(dataset, batch_size=self.args_train_1.batch_size, drop_last=False, shuffle=True)
        
        dataset_val = ClfDataset(order_val, page_val, label_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.args_train_1.batch_size, drop_last=False, shuffle=True)
        
        
        size = len(dataloader.dataset)
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        optimizer = optim.Adam(self.Clf_1.parameters(), lr=self.args_train_1.lr)
        
        for epoch in range(self.args_train_1.epochs):
            self.Clf_1.train()
            train_loss, correct = 0, 0
            for batch, (X_order, X_page, y) in enumerate(dataloader):
                pred = self.Clf_1([X_order, X_page])
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
            self.Clf_1.eval()
            val_loss, correct = 0, 0
            for batch, (X_order, X_page, y) in enumerate(dataloader_val):
                pred = self.Clf_1([X_order, X_page])
                loss = loss_func(pred, y)
                
                val_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= val_size
            correct /= val_size
            if self.verbose:
                tqdm.write(f"Val:  Accuracy: {correct:>8f}, Avg loss: {val_loss:>8f}") 
            
            
            
            
    def Clf_train_2(self, order, page, label):
        val_rate = self.args_train_2.val_rate
        val_size = int(val_rate * len(label))
        order_val = order[:val_size]
        order = order[val_size:]
        page_val = page[:val_size]
        page = page[val_size:]
        label_val = label[:val_size]
        label = label[val_size:]
        
        dataset = ClfDataset(order, page, label)
        dataloader = DataLoader(dataset, batch_size=self.args_train_2.batch_size, drop_last=False, shuffle=True)
        
        dataset_val = ClfDataset(order_val, page_val, label_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.args_train_2.batch_size, drop_last=False, shuffle=True)
        
        
        size = len(dataloader.dataset)
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        optimizer = optim.Adam(self.Clf_2.parameters(), lr=self.args_train_2.lr)
        
        for epoch in range(self.args_train_2.epochs):
            self.Clf_2.train()
            train_loss, correct = 0, 0
            for batch, (X_order, X_page, y) in enumerate(dataloader):
                #pred = self._network([X_order, X_time_series])
                pred = self.Clf_2([X_order, X_page])
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
            self.Clf_2.eval()
            val_loss, correct = 0, 0
            for batch, (X_order, X_page, y) in enumerate(dataloader_val):
                pred = self.Clf_2([X_order, X_page])
                loss = loss_func(pred, y)
                
                val_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= val_size
            correct /= val_size
            if self.verbose:
                tqdm.write(f"Val:  Accuracy: {correct:>8f}, Avg loss: {val_loss:>8f}") 
    
    def train_order(self, order, label): # Train Clf_order only, use Clf_order instead of Clf_1 or Clf_2
        val_rate = self.args_1.val_rate
        val_size = int(val_rate * len(label))
        order_val = order[:val_size]
        order = order[val_size:]
        label_val = label[:val_size]
        label = label[val_size:]
        
        dataset = SingleDataset(order, label)
        dataloader = DataLoader(dataset, batch_size=self.args_1.batch_size, drop_last=False, shuffle=True)
        
        dataset_val = SingleDataset(order_val, label_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.args_1.batch_size, drop_last=False, shuffle=True)
        
        
        size = len(dataloader.dataset)
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        optimizer = optim.Adam(self.Clf_order.parameters(), lr=self.args_1.lr)
        
        for epoch in range(self.args_1.epochs):
            self.Clf_order.train()
            train_loss, correct = 0, 0
            for batch, (X_order, y) in enumerate(dataloader):
                pred = self.Clf_order(X_order)
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
            self.Clf_order.eval()
            val_loss, correct = 0, 0
            for batch, (X_order, y) in enumerate(dataloader_val):
                pred = self.Clf_order(X_order)
                loss = loss_func(pred, y)
                
                val_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= val_size
            correct /= val_size
            if self.verbose:
                tqdm.write(f"Val:  Accuracy: {correct:>8f}, Avg loss: {val_loss:>8f}") 
        
    
    def train_page(self, order, label): # Train Clf_page only, use Clf_order instead of Clf_1 or Clf_2
        val_rate = 0.1
        val_size = int(val_rate * len(label))
        order_val = order[:val_size]
        order = order[val_size:]
        label_val = label[:val_size]
        label = label[val_size:]
        
        dataset = SingleDataset(order, label)
        dataloader = DataLoader(dataset, batch_size=self.args_1.batch_size, drop_last=False, shuffle=True)
        
        dataset_val = SingleDataset(order_val, label_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.args_1.batch_size, drop_last=False, shuffle=True)
        
        
        size = len(dataloader.dataset)
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        optimizer = optim.Adam(self.Clf_order.parameters(), lr=self.args_1.lr)
        
        for epoch in range(self.args_1.epochs):
            self.Clf_page.train()
            train_loss, correct = 0, 0
            for batch, (X_order, y) in enumerate(dataloader):
                pred = self.Clf_page(X_order)
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
            self.Clf_page.eval()
            val_loss, correct = 0, 0
            for batch, (X_order, y) in enumerate(dataloader_val):
                pred = self.Clf_page(X_order)
                loss = loss_func(pred, y)
                
                val_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= val_size
            correct /= val_size
            if self.verbose:
                tqdm.write(f"Val:  Accuracy: {correct:>8f}, Avg loss: {val_loss:>8f}") 
                
    def train_metric(self, metric, label):
        val_rate = self.args_T.val_rate
        val_size = int(val_rate * len(label))
        metric_val = metric[:val_size]
        metric = metric[val_size:]
        label_val = label[:val_size]
        label = label[val_size:]
        train_size = len(metric)
        
        dataset = SingleDataset(metric, label)
        dataloader = DataLoader(dataset, batch_size=self.args_T.batch_size, drop_last=False, shuffle=True)
        dataset_val = SingleDataset(metric_val, label_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.args_T.batch_size, drop_last=False, shuffle=True)
        
        size = len(dataloader.dataset)
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        optimizer = optim.Adam(self.Encoder.parameters(), lr=self.args_T.lr)
        
        for epoch in range(self.args_T.epochs):
            self.Encoder.train()
            train_loss, correct = 0, 0
            for batch, (X_metric, y) in enumerate(dataloader):
                pred = self.Encoder(X_metric)
                    
                    
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
            self.Encoder.eval()
            val_loss, correct = 0, 0
            for batch, (X_metric, y) in enumerate(dataloader_val):
                pred = self.Encoder(X_metric)
                    
                loss = loss_func(pred, y)
                
                train_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= val_size
            correct /= val_size
            if self.verbose:
                tqdm.write(f"Val:  Accuracy: {correct:>8f}, Avg loss: {val_loss:>8f}") 
        
        
        
    def fine_tune(self, order, page, metric, label):
        val_rate = self.args_finetune.val_rate
        val_size = int(val_rate * len(label))
        metric_val = metric[:val_size]
        metric = metric[val_size:]
        order_val = order[:val_size]
        order = order[val_size:]
        page_val = page[:val_size]
        page = page[val_size:]
        label_val = label[:val_size]
        label = label[val_size:]
        train_size = len(metric)
        
        dataset = TotalDataset(order, page, metric, label)
        dataloader = DataLoader(dataset, batch_size=self.args_finetune.batch_size, drop_last=False, shuffle=True)
        dataset_val = TotalDataset(order_val, page_val, metric_val, label_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.args_finetune.batch_size, drop_last=False, shuffle=True)
        
        size = len(dataloader.dataset)
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        optimizer = optim.Adam([{'params':self.Encoder.parameters()}, {'params':self.Clf_2.parameters()}], lr=self.args_finetune.lr)
        
        
        alpha_Clf = 1
        self.alpha_T = 1
        
        
        
        for epoch in range(self.args_finetune.epochs):
            self.Encoder.train()
            self.Clf_2.train()
            train_loss, correct = 0, 0
            for batch, (X_order, X_page, X_metric, y) in enumerate(dataloader):
                pred_Encoder = self.Encoder(X_metric)
                pred_Clf = self.Clf_2([X_order, X_page])
                
                pred = self.alpha_T * pred_Encoder + alpha_Clf * pred_Clf
                    
                    
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
            self.Encoder.eval()
            self.Clf_2.eval()
            val_loss, correct = 0, 0
            for batch, (X_order, X_page, X_metric, y) in enumerate(dataloader_val):
                pred_Encoder = self.Encoder(X_metric)
                pred_Clf = self.Clf_2([X_order, X_page])
                
                pred = self.alpha_T * pred_Encoder + alpha_Clf * pred_Clf
                    
                loss = loss_func(pred, y) 
                val_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= val_size
            correct /= val_size
            if self.verbose:
                tqdm.write(f"Val:  Accuracy: {correct:>8f}, Avg loss: {val_loss:>8f}") 
        
        
        
    def train_concat(self, order, page, metric, label):
        val_rate = self.args_finetune.val_rate
        val_size = int(val_rate * len(label))
        metric_val = metric[:val_size]
        metric = metric[val_size:]
        order_val = order[:val_size]
        order = order[val_size:]
        page_val = page[:val_size]
        page = page[val_size:]
        label_val = label[:val_size]
        label = label[val_size:]
        train_size = len(metric)
        
        dataset = TotalDataset(order, page, metric, label)
        dataloader = DataLoader(dataset, batch_size=self.args_finetune.batch_size, drop_last=False, shuffle=True)
        dataset_val = TotalDataset(order_val, page_val, metric_val, label_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.args_finetune.batch_size, drop_last=False, shuffle=True)
        
        size = len(dataloader.dataset)
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        optimizer = optim.Adam([{'params':self.Encoder.parameters()}, {'params':self.Clf_2.parameters()}, {'params':self.head.parameters()}], lr=self.args_finetune.lr)
        
        
        
        
        
        for epoch in range(self.args_finetune.epochs):
            self.Encoder.train()
            self.Clf_2.train()
            train_loss, correct = 0, 0
            for batch, (X_order, X_page, X_metric, y) in enumerate(dataloader):
                pred_Encoder = self.Encoder(X_metric)
                pred_Clf = self.Clf_2([X_order, X_page])
                
                pred = torch.hstack(pred_Encoder, pred_Clf)
                pred = self.head(pred)
                    
                    
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
            self.Encoder.eval()
            self.Clf_2.eval()
            val_loss, correct = 0, 0
            for batch, (X_order, X_page, X_metric, y) in enumerate(dataloader_val):
                pred_Encoder = self.Encoder(X_metric)
                pred_Clf = self.Clf_2([X_order, X_page])
                
                pred = self.alpha_T * pred_Encoder + alpha_Clf * pred_Clf
                    
                loss = loss_func(pred, y) 
                val_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= val_size
            correct /= val_size
            if self.verbose:
                tqdm.write(f"Val:  Accuracy: {correct:>8f}, Avg loss: {val_loss:>8f}") 
        
    def predict_metric(self, metric, label):
        pred = self.Encoder(metric)
        pred = torch.argmax(pred, dim=1)
        return pred.detach().numpy()
        
    def predict_order(self, order, label):
        pred = self.Clf_order(order)
        pred = torch.argmax(pred, dim=1)
        return pred.detach().numpy()
    
    def predict_page(self, page, label):
        pred = self.Clf_page(page)
        pred = torch.argmax(pred, dim=1)
        return pred.detach().numpy()
    
    def predict_clf2(self, order, page, label):
        pred = self.Clf_2([order, page])
        pred = torch.argmax(pred, dim=1)
        return pred.detach().numpy()
    
    def predict(self, order, page, metric, label):
        self.Clf_2.eval()
        self.Clf_order.eval()
        self.Encoder.eval()
        pred_Encoder = torch.softmax(self.Encoder(metric), dim=1)
        pred_Clf = self.Clf_2([order, page])
        pred_Clf = torch.softmax(pred_Clf, dim=1)
        alpha_Clf = 1
        self.alpha_T = 1
        
        
        pred = torch.argmax(self.alpha_T * pred_Encoder + alpha_Clf * pred_Clf, dim=1)
        return pred.detach().numpy()
    
   
    
        
    def get_metric(self, order, page, metric, label):
        if self.args_T.clf_choose == 1:
            self.Clf_1.eval()
        if self.args_T.clf_choose == 2:
            self.Clf_2.eval()
        self.Clf_order.eval()
        self.Encoder.eval()
        pred_Encoder = torch.softmax(self.Encoder(metric), dim=1)
        if self.args_T.clf_choose == 1:
            pred_Clf = self.Clf_1([order, page])
            alpha_Clf = self.alpha_1
        elif self.args_T.clf_choose == 2:
            pred_Clf = self.Clf_2([order, page])
            alpha_Clf = self.alpha_2
        pred_Clf = torch.softmax(pred_Clf, dim=1)
            
        pred = torch.argmax(self.alpha_T * pred_Encoder + alpha_Clf * pred_Clf, dim=1).detach().numpy()
        label = label.detach().numpy()
        
        
        pred_Encoder = torch.argmax(pred_Encoder, dim=1).detach().numpy()
        pred_Clf = torch.argmax(pred_Clf, dim=1).detach().numpy()
        
        f1_Encoder = f1_score(label, pred_Encoder)
        f1_Clf = f1_score(label, pred_Clf)
        f1_CloudChurn = f1_score(label, pred)
        
        auc_Encoder = roc_auc_score(label, pred_Encoder)
        auc_Clf = roc_auc_score(label, pred_Clf)
        auc_CloudChurn = roc_auc_score(label, pred)
        
        f1_list = [f1_Encoder, f1_Clf, f1_CloudChurn]
        auc_list = [auc_Encoder, auc_Clf, auc_CloudChurn]
        return f1_list, auc_list