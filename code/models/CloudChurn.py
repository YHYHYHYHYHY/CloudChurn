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
        # At the very beginning, we tried 2 ways to combine Clf_order and Clf_page together at the same time, 
        # the 2 combined models are denoted  as Clf_1 and Clf_2, the difference appears at the output layer,
        # Clf_1 add the output of Clf_order and Clf_page together with different weights and the weights
        # are learnable parameters, Clf_2 concat the output of Clf_order and Clf_page and use an extra
        # MLP head to get the final results, in the paper CloudChurn apply Clf_2, but we still keeps
        # the code of Clf_1 here, you may try this way to test its performance.
        if configs.args_T.clf_choose == 1:
            self.Clf_1 = Clf_1(configs.args_train_1)
        if configs.args_T.clf_choose == 2:
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
                #pred = self._network([X_order, X_time_series])
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
    
    
    def get_weight_1(self, order, page, label, report=False):
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        pred = self.Clf_1([order, page]).argmax(1)
        if report:
            print(classification_report(label, pred, digits=3))
            return
        
        y_pred = (pred == label)
        
        
        # ------ Generate Weight List ------------------
        weight_list = []
        error = 0
        for i in range(y_pred.shape[0]):
            if y_pred[i] == False :
                error += 1
        error_rate = error / y_pred.shape[0] # proportion of wrongly classified samples in the entire training set
        alpha = math.log((1 - error_rate) / (error_rate + 0.001)) / 2
        self.alpha_1 = alpha

        exp_sum = 0
        for i in range(y_pred.shape[0]):
            if y_pred[i] == False :
                exp_sum += math.exp(alpha)
            else:
                exp_sum += math.exp(-alpha)

        for i in range(y_pred.shape[0]):
            if y_pred[i] == False :
                w = math.exp(alpha) / exp_sum
            else:
                w = math.exp(-alpha) / exp_sum
            weight_list.append(w)
        
        self.weight_list_1 = weight_list
        
    def get_weight_2(self, order, page, label, report=False):
        loss_func = torch.nn.CrossEntropyLoss(reduce='sum')
        pred = self.Clf_2([order, page]).argmax(1)
        if report:
            print(classification_report(label, pred, digits=3))
            return
        
        y_pred = (pred == label)
        
        # ------ Generate Weight List ------------------
        weight_list = []
        error = 0
        
        for i in range(y_pred.shape[0]):
            if y_pred[i] == False :
                error += 1
        error_rate = error / y_pred.shape[0] # proportion of wrongly classified samples in the entire training set
        alpha = math.log((1 - error_rate) / (error_rate + 0.001)) / 2
        self.alpha_2 = alpha

        exp_sum = 0
        for i in range(y_pred.shape[0]):
            if y_pred[i] == False :
                exp_sum += math.exp(alpha)
            else:
                exp_sum += math.exp(-alpha)

        for i in range(y_pred.shape[0]):
            if y_pred[i] == False :
                w = math.exp(alpha) / exp_sum
            else:
                w = math.exp(-alpha) / exp_sum
            weight_list.append(w)
        
        self.weight_list_2 = weight_list
    
    
    
    
    def fit(self, metric, label):
        index = torch.tensor(np.arange(len(metric)))
        val_rate = self.args_T.val_rate
        val_size = int(val_rate * len(label))
        metric_val = metric[:val_size]
        metric = metric[val_size:]
        label_val = label[:val_size]
        label = label[val_size:]
        train_size = len(metric)
        index_val = index[:val_size]
        index_train = index[val_size:]
        
        dataset = MetricDataset(metric, index_train, label)
        dataloader = DataLoader(dataset, batch_size=self.args_T.batch_size, drop_last=False, shuffle=True)
        dataset_val = MetricDataset(metric_val, index_val, label_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.args_T.batch_size, drop_last=False, shuffle=True)
        
        size = len(dataloader.dataset)
        loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        optimizer = optim.Adam(self.Encoder.parameters(), lr=self.args_T.lr)
        
        if self.args_T.clf_choose == 1:
            weight_list = self.weight_list_1
        elif self.args_T.clf_choose == 2:
            weight_list = self.weight_list_2
        
        for epoch in range(self.args_T.epochs):
            self.Encoder.train()
            train_loss, correct = 0, 0
            for batch, (X_metric, index, y) in enumerate(dataloader):
                pred = self.Encoder(X_metric)
                weight = torch.zeros((index.shape[0]))
                for i in range(index.shape[0]):
                    weight[i] = weight_list[index[i]] 
                    
                    
                loss = loss_func(pred, y) * weight * size
                loss = loss.sum()
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
            for batch, (X_metric, index, y) in enumerate(dataloader_val):
                pred = self.Encoder(X_metric)
                weight = torch.zeros((index.shape[0]))
                for i in range(index.shape[0]):
                    weight[i] = weight_list[index[i]] 
                    
                loss = loss_func(pred, y) * weight * size
                loss = loss.sum()
                val_loss += loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= val_size
            correct /= val_size
            if self.verbose:
                tqdm.write(f"Val:  Accuracy: {correct:>8f}, Avg loss: {val_loss:>8f}") 
        
        # ----------- GetWeight ----------------
        pred = self.Encoder(metric).argmax(1)
        y_pred = (pred == label)
        error = 0
        for i in range(y_pred.shape[0]):
            if y_pred[i] == False :
                error += weight_list[i]
        error_rate = error  # sum of the weights of wrongly classified samples in the entire training set
        
        alpha = math.log((1 - error_rate) / (error_rate + 0.001)) / 2
        if self.args_T.clf_choose == 1:
            weight_delta = self.alpha_1 - alpha
        if self.args_T.clf_choose == 2:
            weight_delta = self.alpha_2 - alpha
        alpha += self.weight_decay * weight_delta
        self.alpha_T = alpha
        
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
        optimizer = optim.Adam(self.Encoder.parameters(), lr=self.args_finetune.lr)
        
        if self.args_T.clf_choose == 1:
            alpha_Clf = self.alpha_1
        elif self.args_T.clf_choose == 2:
            alpha_Clf = self.alpha_2
        
        
        for epoch in range(self.args_finetune.epochs):
            self.Encoder.train()
            if self.args_T.clf_choose == 1:
                self.Clf_1.train()
            elif self.args_T.clf_choose == 2:
                self.Clf_2.train()
            train_loss, correct = 0, 0
            for batch, (X_order, X_page, X_metric, y) in enumerate(dataloader):
                pred_Encoder = self.Encoder(X_metric)
                if self.args_T.clf_choose == 1:
                    pred_Clf = self.Clf_1([X_order, X_page])
                elif self.args_T.clf_choose == 2:
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
            if self.args_T.clf_choose == 1:
                self.Clf_1.eval()
            elif self.args_T.clf_choose == 2:
                self.Clf_2.eval()
            val_loss, correct = 0, 0
            for batch, (X_order, X_page, X_metric, y) in enumerate(dataloader_val):
                pred_Encoder = self.Encoder(X_metric)
                if self.args_T.clf_choose == 1:
                    pred_Clf = self.Clf_1([X_order, X_page])
                elif self.args_T.clf_choose == 2:
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
        print(classification_report(label.detach().numpy(), pred.detach().numpy(), digits=3))
        
    def predict_order(self, order, label):
        if self.args_T.clf_choose == 1:
            pred = self.Clf_1.clf_order(order)
        elif self.args_T.clf_choose == 2:
            pred = self.Clf_2.clf_order(order)
        pred = torch.argmax(pred, dim=1)
        print(classification_report(label.detach().numpy(), pred.detach().numpy(), digits=3))
    
    def predict_page(self, page, label):
        if self.args_T.clf_choose == 1:
            pred = self.Clf_1.clf_page(page)
        elif self.args_T.clf_choose == 2:
            pred = self.Clf_2.clf_page(page)
        pred = torch.argmax(pred, dim=1)
        print(classification_report(label.detach().numpy(), pred.detach().numpy(), digits=3))
    
    def predict(self, order, page, metric, label):
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
        print("weight_Clf:", alpha_Clf, " weight_Encoder:", self.alpha_T)
        pred = torch.argmax(self.alpha_T * pred_Encoder + alpha_Clf * pred_Clf, dim=1)
        print(classification_report(label.detach().numpy(), pred.detach().numpy(), digits=3))
    
    def predict_report(self, order, page, metric, label):
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
        pred = torch.argmax(self.alpha_T * pred_Encoder + alpha_Clf * pred_Clf, dim=1)
        return classification_report(label.detach().numpy(), pred.detach().numpy(), digits=3)
    
    def pred(self, order, page, metric):
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
        pred = torch.argmax(self.alpha_T * pred_Encoder + alpha_Clf * pred_Clf, dim=1)
        return pred
    
    def model_train(self, order, page, metric, label):
        if self.args_T.clf_choose == 1:
            self.Clf_train_1(order, page, label)
            self.get_weight_1(order, page, label)
            self.fit(metric, label)
            self.fine_tune(order, page, metric, label)
        elif self.args_T.clf_choose == 2:
            self.Clf_train_2(order, page, label)
            self.get_weight_2(order, page, label)
            self.fit(metric, label)
            self.fine_tune(order, page, metric, label)
        else:
            raise ValueError("Only 2 classifiers to be chosen")
            
            
    def model_test(self, order, page, metric, label):
        if self.args_T.clf_choose == 1:
            '''
            print("Clf_order results:")
            self.predict_order(order, label)
            print("Clf_page results:")
            self.predict_page(page, label)
            '''
            print("Clf_1 results:")
            self.get_weight_1(order, page, label, report=True)
            print("Encoder results:")
            self.predict_metric(metric, label)
            print("CloudChurn results:")
            self.predict(order, page, metric, label)
        elif self.args_T.clf_choose == 2:
            '''
            print("Clf_order results:")
            self.predict_order(order, label)
            print("Clf_page results:")
            self.predict_page(page, label)
            '''
            print("Clf_2 results:")
            self.get_weight_2(order, page, label, report=True)
            print("Encoder results:")
            self.predict_metric(metric, label)
            print("CloudChurn results:")
            self.predict(order, page, metric, label)
        else:
            raise ValueError("Only 2 classifiers to be chosen")
    
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