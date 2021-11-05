import os
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AdamW
import numpy as np
from tqdm import tqdm
import copy
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoConfig
from utils.load_data import load_dataset
from utils.datautils import CollateWraper



class BaseModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.params = copy.deepcopy(params)
        self.device = device
        
    
    def prepare_model(self):
        self.config = AutoConfig.from_pretrained(self.params.lm,num_labels=self.max_label-self.
        min_label+1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.lm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.config.pad_token_id = self.config.eos_token_id
        self.model = AutoModelForSequenceClassification.from_pretrained(self.params.lm, config=self.config).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)

    def prepare_data(self):
        data = load_dataset(self.params.task, create_hash=False,train=0.6, valid=0.2)
        self.trainset = data['train']
        self.validset = data['valid']
        self.testset = data['test']
        self.max_label = max(data['train_dist'].keys())
        self.min_label = min(data['train_dist'].keys())
        self.majority_class = max(data['test_dist'].values())

    def dataloaders(self, iters=None):
        collate_fn = CollateWraper(self.tokenizer, self.min_label)
        train_loader = torch.utils.data.DataLoader(
            self.trainset, collate_fn=collate_fn, batch_size=self.params.batch_size, num_workers=self.params.workers)
        test_loader = torch.utils.data.DataLoader(
            self.testset, collate_fn=collate_fn, batch_size=self.params.batch_size*2, num_workers=self.params.workers,   shuffle=False, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(
            self.validset, collate_fn=collate_fn, batch_size=self.params.batch_size*2, num_workers=self.params.workers,   shuffle=False, drop_last=False)
        return train_loader, valid_loader, test_loader
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def grad_step(self):
        self.optimizer.step()
    
    def compute_loss(self, outputs,labels, labels2):
        loss_functions = self.params.losses.split(';')
        is_labels2 = self.params.labels2
        losses = []
        for loss_function in loss_functions:
            losses.append(self.loss_layer(loss_function, outputs.logits,labels))
            if is_labels2:
                losses.append(self.loss_layer(loss_function, outputs.logits,labels2))
        return sum(losses)/len(losses)
    def loss_layer(self, loss_function, output, target):
        if loss_function=='cce':
            m = nn.CrossEntropyLoss()
            loss = m(output,target)
        elif loss_function=='qwp':
            m = nn.Softmax()
            probs = m(output)
            score = ((torch.arange(self.max_label-self.min_label+1).to(self.device)[None,:]-target[:,None])**2.)/ ((self.max_label-self.min_label)**2.)
            loss = torch.sum(score*probs)/ len(target)
        return loss
    
    def train_step(self, batch):
        self.zero_grad()
        #
        labels = batch['labels']
        del batch['labels']
        if 'labels2' in batch:
            labels2 = batch['labels2']
            del batch['labels2']
        else:
            labels2 = None
        outputs = self.model(**batch)
        #
        loss = self.compute_loss(outputs,labels,labels2)
        # loss = outputs.loss
        loss.backward()
        self.grad_step()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc = predictions==labels
        return {'loss': loss.detach().cpu(),'acc':acc.detach().cpu(),'kappa':{'preds':predictions.detach().cpu(), 'labels':labels.detach().cpu()}}



    def test_step(self, batch):
        labels = batch['labels']
        del batch['labels']
        if 'labels2' in batch:
            labels2 = batch['labels2']
            del batch['labels2']
        else:
            labels2 = None
        with torch.no_grad():
            outputs = self.model(**batch)
        loss = self.compute_loss(outputs,labels,labels2)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc = predictions==labels
        return {'loss': loss.detach().cpu(),'acc':acc.detach().cpu(),'kappa':{'preds':predictions.detach().cpu(), 'labels':labels.detach().cpu()}}
    

    