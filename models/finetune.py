import os
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AdamW
import numpy as np
from tqdm import tqdm
import copy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.load_data import load_dataset
from utils.datautils import CollateWraper



class BaseModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.params = copy.deepcopy(params)
        self.device = device
    
    def prepare_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.lm)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.params.lm, num_labels=self.max_label-self.min_label+1)
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)

    def prepare_data(self):
        data = load_dataset(self.params.task, create_hash=False,train=0.6, valid=0.2)
        self.trainset = data['train']
        self.validset = data['valid']
        self.testset = data['test']
        self.max_label = max(data['train_dist'].keys())
        self.min_label = min(data['train_dist'].keys())

    def dataloaders(self, iters=None):
        collate_fn = CollateWraper(self.tokenizer, self.min_label)
        train_loader = torch.utils.data.DataLoader(
            self.trainset, collate_fn=collate_fn, batch_size=self.params.batch_size, num_workers=self.params.workers)
        test_loader = torch.utils.data.DataLoader(
            self.testset, collate_fn=collate_fn, batch_size=self.params.batch_size*4, num_workers=self.params.workers,   shuffle=False, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(
            self.validset, collate_fn=collate_fn, batch_size=self.params.batch_size*4, num_workers=self.params.workers,   shuffle=False, drop_last=False)
        return train_loader, valid_loader, test_loader
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def grad_step(self):
        self.optimizer.step()
    
    def train_step(self, batch):
        self.zero_grad()
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        self.grad_step()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc = predictions==batch['labels']
        return {'loss': loss.detach().cpu(),'acc':acc.detach().cpu()}



    def test_step(self, batch):
        with torch.no_grad():
            outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc = predictions==batch['labels']
        return {'loss': loss.detach().cpu(),'acc':acc.detach().cpu()}
    

    