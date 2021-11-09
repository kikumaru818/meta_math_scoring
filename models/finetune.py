import os
import torch
from torch import nn
from torch.nn import functional as F, parameter
from transformers import AdamW
import numpy as np
from tqdm import tqdm
import copy
from transformers import AutoModel,AutoTokenizer, AutoModelForSequenceClassification,AutoConfig,GPT2LMHeadModel
from utils.load_data import load_dataset
from utils.datautils import CollateWraper
from utils.utils import open_json

score_mapper = {
                'verb':
                {2: ['Ġbad', 'Ġgood'], 3:['Ġbad','Ġaverage','Ġgood'],4:['Ġbad','Ġaverage','Ġgood', 'Ġexcellent'] },
                'score': 
                {2: ['Ġ0', 'Ġ1'], 3:['Ġ0','Ġ1','Ġ2'],4:['Ġ0','Ġ1','Ġ2', 'Ġ3'] }
                }

class MultitaskModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model.params.lm)
        classifier_dropout = (
            model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.layers = nn.ModuleList([nn.Linear(model.config.hidden_size, model.max_label[idx]-model.min_label[idx]+1) for idx in range(len(model.max_label))])

    def forward(self, input_ids,token_type_ids, attention_mask):
        outputs = self.base_model(input_ids,token_type_ids=token_type_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs[1])
        outputs = [layer(outputs) for layer in self.layers]
        return outputs


class BaseModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.params = copy.deepcopy(params)
        self.device = device
        
    
    def prepare_model(self):
        if self.params.task!='all':
            self.config = AutoConfig.from_pretrained(self.params.lm,num_labels=self.max_label-self.min_label+1)
        else:
            self.config = AutoConfig.from_pretrained(self.params.lm)
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.lm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.config.pad_token_id = self.config.eos_token_id
        if self.params.generate=='none' and self.params.task!='all':
            self.model = AutoModelForSequenceClassification.from_pretrained(self.params.lm, config=self.config)
        elif self.params.generate=='none' and self.params.task=='all':
            self.model = MultitaskModel(self)
        else:
            assert 'gpt' in self.params.lm, 'only gpt model is implemented with generative tokens'
            self.model = GPT2LMHeadModel.from_pretrained(self.params.lm,config=self.config)
            labels =  score_mapper[self.params.generate][self.max_label-self.min_label+1]
            self.label_ids = self.tokenizer.convert_tokens_to_ids(labels)
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)
        pass


    def prepare_data(self):
        if self.params.task!='all':
            data = load_dataset(self.params.task, create_hash=False,train=0.6, valid=0.2)
            self.trainset = data['train']
            self.validset = data['valid']
            self.testset = data['test']
            self.max_label = max(data['train_dist'].keys())
            self.min_label = min(data['train_dist'].keys())
            self.majority_class = max(data['test_dist'].values())
        else:
            self.params.task_lists = open_json('data/tasks.json')
            data = [load_dataset(task, create_hash=False,train=0.6, valid=0.2) for task in self.params.task_lists]
            self.trainset = [d['train'] for d in data]
            self.validset = [d['valid'] for d in data]
            self.testset = [d['test'] for d in data]
            self.max_label = [max(d['train_dist'].keys()) for d in data]
            self.min_label = [min(d['train_dist'].keys()) for d in data]
            self.majority_class = [max(d['test_dist'].values()) for d in data]
            for task_id in range(len(self.params.task_lists)):
                for d in self.trainset[task_id]:
                    d['tid'] = task_id
                for d in self.validset[task_id]:
                    d['tid']=  task_id
                for d in self.testset[task_id]:
                    d['tid']=  task_id
            self.trainset =  sum(self.trainset, [])
            self.validset =  sum(self.validset, [])
            self.testset =  sum(self.testset, [])

                
            


    def dataloaders(self, iters=None):
        collate_fn = CollateWraper(self.tokenizer, self.min_label, self.params.generate)
        train_loader = torch.utils.data.DataLoader(
            self.trainset, collate_fn=collate_fn,shuffle=True, batch_size=self.params.batch_size, num_workers=self.params.workers)
        test_loader = torch.utils.data.DataLoader(
            self.testset, collate_fn=collate_fn, batch_size=self.params.batch_size*2, num_workers=self.params.workers,   shuffle=False, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(
            self.validset, collate_fn=collate_fn, batch_size=self.params.batch_size*2, num_workers=self.params.workers,   shuffle=False, drop_last=False)
        return train_loader, valid_loader, test_loader
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def grad_step(self):
        self.optimizer.step()
    
    def compute_loss(self,batch, outputs,labels, labels2):
        if self.params.generate=='none':
            return self.bert_style_loss(outputs.logits,labels,labels2)
        else:
            return self.gpt_style_loss(batch, outputs,labels,labels2)
    
    def gpt_style_loss(self,batch, outputs,labels, labels2):
        predict_positions = torch.sum(batch['attention_mask'], dim=-1)-2
        logits = outputs.logits[torch.arange(len(labels)), predict_positions, :][:, self.label_ids]
        return self.bert_style_loss(logits,labels,labels2)

    def bert_style_loss(self,logits,labels, labels2):
        loss_functions = self.params.losses.split(';')
        is_labels2 = self.params.labels2
        losses = []
        for loss_function in loss_functions:
            losses.append(self.loss_layer(loss_function, logits,labels))
            if is_labels2:
                losses.append(self.loss_layer(loss_function, logits,labels2))
        return sum(losses)/len(losses), logits
    
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
        if 'tid' in batch:
            task_ids = batch['tid']
            del batch['tid']
        outputs = self.model(**batch)
        if self.params.task=='all':
            self.is_training = True
            return self.multitask_loss(outputs,labels,labels2,task_ids)
        #
        loss,logits = self.compute_loss(batch,outputs,labels,labels2)
        # loss = outputs.loss
        loss.backward()
        self.grad_step()
        predictions = torch.argmax(logits, dim=-1)
        acc = predictions==labels
        return {'loss': loss.detach().cpu(),'acc':acc.detach().cpu(),'kappa':{'preds':predictions.detach().cpu(), 'labels':labels.detach().cpu()}}
    
    def multitask_loss(self,outputs, labels,labels2,task_ids):
        loss = None
        res = {}
        for tid in range(len(self.params.task_lists)):
            flag = task_ids ==tid
            if flag.sum()==0:
                continue
            logits = outputs[tid][flag]
            if loss is None:
                loss = self.bert_style_loss(logits,labels[flag],labels2[flag])[0]*flag.sum()
            else:
                loss = loss +self.bert_style_loss(logits,labels[flag],labels2[flag])[0]*flag.sum()
            predictions = torch.argmax(logits, dim=-1)
            acc = predictions==labels[flag]
            res['accuracy_'+str(tid)] = acc
            res['kappa_'+str(tid)] =  {'preds':predictions.detach().cpu(), 'labels':labels[flag].detach().cpu()}
        loss /= len(labels)
        if self.is_training:
            loss.backward()
            self.grad_step()
        res['loss'] =  loss.detach().cpu()
        return res
        # return {'loss': loss.detach().cpu(),'acc':acc.detach().cpu(),'kappa':{'preds':predictions.detach().cpu(), 'labels':labels.detach().cpu()}}
            
        





    def test_step(self, batch):
        labels = batch['labels']
        del batch['labels']
        if 'labels2' in batch:
            labels2 = batch['labels2']
            del batch['labels2']
        else:
            labels2 = None
        if 'tid' in batch:
            task_ids = batch['tid']
            del batch['tid']
        with torch.no_grad():
            outputs = self.model(**batch)
        if self.params.task=='all':
            self.is_training = False
            return self.multitask_loss(outputs,labels,labels2,task_ids)
        loss,logits = self.compute_loss(batch, outputs,labels,labels2)
        predictions = torch.argmax(logits, dim=-1)
        acc = predictions==labels
        return {'loss': loss.detach().cpu(),'accuracy':acc.detach().cpu(),'kappa':{'preds':predictions.detach().cpu(), 'labels':labels.detach().cpu()}}
    

    