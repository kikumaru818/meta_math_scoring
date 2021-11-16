from collections import defaultdict
import math
import os
import torch
from torch import nn
from torch._C import device
from torch.nn import functional as F, parameter
from transformers import AdamW
import numpy as np
from tqdm import tqdm
import copy
from transformers import AutoModel,AutoTokenizer, AutoModelForSequenceClassification,AutoConfig,GPT2LMHeadModel,get_constant_schedule_with_warmup
from utils.load_data import load_dataset
from utils.datautils import CollateWraper, tokenize_function, ProtoSampler
from utils.utils import open_json
from torch.nn.utils.rnn import pad_sequence

score_mapper = {
                'verb':
                {2: ['Ġbad', 'Ġgood'], 3:['Ġbad','Ġaverage','Ġgood'],4:['Ġbad','Ġaverage','Ġgood', 'Ġexcellent'] },
                'score': 
                {2: ['Ġ0', 'Ġ1'], 3:['Ġ0','Ġ1','Ġ2'],4:['Ġ0','Ġ1','Ġ2', 'Ġ3'] }
                }

class MultitaskModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model.params.lm)
        self.single_head = model.params.single_head
        classifier_dropout = (
            model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if self.single_head:#single class head:
            self.ll = nn.Sequential(nn.Linear(model.config.hidden_size, 4), nn.Softmax(dim=-1))
            self.layers = self.create_mapping(model)
        else: #class specific heads
            self.layers = nn.ModuleList([nn.Linear(model.config.hidden_size, model.max_label[idx]-model.min_label[idx]+1) for idx in range(len(model.max_label))])
    
    def create_mapping(self, model):
        weight = []
        for idx in range(len(model.params.task_lists)):
            n_label = model.max_label[idx]- model.min_label[idx]+1
            matrix = torch.zeros((4,n_label),device=model.device, requires_grad=False)
            if n_label==4:
                matrix[0,0]=matrix[1,1]=matrix[2,2]=matrix[3,3] =1.
            elif n_label==3:
                matrix[0,0]=matrix[3,2] =matrix[1,1]=matrix[2,1] =1.
            else:
                matrix[0,0]=matrix[1,0] = matrix[2,1]=matrix[3,1] = 1.
            weight.append(matrix)
        return weight

    def forward(self, input_ids=None,token_type_ids=None, attention_mask=None, inputs_embeds=None):
        if input_ids is not None:
            outputs = self.bert(input_ids,token_type_ids=token_type_ids, attention_mask=attention_mask)
        elif inputs_embeds is not None:
            outputs = self.bert(inputs_embeds=inputs_embeds,token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            raise AssertionError('Pass Input Embeds or input_ids')
        outputs = self.dropout(outputs[1])
        if self.single_head:#single class head:
            outputs =  self.ll(outputs)
            outputs = [torch.matmul(outputs,layer) for layer in self.layers]#probs
            outputs = [torch.log(output+1e-6) for output in outputs]
        else:
            outputs = [layer(outputs) for layer in self.layers]
        return outputs


class BaseModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.params = copy.deepcopy(params)
        self.device = device
        self.counter = 0
        
    
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
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer,num_warmup_steps =min(int(len(self.trainset)//self.params.batch_size),self.params.max_epochs)*5  )
        if self.params.include_passage:
            if self.params.task!='all':
                self.question =  tokenize_function(self.tokenizer, self.question)
                self.passage = tokenize_function(self.tokenizer, self.passage)
                self.question = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in self.question.items()}
                self.passage = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in self.passage.items()}
            else:
                self.question =  [tokenize_function(self.tokenizer, d) for d in self.question]
                self.passage = [tokenize_function(self.tokenizer, d) for d in self.passage]
                self.question = [{k: v.to(self.device) if torch.is_tensor(v) else v for k, v in d.items()} for d in self.question]
                self.passage = [{k: v.to(self.device) if torch.is_tensor(v) else v for k, v in d.items()} for d in self.passage]
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
            self.human_kappa = data['human_kappa']
            if self.params.include_passage:
                questions, passages = open_json('data/questions.json'), open_json('data/task_passages.json')
                self.question =   [questions[self.params.task]]
                self.passage = passages[passages[self.params.task]]
            if self.params.include_question:
                question = open_json('data/questions.json')[self.params.task]
                for dataset in [self.trainset,self.validset, self.testset]:
                    for d in dataset:
                        d['txt'] = question+ ' [SEP] '+d['txt']
                        # d['txt'] = d['txt']+ ' [SEP] '+question

        else:
            self.params.task_lists = open_json('data/tasks.json')
            data = [load_dataset(task, create_hash=False,train=0.6, valid=0.2) for task in self.params.task_lists]
            self.trainset = [d['train'] for d in data]
            self.validset = [d['valid'] for d in data]
            self.testset = [d['test'] for d in data]
            self.max_label = [max(d['train_dist'].keys()) for d in data]
            self.min_label = [min(d['train_dist'].keys()) for d in data]
            self.majority_class = [max(d['test_dist'].values()) for d in data]
            self.human_kappa = [d['human_kappa'] for d in data]
            for task_id in range(len(self.params.task_lists)):
                for d in self.trainset[task_id]:
                    d['tid'] = task_id
                for d in self.validset[task_id]:
                    d['tid']=  task_id
                for d in self.testset[task_id]:
                    d['tid']=  task_id
            if self.params.single_head:
                generic_tasks = open_json('data/generic_tasks.json')
                for task, min_label, max_label in generic_tasks:
                    self.params.task_lists.append(task)
                    self.min_label.append(min_label)
                    self.max_label.append(max_label)
                    dataset = open_json('data/'+task+'.json')
                    for d in dataset:
                        d['tid'] =  len(self.params.task_lists)-1
                    self.validset.append(dataset)
                    self.testset.append(dataset)

            self.trainset =  sum(self.trainset, [])
            self.validset =  sum(self.validset, [])
            self.testset =  sum(self.testset, [])
            if self.params.include_passage:
                questions, passages = open_json('data/questions.json'),     open_json('data/task_passages.json')
                self.question =   [ [questions[d]] for d in self.params.task_lists]
                self.passage = [passages[passages[d]] for d in self.params.task_lists]
            if self.params.include_question:
                questions = open_json('data/questions.json')
                for dataset in [self.trainset,self.validset, self.testset]:
                    for d in dataset:
                        d['txt'] = questions[self.params.task_lists[d['tid']]]+ ' [SEP] '+d['txt']
                        # d['txt'] = d['txt'] +' [SEP] ' +questions[self.params.task_lists[d['tid']]]

                
            


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
        self.scheduler.step()
    
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
            max_label = self.max_label[self.tid] if isinstance(self.max_label, list) else self.max_label
            min_label = self.min_label[self.tid] if isinstance(self.min_label, list) else self.min_label
            score = ((torch.arange(max_label-min_label+1).to(self.device)[None,:]-target[:,None])**2.)/ ((max_label-min_label)**2.)
            loss = torch.sum(score*probs)/ len(target)
        return loss
    
    def update_passage_cache(self):
        with torch.no_grad():
            if self.params.task!='all':
                self.passage_embedding = self.model(**self.passage, output_hidden_states=True).hidden_states[-1][:,0]
                self.question_embedding = self.model(**self.question, output_hidden_states=True).hidden_states[-1][:,0]
            else:
                self.passage_embedding = [self.model.bert(**d, output_hidden_states=True).hidden_states[-1][:,0] for d in self.passage]
                self.question_embedding = [self.model.bert(**d, output_hidden_states=True).hidden_states[-1][:,0]  for d in self.question]

    def append_passage_question_embeddings(self,batch, task_ids):
        input_ids = batch['input_ids']
        input_ids[:,0]=  self.tokenizer.convert_tokens_to_ids('[SEP]')
        class_tokens = torch.zeros(len(input_ids),1).long().to(self.device)+self.tokenizer.convert_tokens_to_ids('[CLS]')
        input_embedding  = self.model.bert.embeddings.word_embeddings(input_ids)
        class_embedding = self.model.bert.embeddings.word_embeddings(class_tokens)
        if task_ids is None:
            input_embeds = torch.cat([class_embedding, self.question_embedding[None,:,:].expand(len(input_ids),-1,-1), self.passage_embedding[None,:,:].expand(len(input_ids),-1,-1), input_embedding], dim =1)
            concat_attention_mask = torch.ones(len(input_embedding),len(self.passage_embedding)+2).long().to(self.device)
            concat_token_type_ids = torch.zeros(len(input_embedding),len(self.passage_embedding)+2).long().to(self.device)
            attention_mask = torch.cat([concat_attention_mask, batch['attention_mask']], dim=1)
            token_type_ids = torch.cat([concat_token_type_ids, batch['token_type_ids']], dim=1)
        else:
            input_embeds, attention_mask, token_type_ids = [], [], []
            for idx in range(len(input_ids)):
                embeds = torch.cat([class_embedding[idx], self.question_embedding[task_ids[idx]],  self.passage_embedding[task_ids[idx]], input_embedding[idx]         ], dim =0)
                mask = torch.cat([torch.ones(len(self.passage_embedding[task_ids[idx]])+2).long().to(self.device), batch['attention_mask'][idx]   ], dim =0)
                token_ids = torch.cat([torch.zeros(len(self.passage_embedding[task_ids[idx]])+2).long().to(self.device), batch['token_type_ids'][idx]   ], dim =0)
                input_embeds.append(embeds)
                attention_mask.append(mask)
                token_type_ids.append(token_ids)
            input_embeds =  pad_sequence(input_embeds,batch_first=True)
            attention_mask =  pad_sequence(attention_mask,batch_first=True)
            token_type_ids =  pad_sequence(token_type_ids,batch_first=True)
        batch['attention_mask'] = attention_mask[:, :self.config.max_position_embeddings]
        batch['token_type_ids'] = token_type_ids[:, :self.config.max_position_embeddings]
        batch['inputs_embeds'] =  input_embeds[:, :self.config.max_position_embeddings]
            
        del batch['input_ids']
        pass

    def train_step(self, batch):
        self.zero_grad()
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
        else:
            task_ids = None

        if self.params.include_passage:# update passage and question embeddings
            if self.counter%self.params.update_every==0:
                self.update_passage_cache()
            self.counter += 1
            self.append_passage_question_embeddings(batch, task_ids)
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
            self.tid = tid
            flag = task_ids ==tid
            if flag.sum()==0:
                res['accuracy_'+str(tid)] = []
                res['kappa_'+str(tid)] =  {}
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
        else:
            task_ids = None
        if self.params.include_passage:# update passage and question embeddings
            with torch.no_grad():
                self.append_passage_question_embeddings(batch,task_ids)

        with torch.no_grad():
            outputs = self.model(**batch)
        if self.params.task=='all':
            self.is_training = False
            return self.multitask_loss(outputs,labels,labels2,task_ids)
        loss,logits = self.compute_loss(batch, outputs,labels,labels2)
        predictions = torch.argmax(logits, dim=-1)
        acc = predictions==labels
        return {'loss': loss.detach().cpu(),'accuracy':acc.detach().cpu(),'kappa':{'preds':predictions.detach().cpu(), 'labels':labels.detach().cpu()}}
    

class ProtoModel(BaseModel):
    def __init__(self, params, device):
        super().__init__(params,device)

    def prepare_model(self):
        self.config = AutoConfig.from_pretrained(self.params.lm)
        self.prototypes =  [None] if self.params.task !='all' else [None]*len(self.params.task_lists)
        self.stored_prototypes =  [{}] if self.params.task !='all' else [{}]*len(self.params.task_lists)
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.lm)
        self.model = AutoModel.from_pretrained(self.params.lm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.config.pad_token_id = self.config.eos_token_id
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer,num_warmup_steps =min(int(len(self.trainset)//self.params.batch_size * (self.params.proto_count+1)),self.params.max_epochs)*5  )

    def dataloaders(self, iters=None):
        self.counter = 0
        train_batch_sampler = ProtoSampler(self.trainset, self.params.batch_size, self.params.task, test =False)
        test_batch_sampler = ProtoSampler(self.testset, self.params.batch_size*2, self.params.task, test =True)
        valid_batch_sampler = ProtoSampler(self.validset, self.params.batch_size*2, self.params.task, test =True)
        collate_fn = CollateWraper(self.tokenizer, self.min_label, self.params.generate)
        train_loader = torch.utils.data.DataLoader(
            self.trainset, collate_fn=collate_fn, batch_sampler=train_batch_sampler, num_workers=self.params.workers)
        test_loader = torch.utils.data.DataLoader(
            self.testset, collate_fn=collate_fn, batch_sampler=test_batch_sampler, num_workers=self.params.workers)
        valid_loader = torch.utils.data.DataLoader(
            self.validset, collate_fn=collate_fn, batch_sampler=valid_batch_sampler, num_workers=self.params.workers)
        return train_loader, valid_loader, test_loader
    
    def add_prototypes(self,batch,tid):
        labels, labels2 = batch['labels'], batch['labels2']
        del batch['labels']
        del batch['labels2']
        with torch.no_grad():
            outputs = self.model(**batch)[1]
        self.prototypes[tid]['features'].append(outputs)
        self.prototypes[tid]['labels'].append(labels)
        self.prototypes[tid]['labels2'].append(labels2)
        if len(self.prototypes[tid]['features'])== self.params.proto_count:
            #concat
            self.stored_prototypes[tid]['features'] = torch.concat(self.prototypes[tid]['features'], dim =0)
            self.stored_prototypes[tid]['labels'] = torch.concat(self.prototypes[tid]['labels'], dim =0)
            self.stored_prototypes[tid]['labels2'] = torch.concat(self.prototypes[tid]['labels2'], dim =0)
        return

    def train_step(self, batch):
        self.counter += 1
        tid = int(batch['tid'][0]) if 'tid' in batch else 0
        if self.counter % (self.params.proto_count+1)==1: #new prototypes
            self.prototypes[tid] =  {'features':[], 'labels':[], 'labels2':[]}
        if self.counter % (self.params.proto_count+1)!=0:#Train iteration
            self.add_prototypes(batch, tid)
            return None
        #Training steps
        labels, labels2 = batch['labels'], batch['labels2']
        del batch['labels']
        del batch['labels2']
        outputs = self.model(**batch)[1]#32*768
        m = nn.Softmax(dim=-1)
        outputs =  m(torch.matmul(outputs, self.stored_prototypes[tid]['features'].T)/ math.sqrt(self.config.hidden_size))#32x128
        proto_weights =  torch.zeros(len(self.stored_prototypes[tid]['features']), self.max_label-self.min_label+1).to(self.device)
        proto_weights[torch.arange(len(proto_weights)), self.stored_prototypes[tid]['labels']] += 0.5 
        proto_weights[torch.arange(len(proto_weights)), self.stored_prototypes[tid]['labels2']] += 0.5 
        outputs = torch.log(torch.matmul(outputs, proto_weights)+1e-7)
        loss, logits =  self.bert_style_loss(outputs,labels, labels2)
        loss.backward()
        self.grad_step()
        predictions = torch.argmax(logits, dim=-1)
        acc = predictions==labels
        return {'loss': loss.detach().cpu(),'acc':acc.detach().cpu(),'kappa':{'preds':predictions.detach().cpu(), 'labels':labels.detach().cpu()}}

    def test_step(self, batch):
        tid = int(batch['tid'][0]) if 'tid' in batch else 0
        labels, labels2 = batch['labels'], batch['labels2']
        del batch['labels']
        del batch['labels2']
        with torch.no_grad():
            outputs = self.model(**batch)[1]#32*768
            m = nn.Softmax(dim=-1)
            outputs =  m(torch.matmul(outputs, self.stored_prototypes[tid]['features'].T)/ math.sqrt(self.config.hidden_size))#32x128
            proto_weights =  torch.zeros(len(self.stored_prototypes[tid]['features']), self.max_label-self.min_label+1).to(self.device)
            proto_weights[torch.arange(len(proto_weights)), self.stored_prototypes[tid]['labels']] += 0.5 
            proto_weights[torch.arange(len(proto_weights)), self.stored_prototypes[tid]['labels2']] += 0.5 
            outputs = torch.log(torch.matmul(outputs, proto_weights)+1e-7)
            loss, logits =  self.bert_style_loss(outputs,labels, labels2)
        predictions = torch.argmax(logits, dim=-1)
        acc = predictions==labels
        return {'loss': loss.detach().cpu(),'acc':acc.detach().cpu(),'kappa':{'preds':predictions.detach().cpu(), 'labels':labels.detach().cpu()}}




        



