from collections import defaultdict
import torch
import random
from torch.utils.data.sampler import Sampler

def tokenize_function(tokenizer, sentences):
    return tokenizer(sentences, padding=True, truncation=True,return_tensors="pt")

class ProtoSampler(Sampler):
    def __init__(self, dataset,batch_size,task, test):
        if task=='all':#all tasks
            pass
        else:
            if test:
                indices = torch.arange(len(dataset))
                self.batches = [batch.tolist() for batch in torch.split(indices, batch_size)]
            else:#train mix and match
                indices = defaultdict(list)
                for idx,d in enumerate(dataset):
                    indices[d['l1'] if d['l1']>=0 else d['l2']].append(idx)
                for k, v in indices.items():
                    random.shuffle(v)
                label_proportion = {k: len(v) for k,v in indices.items()}
                total_cnt =  sum(label_proportion.values())
                label_proportion = {k : max(3,int((v*batch_size)/total_cnt)) for k,v in label_proportion.items()}
                self.batches, offset, flag =[], defaultdict(int) , True
                while flag:
                    out = []
                    for k,v in label_proportion.items():
                        if offset[k]+v<= len(indices[k]):
                            out += indices[k][offset[k]: offset[k]+v]
                            offset[k] +=v
                        else:  flag = False
                    if flag: self.batches.append(out)
                pass
        
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)

class CollateWraper(object):
    def __init__(self, tokenizer,min_label,generate='none'):
        self.tokenizer = tokenizer
        self.min_label = min_label
        if generate!= 'none':
            self.append_text = ' The writing quality is good'
        else:
            self.append_text = ""
    def __call__(self, batch):
        features = [d['txt']+self.append_text for d in batch]
        if isinstance(self.min_label, list):
            labels  =  torch.tensor([d['l1']-self.min_label[d['tid']] if d['l1']>=0 else d['l2']- self.min_label[d['tid']] for d in batch]).long()
            labels2  =  torch.tensor([d['l2']-self.min_label[d['tid']] if d['l2']>=0 else d['l1']-self.min_label[d['tid']] for d in batch]).long()
        else:
            labels  =  torch.tensor([d['l1'] if d['l1']>=0 else d['l2'] for d in batch]).long()-self.min_label
            labels2  =  torch.tensor([d['l2'] if d['l2']>=0 else d['l1'] for d in batch]).long()-self.min_label

        inputs = tokenize_function(self.tokenizer,features)
        inputs['labels'] = labels
        inputs['labels2'] = labels2
        if isinstance(self.min_label, list):
            inputs['tid'] = torch.tensor([d['tid'] for d in batch]).long()
        return inputs

        

    