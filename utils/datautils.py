import torch

def tokenize_function(tokenizer, sentences):
    return tokenizer(sentences, padding=True, truncation=True,return_tensors="pt")

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
        labels  =  torch.tensor([d['l1'] if d['l1']>=0 else d['l2'] for d in batch]).long()-self.min_label
        labels2  =  torch.tensor([d['l2'] if d['l2']>=0 else d['l1'] for d in batch]).long()-self.min_label

        inputs = tokenize_function(self.tokenizer,features)
        inputs['labels'] = labels
        inputs['labels2'] = labels2
        
        return inputs

        

    