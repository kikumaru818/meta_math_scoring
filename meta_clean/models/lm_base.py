import copy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AdamW, GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
import torch
from torch import nn
from models.Sbert_classifer import SBERT_Classifer
from utils.data_utils import CollateWraper
from utils.load_data import load_dataset_base


class LanguageModelBase(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.params = copy.deepcopy(params)
        self.device = device


    def prepare_model(self):
        if( self.params.generative_model ):
            self.config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False, num_labels=self.num_labels)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.config.pad_token_id = self.config.eos_token_id
            self.model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=self.config).to(self.device)
        if (self.params.sbert):
            self.model = SBERT_Classifer().to(self.device)
            if 'saved_model' in  self.params.lm:
                self.model.load_state_dict(torch.load(self.params.lm))
            self.tokenizer = self.model.model.tokenizer
            self.config = None
        else:
            self.config = AutoConfig.from_pretrained(self.params.lm, num_labels=self.num_labels) 
            self.tokenizer = AutoTokenizer.from_pretrained(self.params.tok)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.config.pad_token_id = self.config.eos_token_id
            self.model = AutoModelForSequenceClassification.from_pretrained(self.params.lm, config=self.config).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)
        # multi gpu training
        if( torch.cuda.device_count() > 1 ):
            self.model = nn.DataParallel(self.model)


    def prepare_data(self):
        data = load_dataset_base(self.params.task, self.params.debug)
        self.trainset = data['train']
        self.validset = data['valid']
        self.testset = data['test']
        self.max_label = max(data['train_dist'].keys())
        self.min_label = min(data['train_dist'].keys())
        self.majority_class = max(data['test_dist'].values())
        self.num_labels = self.max_label - self.min_label + 1


    def dataloaders(self):
        train_loader = torch.utils.data.DataLoader(self.trainset, collate_fn=CollateWraper(self.tokenizer, self.min_label), 
                                    batch_size=self.params.batch_size, num_workers=self.params.workers, shuffle=True, drop_last=False)
        # actual batch_size after collating = batch_size * num_test_avg if in_context_tuning is enabled
        valid_loader = torch.utils.data.DataLoader(self.validset, collate_fn=CollateWraper(self.tokenizer, self.min_label), 
                                    batch_size=self.params.batch_size, num_workers=self.params.workers, shuffle=False, drop_last=False)
        # actual batch_size after collating = batch_size * num_test_avg if in_context_tuning is enabled
        test_loader = torch.utils.data.DataLoader(self.testset, collate_fn=CollateWraper(self.tokenizer, self.min_label), 
                                    batch_size=self.params.batch_size, num_workers=self.params.workers, shuffle=False, drop_last=False)

        
        return train_loader, valid_loader, test_loader


    def zero_grad(self):
        self.optimizer.zero_grad()


    def grad_step(self, scaler):
        if( self.params.amp ):
            scaler.step(self.optimizer)
        else:
            self.optimizer.step()


    def train_step(self, batch, scaler):
        self.zero_grad()
        if( self.params.amp ):
            # casts operations to mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch["inputs"])
        else:
            outputs = self.model(**batch["inputs"])
        loss = outputs.loss
        # multi gpu training
        if( torch.cuda.device_count() > 1 ):
            # scales the loss, and calls backward() to create scaled gradients
            if( self.params.amp ):
                scaler.scale(loss.sum()).backward()
            else:
                loss.sum().backward()
        else:
            # scales the loss, and calls backward() to create scaled gradients
            if( self.params.amp ):
                scaler.scale(loss).backward()
            else:
                loss.backward()
        self.grad_step(scaler)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc = ( predictions == batch["inputs"]["labels"] )
        # updates the scale for next iteration
        if( self.params.amp ):
            scaler.update()
        
        return {'loss': loss.detach().cpu(),
                'acc':acc.detach().cpu(),
                'kappa':{
                    'preds':predictions.detach().cpu(), 
                    'labels':batch["inputs"]["labels"].detach().cpu()
                    }
                }


    def eval_step(self, batch):
        # same as in test step in LanguageModelBase
        out = self.test_step(batch)

        return out


    def test_step(self, batch):
        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch["inputs"])
            else:
                outputs = self.model(**batch["inputs"])
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc = (predictions == batch["inputs"]["labels"])
            
        return {
                'loss': loss.detach().cpu(),
                'acc':acc.detach().cpu(),
                'kappa':{
                    'preds':predictions.detach().cpu(), 
                    'labels':batch["inputs"]["labels"].detach().cpu()
                    }
                }