import torch
from torch import nn

from models.lm_base import LanguageModelBase
from utils.data_utils import CollateWraperInContextTuning
from utils.load_data import load_dataset_in_context_tuning



class LanguageModelInContextTuning(LanguageModelBase):
    def __init__(self, params, device):
        super().__init__(params, device)


    def prepare_data(self):
        data, self.examples_train, _, _ = load_dataset_in_context_tuning(self.params.task, self.params.debug)
        self.trainset = data['train']
        self.validset = data['valid']
        self.testset = data['test']
        self.max_label = max(data['train_dist'].keys())
        self.min_label = min(data['train_dist'].keys())
        self.majority_class = max(data['test_dist'].values())
        self.test_batch_size = 12
        self.num_labels = self.max_label - self.min_label + 1


    def dataloaders(self):
        collate_fn_train = CollateWraperInContextTuning(self.tokenizer, self.min_label, self.max_label, 
                                self.examples_train, self.params.num_examples, self.params.trunc_len, mode="train", 
                                max_seq_len = self.model.config.max_position_embeddings)
        # each val sample like train samples, are evaluated with only k=1 times average -> speeds up training
        collate_fn_val = CollateWraperInContextTuning(self.tokenizer, self.min_label, self.max_label, 
                                self.examples_train, self.params.num_examples, self.params.trunc_len, mode="train",
                                max_seq_len = self.model.config.max_position_embeddings)
        collate_fn_test = CollateWraperInContextTuning(self.tokenizer, self.min_label, self.max_label, 
                                self.examples_train, self.params.num_examples, self.params.trunc_len, 
                                mode="test", num_test_avg=self.params.num_test_avg, test_batch_size=self.test_batch_size,
                                max_seq_len = self.model.config.max_position_embeddings)
        

        train_loader = torch.utils.data.DataLoader(self.trainset, collate_fn=collate_fn_train, batch_size=self.params.batch_size, 
                                    num_workers=self.params.workers, shuffle=True, drop_last=False)
 
        # else batch_size like train_loader when CollateWraperTrainInContextTuning is used
        valid_loader = torch.utils.data.DataLoader(self.validset, collate_fn=collate_fn_val, batch_size=self.test_batch_size, 
                                    num_workers=self.params.workers, shuffle=False, drop_last=False)
        # actual batch_size after collating = batch_size * num_test_avg if using CollateWraperTestInContextTuning
        #valid_loader = torch.utils.data.DataLoader(self.validset, collate_fn=collate_fn_val, batch_size=test_batch_size, 
        #                            num_workers=self.params.workers, shuffle=False, drop_last=False)
        # actual batch_size after collating = batch_size * num_test_avg 
        test_loader = torch.utils.data.DataLoader(self.testset, collate_fn=collate_fn_test, batch_size=self.test_batch_size, 
                                    num_workers=self.params.workers, shuffle=False, drop_last=False)

        return train_loader, valid_loader, test_loader


    def eval_step(self, batch):
        # same as in test step in LanguageModelBase
        out = super().test_step(batch)

        return out


    def test_step(self, batch):
        # for eval batch, no test time averaging, eval batch is contructed the same as in train batch 
        # and uses eval_step() in LanguageModelBase

        # test time averaging for test batch, test batch input will always consist of single datapoint
        # test batch will be the same test datapoint repeated num_test_avg times for test epoch
        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch["inputs"])
            else:
                outputs = self.model(**batch["inputs"])
        loss = outputs.loss
        #print("loss", loss)
        
        '''
        # logits dim = batch_size X num_classes
        logits = outputs.logits
        #print("logits", logits)
        # softmax_outs dim = batch_size X num_classes
        softmax_outs = nn.functional.softmax(logits, dim=-1)
        #print("softmax_outs", softmax_outs)
        # mean averaging on softmax_outs across test_samples/rows
        # outs dim = 1 X num_classes
        outs = torch.mean(softmax_outs, dim=0)
        #print("outs", outs)
        # predictions dim = 1 X 1
        predictions = torch.argmax(outs, dim=-1)
        #print("predictions", predictions)
        # all vals in batch["inputs"]["labels"] are equal since they correspond to the same test datapoint
        batch["inputs"]["labels"] = batch["inputs"]["labels"][0]
        #print("batch["inputs"]["labels"]", batch["inputs"]["labels"])
        acc = (predictions == batch["inputs"]["labels"])
        #print("acc", acc)
        '''

        # logits dim = batch_size X num_classes
        # where batch_size = test_batch_size*num_test_avg
        logits = outputs.logits
        #print("logits", logits)
        # softmax_outs dim = batch_size X num_classes
        softmax_outs = nn.functional.softmax(logits, dim=-1)
        #print("softmax_outs", softmax_outs)
        # reshaped softmax_outs dim = test_batch_size X num_test_avg X num_classes
        softmax_outs = torch.reshape(softmax_outs, (batch["actual_batch_size"], self.params.num_test_avg, -1))
        #print("reshaped softmax_outs", softmax_outs)
        # mean averaging on softmax_outs across test_samples
        # outs dim = test_batch_size X num_classes
        outs = torch.mean(softmax_outs, dim=1)
        #print("outs", outs)
        # predictions dim = test_batch_size X 1
        predictions = torch.argmax(outs, dim=-1)
        #print("predictions", predictions)
        # all vals in batch["inputs"]["labels"] are equal since they correspond to the same test datapoint
        #print('batch["inputs"]["labels"]', batch["inputs"]["labels"])
        # pick every num_test_avg labels since labels are repeated
        batch["inputs"]["labels"] = batch["inputs"]["labels"][::self.params.num_test_avg]
        #print('sampled batch["inputs"]["labels"]', batch["inputs"]["labels"])
        acc = (predictions == batch["inputs"]["labels"])
        #print("acc", acc)

        return {
                'loss': loss.detach().cpu(),
                'acc':acc.detach().cpu(),
                'kappa':{
                    'preds':predictions.detach().cpu(), 
                    'labels':batch["inputs"]["labels"].detach().cpu()
                    }
                }