import torch
from torch import nn

from models.lm_base import LanguageModelBase
from utils.data_utils import CollateWraperInContextTuningMetaLearning_Math
from utils.load_data import load_dataset_in_context_tuning_with_meta_learning as load

class LanguageModelInContextTuningMetaLearning(LanguageModelBase):
    def __init__(self, params, device, task_list, task_to_question, task_to_passage, passages, submit_mode=False, meta_learning_single=False, generic_task=False):
        super().__init__(params, device)
        self.task_list = task_list
        self.task_to_question = task_to_question
        self.task_to_passage = task_to_passage
        self.passages = passages
        self.submit_mode = submit_mode
        self.meta_learning_single = meta_learning_single
        self.generic_task = generic_task
        if self.params.math:
            from utils.load_data_math import load_dataset_in_context_tuning_with_meta_learning
            self.load = load_dataset_in_context_tuning_with_meta_learning
            if self.params.meta:
                from utils.load_data_math import load_dataset_in_context_tuning_with_meta_learning_question_split
                self.load = load_dataset_in_context_tuning_with_meta_learning_question_split
            if self.params.finetune:
                from utils.load_data_math import load_dataset_in_context_tuning_with_meta_learning_finetune
                self.load = load_dataset_in_context_tuning_with_meta_learning_finetune
        else:
            self.load = load


    def prepare_data(self):

        if self.params.math:
            self.data_meta, min_label, max_label = self.load(debug = self.params.debug, data_path = self.params.data_folder,
                                                             task_list = self.task_list, submit_mode=self.submit_mode,
                                                             meta_learning_single=self.meta_learning_single,
                                                             generic_task=self.generic_task, cross_val_fold=self.params.cross_val_fold,
                                                             alias = self.params.alias, fold = self.params.fold, n_example = self.params.new_examples)
        else:
            self.data_meta, min_label, max_label = self.load(self.params.debug, self.params.data_folder,
                            self.task_list, self.submit_mode, self.meta_learning_single, self.generic_task, self.params.cross_val_fold)
        self.trainset = self.data_meta['train']
        self.validsets = {}
        self.testsets = {}
        self.task_list = [ i for i in list(self.data_meta.keys()) if i != 'train']
        val_list = []
        for task in self.task_list:
            try:
                self.validsets[task] = self.data_meta[task]['valid']
                self.testsets[task] = self.data_meta[task]['test']
                val_list.append(task)
            except:
                continue
        self.test_batch_size = 12
        self.val_batch_size = 12
        if (self.meta_learning_single):
            self.min_label = min_label
            self.max_label = max_label
        else:
            # for meta learning model, min=1, max=4 is fixed as single classification layer for all tasks
            if self.params.math:
                self.min_label = 0
            else:
                self.min_label = 1
            self.max_label = 4
        self.num_labels = self.max_label - self.min_label + 1
        print(len(self.trainset))
        self.task_list = val_list


    def dataloaders(self):
        if self.params.math:
            CollateWraperInContextTuningMetaLearning = CollateWraperInContextTuningMetaLearning_Math
        collate_fn_train = CollateWraperInContextTuningMetaLearning(self.tokenizer, self.data_meta, self.task_to_question, 
                                                                    self.params.num_examples, 
                                                                    self.params.trunc_len, mode="train", 
                                                                    generative_model=self.params.generative_model,
                                                                    use_demographic = self.params.demographic, args=self.params)
        # each val sample like train samples, are evaluated with only k=1 times average -> speeds up training -> naep submission
        # for cross validation, we do test time averaging for val set also -> for paper results
        collate_fn_val = CollateWraperInContextTuningMetaLearning(self.tokenizer, self.data_meta, self.task_to_question, 
                                                                    self.params.num_examples, 
                                                                    self.params.trunc_len, mode="val", num_val_avg=self.params.num_val_avg,
                                                                    val_batch_size=self.val_batch_size, 
                                                                    generative_model=self.params.generative_model,
                                                                    use_demographic = self.params.demographic, args=self.params)
        collate_fn_test = CollateWraperInContextTuningMetaLearning(self.tokenizer, self.data_meta, self.task_to_question, 
                                                            self.params.num_examples, 
                                                            self.params.trunc_len, mode="test", num_test_avg=self.params.num_test_avg,
                                                            test_batch_size=self.test_batch_size, 
                                                            generative_model=self.params.generative_model,
                                                            use_demographic = self.params.demographic, args=self.params)
        

        train_loader = torch.utils.data.DataLoader(self.trainset, collate_fn=collate_fn_train, batch_size=self.params.batch_size, 
                                                    num_workers=self.params.workers, shuffle=True, drop_last=False)
        valid_loaders = {}
        test_loaders = {}
        for task in self.task_list:
            # batch_size=self.params.batch_size like train_loader when collate_fn_val=CollateWraperTrainInContextTuning
            valid_loaders[task] = torch.utils.data.DataLoader(self.validsets[task], collate_fn=collate_fn_val, 
                                                batch_size=self.val_batch_size, num_workers=self.params.workers, 
                                                shuffle=False, drop_last=False)
            # actual batch_size after collating = batch_size * num_test_avg 
            test_loaders[task] = torch.utils.data.DataLoader(self.testsets[task], collate_fn=collate_fn_test, 
                                                batch_size=self.test_batch_size, num_workers=self.params.workers, 
                                                shuffle=False, drop_last=False)

        return train_loader, valid_loaders, test_loaders


    def train_step(self, batch, scaler, loss_func):
        self.zero_grad()
        if( self.params.amp ):
            # casts operations to mixed precision
            with torch.cuda.amp.autocast():
                if( self.params.generative_model ):
                    outputs = self.model(**batch["inputs"])
                else:
                    outputs = self.model(**batch["inputs"])
        else:
            if( self.params.generative_model ):
                outputs = self.model(**batch["inputs"])
            else:
                outputs = self.model(**batch["inputs"])
        logits = outputs.logits

        # apply a softmax max over invalid class labels = negative infinity
        # https://stackoverflow.com/questions/57548180/filling-torch-tensor-with-zeros-after-certain-index
        mask = torch.zeros(logits.shape[0], logits.shape[1] + 1, dtype=logits.dtype, device=logits.device)
        if not self.params.no_scale:
            mask[(torch.arange(logits.shape[0]), batch["max_labels"])] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        masked_logits = logits.masked_fill_(mask.eq(1), value=float('-inf'))

        # calculate masked cross entropy loss
        loss = loss_func(masked_logits.view(-1, self.num_labels), batch["inputs"]["labels"].view(-1))

        softmax_outs = nn.functional.softmax(masked_logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        acc = ( predictions == batch["inputs"]["labels"] )


        # multi gpu training
        if( torch.cuda.device_count() > 1 ):
            # TODO P2: is this the correct way? resolve warning multi gpu warning
            # scales the loss, and calls backward() to create scaled gradients
            # loss.sum() to sum over parallel batches over multi gpu
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
        """
        #print("\neval\n")
        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch["inputs"])
            else:
                outputs = self.model(**batch["inputs"])
        loss = outputs.loss
        logits = outputs.logits
        #print("logits", logits)

        # apply a softmax max over invalid class labels = negative infinity
        # https://stackoverflow.com/questions/57548180/filling-torch-tensor-with-zeros-after-certain-index
        mask = torch.zeros(logits.shape[0], logits.shape[1] + 1, dtype=logits.dtype, device=logits.device)
        mask[(torch.arange(logits.shape[0]), batch["max_labels"])] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        masked_logits = logits.masked_fill_(mask.eq(1), value=float('-inf')) 
        #print("masked_logits", masked_logits)
        softmax_outs = nn.functional.softmax(masked_logits, dim=-1)
        #print("softmax_outs", softmax_outs)
        predictions = torch.argmax(softmax_outs, dim=-1)
        #print("predictions", predictions)
        #predictions = torch.argmax(logits, dim=-1)

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
        """

        # for eval batch, no test time averaging, eval batch is contructed the same as in train batch 
        # and uses eval_step() in LanguageModelBase
        # -> however for cross validation, we do perform test time averaging for eval batch as well

        # test time averaging for test batch, test batch input will always consist of single datapoint
        # test batch will be the same test datapoint repeated num_test_avg times for test epoch
        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch["inputs"])
            else:
                outputs = self.model(**batch["inputs"])

        '''
        # logits dim = batch_size X num_classes
        logits = outputs.logits
        # softmax_outs dim = batch_size X num_classes
        softmax_outs = nn.functional.softmax(logits, dim=-1)
        # mean averaging on softmax_outs across test_samples/rows
        # outs dim = 1 X num_classes
        outs = torch.mean(softmax_outs, dim=0)
        # predictions dim = 1 X 1
        predictions = torch.argmax(outs, dim=-1)
        # all vals in batch["inputs"]["labels"] are equal since they correspond to the same test datapoint
        batch["inputs"]["labels"] = batch["inputs"]["labels"][0]
        #print(batch["inputs"]["labels"])
        acc = (predictions == batch["inputs"]["labels"])
        #print(acc)
        '''

        loss = outputs.loss
        # logits dim = batch_size X num_classes
        # where batch_size = test_batch_size*num_test_avg
        logits = outputs.logits
        # softmax_outs dim = batch_size X num_classes

        # apply a softmax max over invalid class labels = negative infinity
        # https://stackoverflow.com/questions/57548180/filling-torch-tensor-with-zeros-after-certain-index
        mask = torch.zeros(logits.shape[0], logits.shape[1] + 1, dtype=logits.dtype, device=logits.device)
        if not self.params.no_scale:
            mask[(torch.arange(logits.shape[0]), batch["max_labels"])] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        masked_logits = logits.masked_fill_(mask.eq(1), value=float('-inf'))
        softmax_outs = nn.functional.softmax(masked_logits, dim=-1)
        # reshaped softmax_outs dim = val_batch_size X num_val_avg X num_classes
        softmax_outs = torch.reshape(softmax_outs, (batch["actual_batch_size"], self.params.num_val_avg, -1))
        # mean averaging on softmax_outs across test_samples
        # outs dim = test_batch_size X num_classes
        outs = torch.mean(softmax_outs, dim=1)
        # predictions dim = test_batch_size X 1
        predictions = torch.argmax(outs, dim=-1)
        # all vals in batch["inputs"]["labels"] are equal since they correspond to the same test datapoin
        # pick every num_test_avg labels since labels are repeated
        batch["inputs"]["labels"] = batch["inputs"]["labels"][::self.params.num_val_avg]
        #print('sampled batch["inputs"]["labels"]', batch["inputs"]["labels"])
        acc = (predictions == batch["inputs"]["labels"])
        #print("acc", acc)

        return {
            'loss': loss.detach().cpu(),
            'acc':acc.detach().cpu(),
            'kappa':{
                'preds':predictions.detach().cpu(), 
                'labels':batch["inputs"]["labels"].detach().cpu()
                },
            'counts': batch['counts']
            }


    def test_step(self, batch):
        #print("\ntest\n")
        # for eval batch, no test time averaging, eval batch is contructed the same as in train batch 
        # and uses eval_step() in LanguageModelBase
        # -> however for cross validation, we do perform test time averaging for eval batch as well

        # test time averaging for test batch, test batch input will always consist of single datapoint
        # test batch will be the same test datapoint repeated num_test_avg times for test epoch
        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch["inputs"])
            else:
                outputs = self.model(**batch["inputs"])

        '''
        # logits dim = batch_size X num_classes
        logits = outputs.logits
        # softmax_outs dim = batch_size X num_classes
        softmax_outs = nn.functional.softmax(logits, dim=-1)
        # mean averaging on softmax_outs across test_samples/rows
        # outs dim = 1 X num_classes
        outs = torch.mean(softmax_outs, dim=0)
        # predictions dim = 1 X 1
        predictions = torch.argmax(outs, dim=-1)
        # all vals in batch["inputs"]["labels"] are equal since they correspond to the same test datapoint
        batch["inputs"]["labels"] = batch["inputs"]["labels"][0]
        #print(batch["inputs"]["labels"])
        acc = (predictions == batch["inputs"]["labels"])
        #print(acc)
        '''

        loss = outputs.loss
        # logits dim = batch_size X num_classes
        # where batch_size = test_batch_size*num_test_avg
        logits = outputs.logits
        # apply a softmax max over invalid class labels = negative infinity
        # https://stackoverflow.com/questions/57548180/filling-torch-tensor-with-zeros-after-certain-index
        mask = torch.zeros(logits.shape[0], logits.shape[1] + 1, dtype=logits.dtype, device=logits.device)
        mask[(torch.arange(logits.shape[0]), batch["max_labels"])] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        masked_logits = logits.masked_fill_(mask.eq(1), value=float('-inf'))
        softmax_outs = nn.functional.softmax(masked_logits, dim=-1)
        # reshaped softmax_outs dim = test_batch_size X num_test_avg X num_classes
        softmax_outs = torch.reshape(softmax_outs, (batch["actual_batch_size"], self.params.num_test_avg, -1))
        # mean averaging on softmax_outs across test_samples
        # outs dim = test_batch_size X num_classes
        outs = torch.mean(softmax_outs, dim=1)
        # predictions dim = test_batch_size X 1
        predictions = torch.argmax(outs, dim=-1)
        # all vals in batch["inputs"]["labels"] are equal since they correspond to the same test datapoint
        # pick every num_test_avg labels since labels are repeated
        batch["inputs"]["labels"] = batch["inputs"]["labels"][::self.params.num_test_avg]
        #print('sampled batch["inputs"]["labels"]', batch["inputs"]["labels"])
        acc = (predictions == batch["inputs"]["labels"])



        return {
            'loss': loss.detach().cpu(),
            'acc':acc.detach().cpu(),
            'kappa':{
                'preds':predictions.detach().cpu(), 
                'labels':batch["inputs"]["labels"].detach().cpu()
                }
            }