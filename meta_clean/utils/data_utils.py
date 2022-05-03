import torch
import random
import numpy as np

def tokenize_function(tokenizer, sentences_1, sentences_2=None):
    if(sentences_2 == None):
        return tokenizer(sentences_1, padding=True, truncation=True, return_tensors="pt")
    else:
        return tokenizer(sentences_1, sentences_2, padding=True, truncation=True, return_tensors="pt")


class CollateWraperParent(object):
    def __init__(self, tokenizer, min_label):
        self.tokenizer = tokenizer
        self.min_label = min_label


class CollateWraper(CollateWraperParent):
    def __init__(self, tokenizer, min_label):
        super().__init__(tokenizer, min_label)
    
    def __call__(self, batch):
        # construct features
        features = [d['txt'] for d in batch]
        inputs = tokenize_function(self.tokenizer, features)

        # construct labels
        labels  =  torch.tensor([d['l1'] if d['l1']>=0 else d['l2'] for d in batch]).long() - self.min_label
        inputs['labels'] = labels
        return {"inputs" : inputs}


class CollateWraperInContextTuningMetaLearning(CollateWraperParent):
    def __init__(self, tokenizer, data_meta, task_to_question, num_examples, trunc_len, mode, num_test_avg=1,  
                num_val_avg=1, test_batch_size=1, val_batch_size=1, max_seq_len=512, generative_model=False, use_demographic=False, params = None):
        super().__init__(tokenizer, min_label = 1)  
        # meta learning via in-context tuning
        self.data_meta = data_meta
        self.num_examples = num_examples
        self.trunc_len = trunc_len
        self.label_to_text = {
            1 : "poor",
            2 : "fair",
            3 : "good",
            4 : "excellent"
        }

        # demographic information
        self.use_demographic = use_demographic
        self.gender_map = {
            "1" : "male",
            "2" : "female"
        }
        self.race_map = {
            "1" : "white",
            "2" : "african american",
            "3" : "hispanic",
            "4" : "asian",
            "5" : "american indian",
            "6" : "pacific islander",
            "7" : "multiracial"
        }
        
        # get token IDs for labels using GPT2 tokenizer
        self.label_to_label_id = {}
        for label in self.label_to_text:
            self.label_to_label_id[label] = self.tokenizer(self.label_to_text[label])['input_ids']
        
        self.mode = mode
        self.num_test_avg = num_test_avg
        self.num_val_avg = num_val_avg
        self.task_to_question = task_to_question
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size
        # adding an extra 50 words in case num_tokens < num_words
        self.max_seq_len = max_seq_len + 50
        self.generative_model = generative_model
        self.params = params


    def __call__(self, batch):
        if( self.mode == "test" or self.mode == "val" ):
            # test_batch will always have a single datapoint which needs to be replicated num_test_avg times
            # repeat last test sample to make it divisible by test_batch_size since drop_last=False in test loader
            #since drop_last=False in test loader, record actual test_batch_size for last batch
            actual_batch_size = torch.tensor(len(batch)).long()

            # repeat each test sample num_test_avg times sequentially
            new_batch = []
            for d in batch:
                if( self.mode == "test" ):
                    new_batch += [d for k in range(self.num_test_avg)]
                else:
                    new_batch += [d for k in range(self.num_val_avg)]
            batch = new_batch
        else:
            actual_batch_size = torch.tensor(-1).long()
        # construct features: features_1 (answer txt) will have different segment embeddings than features_2 (remaining txt)
        features_1 = []
        features_2 = []
        for d in batch:
            # randomly sample num_examples from each class in train set for datapoint d
            examples_many_per_class = []
            # examples_each_class stores one example from each class
            examples_one_per_class = []
            labels = list(range(d["min"], d["max"] + 1))
            for label in labels:
                examples_class = self.data_meta[d["task"]]["examples"][label]
                # remove current datapoint d from examples_class by checking unique booklet identifiers
                examples_class = [ex for ex in examples_class if ex["bl"] != d["bl"]]
                # sampling num_examples without replacement
                if( len(examples_class) < self.num_examples ):
                    random.shuffle(examples_class)
                    examples_class_d = examples_class
                else:
                    examples_class_d = random.sample(examples_class, self.num_examples)
                #print("examples_class_d[0]", examples_class_d[0])
                if( len(examples_class_d) > 1 ):
                    examples_one_per_class += [examples_class_d[0]]
                    examples_many_per_class += examples_class_d[1:]
                elif( len(examples_class_d) == 1 ):
                    examples_one_per_class += [examples_class_d[0]]
                    examples_many_per_class += []
                else:
                    examples_one_per_class += []
                    examples_many_per_class += []
            
            # construct input txt 
            if( self.use_demographic ):
                input_txt = "score this answer written by {} {} student: ".format(self.gender_map[d["sx"]], self.race_map[d["rc"]]) + d['txt']
            else:
                input_txt = "score this answer: " + d['txt']
            if( self.generative_model ):
                input_txt = "score this answer: " + d['txt'] + " [SEP]"
            features_1.append(input_txt)
            
            # add range of possible scores for datapoint d
            examples_txt = " scores: " + " ".join([ (self.label_to_text[label] + " ") for label in range(d["min"], d["max"] + 1) ])
            # add question text
            examples_txt += "[SEP] question: {} [SEP] ".format(self.task_to_question[d["task"]])
            #todo trunate questions
            # shuffle examples across classes
            random.shuffle(examples_one_per_class)
            random.shuffle(examples_many_per_class)
            # since truncation might occur if text length exceed input length to LM, ensuring one example from each class is present first
            examples_d = examples_one_per_class + examples_many_per_class
            curr_len = len(input_txt.split(" ") + examples_txt.split(" "))
            for i in range(len(examples_d)):
                example = examples_d[i]
                example_txt_tokens = example['txt'].split(" ")
                curr_example_len = len(example_txt_tokens)
                example_txt = " ".join(example_txt_tokens[:self.trunc_len])
                example_label = (example['l1'] if example['l1']>=0 else example['l2'])
                # " [SEP] " at the end of the last example is automatically added by tokenizer
                if( i == (len(examples_d)-1) ):
                    if( self.use_demographic ):
                        examples_txt += ( " example written by {} {} student: ".format(self.gender_map[example["sx"]], self.race_map[example["rc"]]) + example_txt + " score: " + self.label_to_text[example_label] )
                    else:
                        examples_txt += ( " example: " + example_txt + " score: " + self.label_to_text[example_label] )
                else:
                    if( self.use_demographic ):
                        examples_txt += ( " example written by {} {} student: ".format(self.gender_map[example["sx"]], self.race_map[example["rc"]]) + example_txt + " score: " + self.label_to_text[example_label] + " [SEP] " )
                    else:
                        examples_txt += ( " example: " + example_txt + " score: " + self.label_to_text[example_label] + " [SEP] " )
                # stop adding more examples once max_seq_len is reached
                if( (curr_example_len + curr_len) > self.max_seq_len):
                    break
                else:
                    curr_len += curr_example_len
            features_2.append(examples_txt)
        
        inputs = tokenize_function(self.tokenizer, features_1, features_2)
        # construct labels
        labels  =  torch.tensor([ (d['l1']-d["min"]) if d['l1']>=0 else (d['l2']-d["min"]) for d in batch]).long()
        max_labels = torch.tensor([( d["max"]-d["min"]+1) for d in batch]).long()
        inputs['labels'] = labels
        
        return {"inputs" : inputs, "max_labels" : max_labels, "actual_batch_size" : actual_batch_size}


class CollateWraperInContextTuningMetaLearning_Math(CollateWraperParent):
    """
    This is for the math problem input scoring notation
    todo define data's attribute and so on...
    """
    def __init__(self, tokenizer, data_meta, task_to_question, num_examples, trunc_len, mode, num_test_avg=1,
                 num_val_avg=1, test_batch_size=1, val_batch_size=1, max_seq_len=512, generative_model=False,
                 use_demographic=False, args = None):
        super().__init__(tokenizer, min_label=1)
        # meta learning via in-context tuning
        self.data_meta = data_meta
        self.num_examples = num_examples
        self.trunc_len = trunc_len
        self.label_to_text = {
            0: "wrong",
            1: "poor",
            2: "fair",
            3: "good",
            4: "excellent"
        }

        # demographic information
        self.use_demographic = use_demographic

        # get token IDs for labels using GPT2 tokenizer
        self.label_to_label_id = {}
        for label in self.label_to_text:
            self.label_to_label_id[label] = self.tokenizer(self.label_to_text[label])['input_ids']

        self.mode = mode
        self.num_test_avg = num_test_avg
        self.num_val_avg = num_val_avg
        self.task_to_question = task_to_question
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size
        # adding an extra 50 words in case num_tokens < num_words
        self.max_seq_len = max_seq_len + 50
        self.generative_model = generative_model
        self.args = args

    def __call__(self, batch):

        if (self.mode == "test" or self.mode == "val"):

            # since drop_last=False in test loader, record actual test_batch_size for last batch
            actual_batch_size = torch.tensor(len(batch)).long()

            # repeat each test sample num_test_avg times sequentially
            new_batch = []

            for d in batch:

                if (self.mode == "test"):
                    new_batch += [d for k in range(self.num_test_avg)]
                else:
                    new_batch += [d for k in range(self.num_val_avg)]
            batch = new_batch
        else:
            actual_batch_size = torch.tensor(-1).long()

        features_1 = []
        features_2 = []


        sbert_inputs = []

        bl_list = []

        question_length = []
        for d in batch:
            bl_list.append(d['bl'])
            # randomly sample num_examples from each class in train set for datapoint d
            examples_many_per_class = []
            # examples_each_class stores one example from each class
            examples_one_per_class = []
            labels = list(range(d["min"], d["max"] + 1))

            for label in labels:
                examples_class = self.data_meta[d["task"]]["examples"][label]
                # remove current datapoint d from examples_class by checking unique booklet identifiers
                examples_class = [ex for ex in examples_class if ex["bl"] != d["bl"]]
                # sampling num_examples without replacement
                if (len(examples_class) < self.num_examples):
                    if self.args.seed != -1:
                        random.seed(self.args.seed)
                    random.shuffle(examples_class)
                    examples_class_d = examples_class
                else:
                    if self.args.seed != -1:
                        random.seed(self.args.seed)
                    examples_class_d = random.sample(examples_class, self.num_examples)
                # print("examples_class_d[0]", examples_class_d[0])
                if (len(examples_class_d) > 1):
                    examples_one_per_class += [examples_class_d[0]]
                    examples_many_per_class += examples_class_d[1:]
                elif (len(examples_class_d) == 1):
                    examples_one_per_class += [examples_class_d[0]]
                    examples_many_per_class += []
                else:
                    examples_one_per_class += []
                    examples_many_per_class += []

            # construct input txt
            input_txt = "grade: " + d['txt']
            if (self.generative_model):
                input_txt = "grade: " + d['txt'] + " [SEP]"
            features_1.append(input_txt)
            sbert_inputs.append("{} question: {}".format(input_txt, self.task_to_question[d["task"]]))

            # add range of possible scores for datapoint d

            if not self.args.no_scale:
                examples_txt = " scores: " + " ".join(
                    [(self.label_to_text[label] + " ") for label in range(d["min"], d["max"] + 1)]) + '[SEP]'
            else:
                examples_txt = ""
            # add question text
            if not self.args.no_question:
                if self.args.question_id:
                    examples_txt += " question id: {} [SEP] ".format(d["task"])
                else:
                    question_txt_tokens = self.task_to_question[d["task"]].split(" ")
                    example_txt = " ".join(question_txt_tokens[:self.trunc_len])
                    examples_txt += " question: {} [SEP] ".format(example_txt)


                question_length.append( (len(self.tokenizer.encode(examples_txt))))
            else:
                question_length.append(0)
            # shuffle examples across classes
            if self.args.seed != -1:
                random.seed(self.args.seed)
            random.shuffle(examples_one_per_class)
            if self.args.seed != -1:
                random.seed(self.args.seed)
            random.shuffle(examples_many_per_class)
            # since truncation might occur if text length exceed input length to LM, ensuring one example from each class is present first
            examples_d = examples_one_per_class + examples_many_per_class
            if self.num_examples < len(examples_d):
                examples_d = examples_d[0:self.num_examples]
            elif self.num_examples == 0:
                examples_d = []
            curr_len = len(input_txt.split(" ") + examples_txt.split(" "))
            for i in range(len(examples_d)):
                example = examples_d[i]
                # print(example)
                example_txt_tokens = example['txt'].split(" ")
                curr_example_len = len(example_txt_tokens)
                example_txt = " ".join(example_txt_tokens[:self.trunc_len])
                example_label = example['l1']
                # " [SEP] " at the end of the last example is automatically added by tokenizer
                if (i == (len(examples_d) - 1)):

                    examples_txt += (" example: " + example_txt + " score: " + self.label_to_text[example_label])
                else:

                    examples_txt += (" example: " + example_txt + " score: " + self.label_to_text[
                            example_label] + " [SEP] ")
                # stop adding more examples once max_seq_len is reached
                if ((curr_example_len + curr_len) > self.max_seq_len):
                    break
                else:
                    curr_len += curr_example_len
            features_2.append(examples_txt)

        inputs = tokenize_function(self.tokenizer, features_1, features_2)


        #store example counts
        counts = inputs['input_ids'] == self.tokenizer.encode('[SEP]')[1]
        counts = counts.sum(dim=1)


        # construct labels
        labels = torch.tensor([(d['l1'] - d["min"]) for d in batch]).long()
        min_list = [d['min'] for d in batch]
        # store max_label for each d in batch
        max_labels = torch.tensor([(d["max"] - d["min"] + 1) for d in batch]).long()
        inputs['labels'] = labels
        #inputs['l1'] = torch.tensor([d['l1'] for d in batch] ).long()
        question_length = torch.tensor(question_length).long()

        return {"inputs": inputs, "max_labels": max_labels, "actual_batch_size": actual_batch_size,
                'counts': counts, 'bl':bl_list, 'min': min_list, 'q_length': question_length, 'sbert': sbert_inputs}


class CollateWraperInContextTuning(CollateWraperParent):
    def __init__(self, tokenizer, min_label, max_label, examples_train, num_examples, trunc_len, mode, num_test_avg=1, 
                test_batch_size=1, max_seq_len=512):
        super().__init__(tokenizer, min_label)  
        self.max_label = max_label
        # in context tuning
        self.examples_train = examples_train
        self.num_examples = num_examples
        #todo check if suitable for our data
        self.trunc_len = trunc_len
        self.mode = mode
        self.num_test_avg = num_test_avg
        self.test_batch_size = test_batch_size
        # adding an extra 50 words in case num_tokens < num_words
        self.max_seq_len = max_seq_len + 50

    
    def __call__(self, batch):
        if( self.mode == "test" ):
            #since drop_last=False in test loader, record actual test_batch_size for last batch
            actual_batch_size = torch.tensor(len(batch)).long()

            new_batch = []
            for d in batch:
                new_batch += [d for k in range(self.num_test_avg)]
            batch = new_batch
        else:
            actual_batch_size = torch.tensor(-1).long()

        # construct features: features_1 (answer txt) will have different segment embeddings than features_2 (remaining txt)
        features_1 = []
        features_2 = []
        for d in batch:
            # randomly sample num_examples from each class in train set
            examples_d = []
            labels = list(range(self.min_label, self.max_label + 1))
            for label in labels:
                examples_class = self.examples_train[label]
                # remove current datapoint d from examples_class by checking unique booklet identifiers
                examples_class = [ex for ex in examples_class if ex["bl"] != d["bl"]]
                # sampling num_examples without replacement
                if( len(examples_class) < self.num_examples ):
                    random.shuffle(examples_class)
                    examples_class_d = examples_class
                else:
                    examples_class_d = random.sample(examples_class, self.num_examples)
                examples_d += examples_class_d
            
            # construct input txt 
            input_txt = d['txt']
            features_1.append(input_txt)
            
            # shuffle examples across classes
            examples_txt = ""
            random.shuffle(examples_d)
            curr_len = len(input_txt.split(" ") + examples_txt.split(" "))
            for i in range(len(examples_d)):
                example = examples_d[i]
                #todo use bert tokenizer check for it: dont forget to delete cls and sep
                example_txt_tokens = example['txt'].split(" ")
                curr_example_len = len(example_txt_tokens)
                example_txt = " ".join(example_txt_tokens[:self.trunc_len])
                example_label = (example['l1'] if example['l1']>=0 else example['l2']) - self.min_label
                # " [SEP] " at the end of the last example is automatically added by tokenizer
                if( i == (len(examples_d) - 1) ):
                    examples_txt += ( example_txt + " score: " + str(example_label) )
                else:
                    examples_txt += ( example_txt + " score: " + str(example_label) + " [SEP] " )
                # stop adding more examples once max_seq_len is reached
                if( (curr_example_len + curr_len) > self.max_seq_len):
                    break
                else:
                    curr_len += curr_example_len
            features_2.append(examples_txt)
        
        inputs = tokenize_function(self.tokenizer, features_1, features_2)

        # construct labels
        labels  =  torch.tensor([d['l1'] if d['l1']>=0 else d['l2'] for d in batch]).long() - self.min_label
        inputs['labels'] = labels
        
        return {"inputs" : inputs, "actual_batch_size" : actual_batch_size}