import torch
import torch.backends.cudnn as cudnn
import argparse
import random
import yaml
import os
import time
import json
import pathlib
import time
import pandas as pd
from torch import nn
from tqdm import tqdm
from utils import utils
from transformers import logging
from models.lm_incontext_tuning_with_meta_learning import LanguageModelInContextTuningMetaLearning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# disable warnings in hugging face logger
logging.set_verbosity_error()


def add_learner_params():
    parser = argparse.ArgumentParser(description='naep_as_challenge')
    
    parser.add_argument('--name', default='train', help='name for the experiment')
    parser.add_argument('--math', default=True, help='apply on math scoring dataset')
    parser.add_argument('--all', action='store_true', help='Only store all test and valid result,not seperate by tasks')
    parser.add_argument('--meta', action='store_true', help='question split')
    parser.add_argument('--finetune', action = 'store_true', help='the finetune mode for few-shot learning')

    # problem definition
    parser.add_argument('--lm', default='tbs17/MathBERT', help='Base Language model') #tbs17/MathBERT-custom #bert-base-uncased
    parser.add_argument('--tok', default='bert-base-uncased', help='Base Language model')
    # optimizer params
    parser.add_argument('--lr_schedule', default='warmup-const')
    parser.add_argument('--opt', default='adam', help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=100, type=int, help='number of epochs')
    parser.add_argument('--warmup', default=0, type=float, help='number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=2e-5, type=float, help='base learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    # trainer params
    parser.add_argument('--save_freq', default=1, type=int, help='epoch frequency to save the model')
    parser.add_argument('--eval_freq', default=1, type=int, help='epoch frequency for evaluation')
    parser.add_argument('--workers', default=4, type=int, help='number of data loader workers')
    parser.add_argument('--seed', default=999, type=int, help='random seed')    
    # extras
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--save', action='store_true', help='save model every save_freq epochs')
    parser.add_argument('--debug', action='store_true', help='debug mode with less data')
    # input specifications
    #parser.add_argument('--passage', action='store_true', help='add passage text to input answer text')
    parser.add_argument('--no_question', action='store_true', help='add question text to input answer text')
    parser.add_argument('--question_id', action='store_true')
    parser.add_argument('--no_scale', action='store_true', help='if using scale')
    parser.add_argument('--spell_check', action='store_true', help='spell check answer text only')
    # in context tuning
    # if in context tuning is True, then batch_size = batch_size * num_augment at train time
    parser.add_argument('--in_context_tuning', action='store_true', help='apply in-context tuning')
    # ensure test_batch_size*num_test_avg = actual test_batch_size can be loaded onto GPU
    parser.add_argument('--num_test_avg', default=1, type=int, help='number of trials at test time for each test datapoint with different randomly sampled examples to calculate average on')
    parser.add_argument('--num_val_avg', default=1, type=int, help='number of trials at val time for each val datapoint with different randomly sampled examples to calculate average on')
    parser.add_argument('--num_examples', default=25, type=int, help='number of examples from each class to add to input')
    parser.add_argument('--new_examples', default=0, type=int,
                        help='new number of examples from each class to add to input')
    parser.add_argument('--num_augment', default=1, type=int, help='num of times to duplicate training datapoint with different randomly sampled examples')
    parser.add_argument('--trunc_len', default=70, type=int, help='max number of words in each example')
    parser.add_argument('--trunc_len_q', default=70, type=int, help='max number of words in each question')
    # meta learning
    parser.add_argument('--meta_learning', action='store_true', help='apply meta learning via in-context tuning')
    parser.add_argument('--meta_learning_single', action='store_true', help='apply meta learning via in-context tuning only on one task')
    # automatic mixed precision training -> faster training but might affect accuracy slightly
    parser.add_argument('--amp', action='store_true', help='apply automatic mixed precision training')
    # generative models = GPT2
    parser.add_argument('--generative_model', action='store_true', help='use generative models like GPT2')
    parser.add_argument('--sbert',action='store_true',help='use sbert embed meta method')
    parser.add_argument('--freeze_bert',action='store_true' )
    # data_loading
    parser.add_argument('--data_folder', default="small_meta.json", help='folder to use to load dataset splits') #"data_split_answer_spell_checked_submission"
    parser.add_argument('--alias', default='')
    parser.add_argument('--cross_val_fold', default=0, type=int, help='cross validation fold/split to use')
    parser.add_argument('--fold', default=None, type=int, help='define which fold to use')
    # add demographic information for fairness analysis - generative models don't have this option
    parser.add_argument('--demographic', action='store_true', help='use demographic information of student')
    # running on unity server
    parser.add_argument('--unity_server', action='store_true', help='use demographic information of student')
    parser.add_argument('--eval', action='store_true', help='only evaluate')

    params = parser.parse_args()
    
    return params


def train_meta(args, run, device, saved_models_dir):
    result_file = '../data/math/results.csv'

    if( args.amp ):
        # using pytorch automatic mixed precision (fp16/fp32) for faster training
        # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    alias = args.alias

    # load list of tasks for meta learning

    with open("data/tasks_math{}.json".format(alias), "r") as f:
        task_list = json.load(f)
    # load task to question map for each task

    with open("data/tasks_to_question_math{}.json".format(alias), "r") as f:
        task_to_question = json.load(f)
    
    # prepare data and model for training
    if 'saved_models' in args.lm:
        model_path = os.path.abspath(args.lm)
        args.lm = model_path
    model = LanguageModelInContextTuningMetaLearning(args, device, task_list, task_to_question, None, None)
    model.prepare_data()
    task_list = model.task_list
    model.prepare_model()

    # training variables
    cur_iter = 0
    # dict of metric variables = kappa for each task trained in meta learning
    metrics = {}
    if args.all:
        check_list = ['all']
    else:
        check_list = task_list + ['all']
    for task in check_list:
        metrics[task] = {}
        # best test kappa
        metrics[task]["best_test_metric"] = -1
        # test kappa corresponding to best valid kappa
        metrics[task]["test_metric_for_best_valid_metric"] = -1
        # best valid kappa
        metrics[task]["best_metric"] = -1
        metrics[task]["best_valid_metric"] = -1

    # training loop
    loss_func = nn.CrossEntropyLoss()
    disable_tqdm = False

    if args.eval:
        _, valid_loaders, _ = model.dataloaders()
        cur_iter += 1
        # eval and test epoch separately for every task using single meta learned model
        model.eval()
        if ((cur_iter % args.eval_freq == 0) or (cur_iter >= args.iters)):
            # dict of logs per task
            test_logs, valid_logs = {}, {}
            for task in task_list:
                test_logs[task], valid_logs[task] = [], []
            # eval epoch
            epoch_pbar = tqdm(task_list, desc="Iteration", disable=disable_tqdm)
            for i, task in enumerate(task_list):
                valid_loader = valid_loaders[task]
                for batch in valid_loader:
                    batch_n = {k: v.to(device) for k, v in batch.items() if type(v) is not list}
                    if args.sbert:
                        batch_n['inputs']['input_ids'] = batch['sbert']
                    logs = model.eval_step(batch_n)
                    logs.update({'bl': batch['bl']})
                    length = int(len(batch['min']) / args.num_val_avg)
                    logs.update({'bl': batch['bl'], 'min': batch['min'][0:length]})
                    valid_logs[task].append(logs)
                epoch_pbar.update(1)
            epoch_pbar.close()

            valid_all = []

            for task in task_list:
                valid_all += valid_logs[task]
                if not args.all:
                    valid_logs[task] = utils.agg_all_metrics(valid_logs[task])
            valid_logs['all'] = utils.agg_all_metrics(valid_all)
            metrics['valid_all'] = valid_logs['all']
            bl_pairs = metrics['valid_all']['bl_pair']
            metrics['valid_all'].pop('bl_pair')


            # push logs to neptune for each task
            if args.all:
                check_list = ['all']
            else:
                check_list = task_list + ['all']

            for task in check_list:
                if (len(valid_logs[task]) > 0):
                    average_score = float(valid_logs[task]["kappa"]) + float(valid_logs[task]["auc"]) + float(
                        valid_logs[task]["mse"])
                    if average_score >= metrics[task]["best_metric"]:
                        #metrics[task]["best_metric"] = average_score
                        metrics[task]["best_acc"] = valid_logs[task]["acc"]
                        metrics[task]["best_kappa"] = valid_logs[task]["kappa"]
                        metrics[task]["best_auc"] = valid_logs[task]["auc"]
                        metrics[task]["best_mse"] = valid_logs[task]["mse"]
                        if task != 'all':
                            metrics[task]['number_examples'] = len(model.data_meta[task]['train'])
                            metrics[task]['problem'] = task_to_question[task]

                        if task == 'all':
                            metrics[task]["num"] = args.new_examples
                            dir_model = saved_models_dir + args.name + "/" + "epoch_{}/".format(
                                cur_iter)
                            pathlib.Path(dir_model).mkdir(parents=True, exist_ok=True)
                            json.dump(metrics, open(os.path.join(dir_model, "log_history.json"), "w"), indent=2,
                                      ensure_ascii=False)
                            from utils.load_data_math import assign_label_to_csv
                            path_all = '../data/math/qc{}.csv'.format(args.alias)
                            path = dir_model + args.name + '.csv'
                            assign_label_to_csv(path, path_all, bl_pairs, fold=args.fold)


    while( cur_iter < args.iters ):
        train_loader, valid_loaders, test_loaders = model.dataloaders()
        start_time = time.time()
        cur_iter += 1

        # train epoch on one big trainset = union of trainsets across tasks in meta learning
        model.train()
        train_logs = []
        epoch_pbar = tqdm(train_loader, desc="Iteration", disable=disable_tqdm)

        for i, batch in enumerate(train_loader):
            if args.debug:
                if i == 100:
                    print('debug mode, end early')
                    break
            epoch_pbar.update(1)
            batch_n = {k: v.to(device) for k, v in batch.items() if type(v) is not list}
            if args.sbert:
                batch_n['inputs']['input_ids'] = batch['sbert']
            logs = model.train_step(batch_n, scaler, loss_func)
            logs.update({'bl': batch['bl'], 'min': batch['min'], 'counts':batch['counts'], 'q_length':batch['q_length']})
            train_logs.append(logs)

        epoch_pbar.close()
        # push logs to neptune
        train_it_time = time.time() - start_time
        train_logs = utils.agg_all_metrics(train_logs)

        # eval and test epoch separately for every task using single meta learned model 
        model.eval()
        if( (cur_iter % args.eval_freq == 0) or (cur_iter >= args.iters) ):
            # dict of logs per task
            test_logs, valid_logs = {}, {}
            for task in task_list:
                test_logs[task], valid_logs[task] = [], []
            # eval epoch
            epoch_pbar = tqdm(task_list, desc="Iteration", disable=disable_tqdm)
            for i, task in enumerate(task_list):
                if args.debug:
                    if i == 10:
                        print('debug mode, end early')
                        break
                epoch_pbar.update(1)
                valid_loader = valid_loaders[task]
                for batch in valid_loader:
                    batch_n = {k: v.to(device) for k, v in batch.items() if type(v) is not list}
                    if args.sbert:
                        batch_n['inputs']['input_ids'] = batch['sbert']
                    logs = model.eval_step(batch_n)
                    length = int(len(batch['min']) / args.num_val_avg)
                    logs.update({'bl': batch['bl'], 'min': batch['min'][0:length],
                                 'counts':batch['counts'], 'q_length':batch['q_length']})
                    valid_logs[task].append(logs)
            epoch_pbar.close()
            valid_all = []
            for task in task_list:
                valid_all += valid_logs[task]
                if not args.all:
                    valid_logs[task] = utils.agg_all_metrics(valid_logs[task])
            valid_logs['all'] = utils.agg_all_metrics(valid_all)
            metrics['valid_all'] = valid_logs['all']
            bl_pairs = metrics['valid_all']['bl_pair']
            metrics['valid_all'].pop('bl_pair')
            # push logs to neptune for each task
            if args.all:
                check_list = ['all']
            else:
                check_list = task_list + ['all']

            for task in check_list:
                if( len(valid_logs[task]) > 0 ):
                    average = float(valid_logs[task]["kappa"]) + float(valid_logs[task]["auc"])
                    if average >= metrics[task]["best_metric"]:
                        metrics[task]["best_metric"] = average
                        metrics[task]["best_acc"] = valid_logs[task]["acc"]
                        metrics[task]["best_kappa"] = valid_logs[task]["kappa"]
                        metrics[task]["best_auc"] = valid_logs[task]["auc"]
                        metrics[task]["best_mse"] = valid_logs[task]["mse"]
                        if task != 'all':
                            metrics[task]['number_examples'] = len(model.data_meta[task]['train'])
                            metrics[task]['problem'] = task_to_question[task]
                        if task == 'all':
                            dir_model = saved_models_dir + args.name + "/best/"
                            pathlib.Path(dir_model).mkdir(parents=True, exist_ok=True)
                            json.dump(metrics, open(os.path.join(dir_model, "log_history.json"), "w"), indent=2,
                                      ensure_ascii=False)
                            from utils.load_data_math import assign_label_to_csv
                            path_all = '../data/math/qc{}.csv'.format(args.alias)
                            path = dir_model + args.name + '.csv'
                            assign_label_to_csv(path, path_all, bl_pairs, fold=args.fold)

                            if args.save:
                                if args.sbert:
                                    torch.save(model.model.state_dict(), dir_model + 'best.model')
                                else:
                                    model.model.save_pretrained(dir_model)
                                    model.tokenizer.save_pretrained(dir_model)


def main(args):

    if ( torch.cuda.is_available() ):
        if ( args.unity_server ):
            # unity server saved models dir
            saved_models_dir = "saved_models/"
        else:
            # gypsum server saved models dir
            saved_models_dir = "saved_models/"
    else:
        # local saved models dir
        saved_models_dir = "saved_models/"
    
    # set random seed if specified
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'no gpu found!'
    # book keeping 
    # TODO P2: shouldn't we add a unique identifier to the config file
    args.root = 'logs/' + args.name + '/'
    utils.safe_makedirs(args.root)
    with open(args.root+'config.yml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)
    # link training to neptune
    # train model
    name = args.name

    if args.fold == -1:
        fold_list = range(1,5)
    else:
        fold_list = [args.fold]

    if args.new_examples == -1:
        example_list = [1, 3, 5, 7, 10, 25, 50, 80]
    else:
        example_list = [args.num_examples]

    for i in fold_list:
        for j in example_list:
            args.fold = abs(i)
            args.name = name + '_' + str(abs(i))
            args.name = args.name.replace('example','e'+str(abs(j)))
            #args.num_examples = j
            args.new_examples = j
            run = None
            run["parameters"] = vars(args)
            train_meta(args, run, device, saved_models_dir)





if __name__ == '__main__':
    args = add_learner_params()
    main(args)