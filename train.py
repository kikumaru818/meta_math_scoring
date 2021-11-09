from collections import defaultdict
from utils.load_data import load_dataset
from utils.utils import open_json, safe_makedirs
import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import models
import argparse
from utils import utils
import sys
import yaml

def add_learner_params():
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--name', default='demo',help='Name for the experiment')
    parser.add_argument('--nodes', default='', help='slurm nodes for the experiment')
    parser.add_argument('--slurm_partition', default='',
                        help='slurm partitions for the experiment')
    # problem definition
    parser.add_argument('--lm', default='bert-base-uncased',
                        help='Base Language model'
                        )
    # parser.add_argument('--task', default="Grade 4/2017_DBA_DR04_1715RE4T08G04_09",help='Dataset')
    parser.add_argument('--task', default="all",help='Dataset')
    parser.add_argument('--generate', default='none',help= 'generate last token, none or score or verb')
    parser.add_argument('--losses', default="cce",help='; separated losses among cce, qwp e.g. cce;qwp')
    parser.add_argument('--labels2', action='store_false', help='consider second label with average weight')
    # optimizer params
    parser.add_argument('--lr_schedule', default='warmup-const')
    parser.add_argument('--opt', default='sgd',
                        help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=100, type=int,
                        help='The number of optimizer updates')
    parser.add_argument('--warmup', default=0, type=float,
                        help='The number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Base learning rate')
    # trainer params
    #parser.add_argument('--save_freq', default=1000000000000,
    #                    type=int, help='Frequency to save the model')
    parser.add_argument('--eval_freq', default=1,
                        type=int, help='Evaluation frequency')
    parser.add_argument('--workers', default=2, type=int,
                        help='The number of data loader workers')
    parser.add_argument('--seed', default=999, type=int, help='Random seed')
    # parallelizm params:
    parser.add_argument('--batch_size', default=100, type=int)
    # extras
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--neptune', action='store_true')
    
    
    params = parser.parse_args()
    return params



def main():
    #Bookkeeping
    args = add_learner_params()
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    args.root = 'logs/'+args.name+'/'
    safe_makedirs(args.root)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'no gpu found!'
    with open(args.root+'config.yml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)
    if args.neptune:
        import neptune.new as neptune
        project = "arighosh/naep"
        run = neptune.init(
                project=project,
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                capture_hardware_metrics =False,
                name =args.name,
                )  
        run["parameters"] = vars(args)
    
    ###
    model = models.finetune.BaseModel(args, device=device)
    model.prepare_data()
    model.prepare_model()
    #
    if args.neptune and not isinstance(model.majority_class, list):
        run["parameters/majority_class"] = model.majority_class
        run["parameters/n_labels"] = model.max_label-model.min_label +1

    ###
    cur_iter = 0
    continue_training = cur_iter < args.iters
    #
    data_time, it_time = 0, 0
    #metric
    best_metrics =  defaultdict(lambda: float('-inf'))
    best_metrics_with_valid =  defaultdict(lambda: float('-inf'))
    best_valid_metrics =  defaultdict(lambda: float('-inf'))
    while continue_training:
        train_loader, valid_loader,test_loader = model.dataloaders(
            iters=args.iters)
        train_logs = []
        model.train()
        start_time = time.time()
        cur_iter += 1
        for _, batch in enumerate(train_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            data_time += time.time() - start_time
            logs = model.train_step(batch)  
            train_logs.append(logs)
            
        # save logs for the batch
        if cur_iter % args.eval_freq == 0 or cur_iter >= args.iters:
            test_start_time = time.time()
            test_logs,valid_logs = [], []
            for batch in test_loader:
                batch = {k: v.to(device)  if torch.is_tensor(v) else v for k, v in batch.items()}
                logs = model.test_step(batch)
                test_logs.append(logs)
            
            for batch in valid_loader:
                batch = {k: v.to(device)  if torch.is_tensor(v) else v for k, v in batch.items()}
                logs = model.test_step(batch)
                valid_logs.append(logs)
                
            test_logs = utils.agg_all_metrics(test_logs)
            valid_logs = utils.agg_all_metrics(valid_logs)
            test_it_time = time.time()-test_start_time
            it_time += time.time() - start_time
            train_logs = utils.agg_all_metrics(train_logs)
            #Logging
            for key in valid_logs:
                if float(valid_logs[key])>best_valid_metrics[key]:
                    best_valid_metrics[key] = float(valid_logs[key])
                    best_metrics_with_valid[key] = float(test_logs[key])
            for key in test_logs:
                if float(test_logs[key])>best_metrics[key]:
                    best_metrics[key] = float(test_logs[key])
            if args.neptune:
                for key in test_logs:
                    run["metrics/test/"+key].log(test_logs[key])
                for key in valid_logs:
                    run["metrics/valid/"+key].log(valid_logs[key])
                for key in train_logs:
                    run["metrics/train/"+key].log(train_logs[key])
                for key in best_metrics:
                    run["metrics/test/best_"+key].log(best_metrics[key])
                for key in best_metrics_with_valid:
                    run["metrics/test/best_"+key+"_with_valid"].log(best_metrics_with_valid[key])
                run["logs/train/it_time"].log(it_time)
                run["logs/cur_iter"].log(cur_iter)
            data_time, it_time = 0, 0
            train_logs = []
        if cur_iter >= args.iters:
            continue_training = False
            break
        start_time = time.time()



if __name__ == '__main__':
    main()