#from transformers.utils.logging import remove_handler
import utils
from utils.utils import open_json,dump_json, human_kappa
from collections import defaultdict
import hashlib
import random
import os
SEED = 999
RAW_DIR = "../data/NAEP_AS_Challenge_Data/Items for Item-Specific Models/"
HASH_PATH =  'data/tasks_hash.json'
SUBMISSION_HASH_PATH =  'data/submission_tasks_hash.json'
#schema = {'bl':0, 'l1':1, 'l2':2, 'sx':3, 'rc':4, 'an':5, 'wc':6, 'txt':7}
def safe_parse(val):
    try:
        return int(val)
    except ValueError:
        return -1

def compute_mapper(header):
    words = header.strip('\n').split(',')
    words = {w:i for i,w in enumerate(words)}
    #BookletNumber	Score1	Score2	DSEX	SRACE10	ACCnum	WordCount	ReadingTextResponse
    schema = {'bl': words["BookletNumber"],'l1':words.get('Score1',-1), 'l2':words.get('Score2', -1), 'sx':words['DSEX'], 'rc':words['SRACE10'], 'an': words['ACCnum'],'wc': words['WordCount'], 'txt': words['ReadingTextResponse']
    }
    return schema

def parse_csv(filename):
    data = []
    path_ = os.path.normpath(filename)
    with open(path_, 'r',encoding='utf-8-sig') as fp:
        lines = fp.readlines()
        schema =  compute_mapper(lines[0])
        for line in lines[1:]:
            line = line.strip('\n')
            words = line.split(',')
            out = {}
            for k,v in schema.items():
                if v==-1:
                    out['l1']=out['l2']=1
                    continue
                if k =='txt':
                    out[k] =  ','.join(words[v:])
                else:
                    out[k] = words[v] if k not in {'l1', 'l2', 'wc'} else safe_parse(words[v])
            if out['l1']==-1 and out['l2'] ==-1:
                continue
            data.append(out)
    return data
def compute_distribution(output):
    append_keys = {}
    for k,v in output.items():
        dist, new_key, total_count  = defaultdict(float),k+'_dist', 0.
        for d in v:
            n_rating =  (d['l1']>=0) + (d['l2']>=0)
            total_count +=n_rating
            if d['l1']!=-1: dist[d['l1']] += 1./n_rating
            if d['l2']!=-1: dist[d['l2']] += 1./n_rating
        for l in dist:
            dist[l] /=total_count
        append_keys[new_key] = dist
    for k in append_keys:
        output[k] = append_keys[k]

def compute_hashes(data):
    hashes = {}
    for partition in ['train', 'valid', 'test']:
        orders = [d['bl'] for d in data[partition]]
        hashes[partition] = hashlib.md5(''.join(orders).encode('utf-8')).hexdigest()
    return hashes

def create_splits(data,train,valid):
    n = len(data)
    n_train, n_val = int(n*train),int(n*(train+valid))
    random.Random(SEED).shuffle(data)
    train_data, val_data, test_data = data[:n_train], data[n_train:n_val], data[n_val:]
    output = {'train':train_data, 'valid':val_data, 'test':test_data}
    compute_distribution(output)
    return output
    

def load_dataset(task, create_hash, train,valid):
    """
        Returns {'train/val/test': [{key:value}], 'train/val/test_dist':{label:percentage}}
        feature schema : {'bl':0, 'l1':1, 'l2':2, 'sx':3, 'rc':4, 'an':5, 'wc':6, 'txt':7} word val represent column number of 
        ['BookletNumber', 'Score1', 'Score2', 'DSEX', 'SRACE10', 'ACCnum', 'WordCount', 'ReadingTextResponse']
    """
    train_file = RAW_DIR+task+"/"+task.split('/')[1]+"_Training.csv"
    val_file = RAW_DIR+task+"/"+task.split('/')[1]+"_Validation_DS&SS.csv"
    data, data_2  =  parse_csv(train_file), parse_csv(val_file)
    data.extend(data_2)
    best_kappa = human_kappa(data)
    data = create_splits(data, train,valid)
    hashes =  compute_hashes(data)
    data['human_kappa'] = best_kappa
    tasks_hash = open_json(HASH_PATH)
    if create_hash:
        tasks_hash[task] = {'train':hashes['train'], 'valid':hashes['valid'], 'test':hashes['test']}
        dump_json(HASH_PATH, tasks_hash)
    else:
        assert tasks_hash.get(task, {}).get('train', "0")==hashes['train'], 'Train Split is not Matching.'
        assert tasks_hash.get(task, {}).get('valid', "0")==hashes['valid'], 'Validation Split is not Matching.'
        assert tasks_hash.get(task, {}).get('test', "0")==hashes['test'], 'Test Split is not Matching.'
    return data

def submission_load_dataset(task, create_hash, train,valid):
    """
        Returns {'train/val/test': [{key:value}], 'train/val/test_dist':{label:percentage}}
        ['BookletNumber', 'Score1', 'Score2', 'DSEX', 'SRACE10', 'ACCnum', 'WordCount', 'ReadingTextResponse']
    """
    train_file = RAW_DIR+task+"/"+task.split('/')[1]+"_Training.csv"
    val_file = RAW_DIR+task+"/"+task.split('/')[1]+"_Validation_DS&SS.csv"
    test_file = RAW_DIR+task+"/"+task.split('/')[1]+"_Test.csv"
    data, data_2, test_data  =  parse_csv(train_file), parse_csv(val_file), parse_csv(test_file)
    data.extend(data_2)
    best_kappa = human_kappa(data)
    data = create_splits(data, train,valid)
    data['human_kappa'] = best_kappa
    data['test'] =  test_data
    hashes =  compute_hashes(data)
    tasks_hash = open_json(SUBMISSION_HASH_PATH)
    if create_hash:
        tasks_hash[task] = {'train':hashes['train'], 'valid':hashes['valid'], 'test':hashes['test']}
        dump_json(SUBMISSION_HASH_PATH, tasks_hash)
    else:
        assert tasks_hash.get(task, {}).get('train', "0")==hashes['train'], 'Train Split is not Matching.'
        assert tasks_hash.get(task, {}).get('valid', "0")==hashes['valid'], 'Validation Split is not Matching.'
        assert tasks_hash.get(task, {}).get('test', "0")==hashes['test'], 'Test Split is not Matching.'
    return data

    

def main():
    tasks = open_json('data/tasks.json')
    for task in tasks:
        data = load_dataset(task, create_hash=False,train=0.6, valid=0.2)
        pass
    
    


if __name__ == '__main__':
    main()