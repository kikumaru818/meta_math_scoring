from transformers.utils.logging import remove_handler
from utils.utils import open_json,dump_json
from collections import defaultdict
import hashlib
import random
import os
SEED = 999
RAW_DIR = "../data/NAEP_AS_Challenge_Data/Items for Item-Specific Models/"
HASH_PATH =  'data/tasks_hash.json'
schema = {'bl':0, 'l1':1, 'l2':2, 'sx':3, 'rc':4, 'an':5, 'wc':6, 'txt':7}
def safe_parse(val):
    try:
        return int(val)
    except ValueError:
        return -1

def parse_csv(filename):
    data = []
    path_ = os.path.normpath(filename)
    #filename = filename.replace(' ', '\\ ')
    with open(path_, 'r') as fp:
        lines = fp.readlines()
        for line in lines[1:]:
            line = line.strip('\n')
            words = line.split(',')
            out = {}
            for k,v in schema.items():
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
    for partition in ['train', 'val', 'test']:
        orders = [d['bl'] for d in data[partition]]
        hashes[partition] = hashlib.md5(''.join(orders).encode('utf-8')).hexdigest()
    return hashes

def create_splits(data,train,val, test):
    n = len(data)
    n_train, n_val = int(n*train),int(n*(train+val))
    random.Random(SEED).shuffle(data)
    train_data, val_data, test_data = data[:n_train], data[n_train:n_val], data[n_val:]
    output = {'train':train_data, 'val':val_data, 'test':test_data}
    compute_distribution(output)
    hashes =  compute_hashes(output)
    return output,hashes
    

def load_dataset(task, create_hash, train,val, test):
    """
        Returns {'train/val/test': [{key:value}], 'train/val/test_dist':{label:percentage}}
        feature schema : {'bl':0, 'l1':1, 'l2':2, 'sx':3, 'rc':4, 'an':5, 'wc':6, 'txt':7} word val represent column number of 
        ['BookletNumber', 'Score1', 'Score2', 'DSEX', 'SRACE10', 'ACCnum', 'WordCount', 'ReadingTextResponse']
    """
    train_file = RAW_DIR+task+"/"+task.split('/')[1]+"_Training.csv"
    val_file = RAW_DIR+task+"/"+task.split('/')[1]+"_Validation_DS&SS.csv"
    data, data_2  =  parse_csv(train_file), parse_csv(val_file)
    data.extend(data_2)
    data, hashes = create_splits(data, train,val, test)
    tasks_hash = open_json(HASH_PATH)
    if create_hash:
        tasks_hash[task] = {'train':hashes['train'], 'val':hashes['val'], 'test':hashes['test']}
        dump_json(HASH_PATH, tasks_hash)
    else:
        assert tasks_hash.get(task, {}).get('train', "0")==hashes['train'], 'Train Split is not Matching.'
        assert tasks_hash.get(task, {}).get('val', "0")==hashes['val'], 'Validation Split is not Matching.'
        assert tasks_hash.get(task, {}).get('test', "0")==hashes['test'], 'Test Split is not Matching.'
    return data

    

def main():
    tasks = open_json('data/tasks.json')
    for task in tasks:
        data = load_dataset(task, create_hash=False,train=0.6, val=0.2, test=0.2)
        pass
    
    


if __name__ == '__main__':
    main()