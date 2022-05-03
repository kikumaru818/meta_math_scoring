import json
import pandas as pd
from collections import defaultdict
import ast
import numpy as np
import random
from langdetect import detect, detect_langs


def filter_description(task, key=None):
    # descript page information, no detail description
    task = clean(task)
    if len(str(task)) < 15:
        return False
    task = task.lower()
    if task == np.nan:
        return False
    if 'xiao wang' in str(task):
        return False
    if 'translate' in str(task) and 'chinese' in str(task):
        return False
    # description is not long enough

    try:
        lang = detect(task)
        langs = detect_langs(task)
    except:
        return False
    if lang != 'en':
        if langs[0].lang == 'en' and langs[0].prob > 0.5:
            return True
        return False
    # description contain broken words
    return True


def clean(task):
    task = task.replace(u'\r', u' ')
    task = task.replace(u'\n', u' ')
    task = task.replace(u'\\', u' ')
    task = task.replace(u'\xa0', u' ')
    task = task.replace('Modified from EngageNY ©Great Minds Disclaimer', ' ')
    task = task.replace('copied for free from openupresources.orgShow', ' ')
    broken_tokens = 'âð¥¸¶´ð·ð¤§¦'
    for i in broken_tokens:
        task = task.replace(i, ' ')

    task = task.split(' ')
    task = [t for t in task if len(t) > 0]
    if len(task) == 0:
        return np.nan
    task = ' '.join(task)
    return task


def prepare_data(path, alias = '_full'):
    df = pd.read_csv(path, index_col=0, encoding='utf-8')
    clean_df = df[df['cleaned_answer_text'].notnull()]
    clean_df = clean_df.rename(
        columns={"clean_full_problem":"cleaned_problem_body", "grade":'teacher_grades'})
    try:
        clean_df = clean_df[clean_df['cleaned_problem_body'].notnull()]
    except:
        clean_df = clean_df[clean_df['clean_full_problem'].notnull()]
    clean_df = clean_df[clean_df['teacher_grades'].notnull()]
    #0. only get useful colums
    clean_df = clean_df[['problem_id', 'problem_log_id', 'cleaned_problem_body','cleaned_answer_text','teacher_grades','folds']]
    #1 rename it with convience
    clean_df = clean_df.rename(columns={'problem_log_id': 'bl', "problem_id": "id", "cleaned_problem_body": "p", 'cleaned_answer_text':'txt', 'teacher_grades':'l1'})
    clean_df['l1'] -= 1
    clean_df['l1'] = clean_df['l1'].astype(int)
    #2. create json file map for problem_id to problem_body
    tr = clean_df.groupby(['id', 'p'])['id'].count()
    tr = tr.to_dict()
    task_map ={str(k[0]):clean(k[1])  for k in list(tr.keys())}
    task_list = list(task_map.keys())
    task_list = [str(l) for l in task_list]
    task_list_int = [int(l) for l in task_list]

    clean_df = clean_df.loc[clean_df['id'].isin(task_list_int)]
    #4. group example list by problem, each problem consider as the task
    tr = list(clean_df.groupby('id'))
    meta_data = {str(l[0]): l[1].to_dict('records') for l in tr}

    with open('data/tasks_math{}.json'.format(alias), 'w') as outfile:
        outfile.write(json.dumps(task_list, sort_keys=True, indent=4))
    with open('data/tasks_to_question_math{}.json'.format(alias), 'w') as outfile:
        outfile.write(json.dumps(task_map, sort_keys=True, indent=4))
    with open('data/meta{}.json'.format(alias), 'w') as outfile:
        outfile.write(json.dumps(meta_data, sort_keys=True, indent=4))

def clean_up_data_for_meta(path, alias='_full'):
    df = pd.read_csv(path, index_col=0, encoding='utf-8')
    clean_df = df[df['cleaned_answer_text'].notnull()]
    clean_df = clean_df.rename(
        columns={"clean_full_problem": "cleaned_problem_body", "grade": 'teacher_grades'})
    try:
        clean_df = clean_df[clean_df['cleaned_problem_body'].notnull()]
    except:
        clean_df = clean_df[clean_df['clean_full_problem'].notnull()]
    clean_df = clean_df[clean_df['teacher_grades'].notnull()]
    # 0. only get useful colums
    clean_df = clean_df[
        ['problem_id', 'problem_log_id', 'cleaned_problem_body', 'cleaned_answer_text', 'teacher_grades', 'folds']]
    # 1 rename it with convience
    clean_df = clean_df.rename(
        columns={'problem_log_id': 'bl', "problem_id": "id", "cleaned_problem_body": "p", 'cleaned_answer_text': 'txt',
                 'teacher_grades': 'l1'})
    clean_df['l1'] -= 1
    clean_df['l1'] = clean_df['l1'].astype(int)
    # 2. create json file map for problem_id to problem_body
    tr = clean_df.groupby(['id', 'p'])['id'].count()
    tr = tr.to_dict()
    task_map = {str(k[0]): k[1] for k in list(tr.keys())}
    task_map = filter_question(clean_df, sample_num=25)
    # 3 create task list, each problem is a task
    task_list = list(task_map.keys())
    task_list = [str(l) for l in task_list]
    task_list_int = [int(l) for l in task_list]

    clean_df = clean_df.loc[clean_df['id'].isin(task_list_int)]
    # 4. group example list by problem, each problem consider as the task
    tr = list(clean_df.groupby('id'))
    meta_data = {str(l[0]): l[1].to_dict('records') for l in tr}

    with open('data/tasks_math{}.json'.format(alias), 'w') as outfile:
        outfile.write(json.dumps(task_list, sort_keys=True, indent=4))
    with open('data/tasks_to_question_math{}.json'.format(alias), 'w') as outfile:
        outfile.write(json.dumps(task_map, sort_keys=True, indent=4))
    with open('data/meta{}.json'.format(alias), 'w') as outfile:
        outfile.write(json.dumps(meta_data, sort_keys=True, indent=4))

def filter_question(df, sample_num = 25):
    tr = df.groupby(['id', 'p'])['id'].count()
    tr = tr.to_dict()

    #1. if number example < 25, delete
    task_map = {str(k[0]): k[1] for k in list(tr.keys()) if tr[k] >= sample_num}
    #2. filter out question that didn't have description
    def filter_description(task, key=None):
        #descript page information, no detail description
        task = clean(task)
        if len(str(task)) < 15:
            return False
        task = task.lower()
        if task == np.nan:
            return False
        if 'xiao wang' in str(task):
            return False
        if 'translate' in str(task) and 'chinese' in str(task):
            return False
        #description is not long enough

        try:
            lang = detect(task)
            langs = detect_langs(task)
        except:
            return False
        if lang != 'en':
            if langs[0].lang == 'en' and langs[0].prob > 0.5:
                return True
            return False
        #description contain broken words
        return True
    def clean(task):
        task = task.replace(u'\r', u' ')
        task = task.replace(u'\n', u' ')
        task = task.replace(u'\\', u' ')
        task = task.replace(u'\xa0', u' ')
        task = task.replace('Modified from EngageNY ©Great Minds Disclaimer',' ')
        broken_tokens = 'âð¥¸¶´ð·ð¤§¦'
        for i in broken_tokens:
            task = task.replace(i, ' ')

        task = task.split(' ')
        task = [t for t in task if len(t) > 0]
        if len(task) == 0:
            return np.nan
        task = ' '.join(task)
        return task
    task_map = {str(k): clean(task_map[k]) for k in list(task_map.keys()) if filter_description(task_map[k], k) }
    #3. filter out problem that all grade are same ( meaningless data )
    filter_task_map = {}
    for i in task_map.keys():
        i = int(i)
        task_pd = df.loc[df['id'] == i]
        count = task_pd.groupby(['l1']).count()
        if len(count) != 1:
            filter_task_map.update({str(i):task_map[str(i)]})
    return filter_task_map



def load_data(path='data/'):

    with open( path + 'tasks_math.json') as outfile:
        task_list = json.load(outfile)
    with open(path + 'tasks_to_question_math.json') as outfile:
        task_map = json.loads(outfile.read())
    with open(path + 'small_meta.json') as outfile:
        meta_data = json.loads(outfile.read())
    return task_list, task_map, meta_data

def check_load_dataset():
    task_list, task_map, meta_data = load_data()
    load_dataset_in_context_tuning_with_meta_learning(task_list=task_list)

def prepare_dataset_in_context_tuning(train_dataset, raw, n_example = -1):
    examples = defaultdict(list)
    max_label = 0
    min_label = 100
    if n_example != -1:
        train_dataset = train_dataset[0:n_example]

    for datapoint in train_dataset:
        label = datapoint['l1']
        examples[label].append(datapoint)
    for datapoint in raw:
        label = datapoint['l1']
        if max_label < label:
            max_label = label
        if min_label > label:
            min_label = label


    return examples, min_label, max_label

def split_raw_data(raw, task_list, seed = 20, fold = None):
    if fold is not None:
        train = {}
        val = {}
        test = {}
        no_train = 0
        no_val = 0

        test_num = 0
        test_all = 0

        train_num = 0
        train_all = 0
        for task in task_list:
            temp = raw[task]
            train_temp = [t for t in temp if t['folds'] != fold]
            val_temp = [t for t in temp if t['folds'] == fold]
            test_temp = val_temp
            assert len(train_temp) + len(val_temp) == len(temp)
            if len(train_temp) == 0:
                no_train += 1
                test_num += len(val_temp)
                val[task] = val_temp
                train[task] = val_temp
                test[task] = test_temp
                assert len(val_temp) != 0
            elif len(val_temp) == 0:
                no_val += 1
                train_num += len(train_temp)
            else:
                train[task] = train_temp
                val[task] = val_temp
                test[task] = test_temp
            test_all += len(val_temp)
            train_all += len(train_temp)
        print('There are {} task dont have training data and {} task doesnt have testing data'.format(no_train,no_val))
        print('{} over {}  testing data didnt train: {}'.format(test_num, test_all,test_num / test_all))
        print('{} over {}  training data didnt have test data: {}'.format(train_num, train_all, train_num / train_all))
        print('There are {} percentage of the data use for training'.format(train_all/(train_all+test_all)))
        return train,val,test

    split = [0.8, 1.0, 1.0]
    train = {}
    val = {}
    test = {}
    for task in task_list:
        temp = raw[task]
        if seed != -1:
            random.seed(seed)
        random.shuffle(temp)
        length = len(temp)
        train_end = int(split[0] * length)
        train_temp = temp[0:train_end]
        val_temp = temp[train_end:]
        test_temp = val_temp

        train[task] = train_temp
        val[task] = val_temp
        test[task] = test_temp
    return train, val, test

def split_raw_data_meta(raw, task_list, seed=20, fold=None):
    if fold is not None:
        train = {}
        val = {}
        test = {}
        no_train = 0
        no_val = 0

        test_num = 0
        test_all = 0

        train_num = 0
        train_all = 0
        train_task_list = []
        val_task_list = []
        for task in task_list:
            temp = raw[task]
            train_temp = [t for t in temp if t['folds'] != fold]
            val_temp = [t for t in temp if t['folds'] == fold]
            test_temp = val_temp
            assert len(train_temp) + len(val_temp) == len(temp)
            if len(train_temp) == 0:
                no_train += 1
                test_num += len(val_temp)
                val[task] = val_temp
                test[task] = test_temp
                assert len(val_temp) != 0
                val_task_list.append(task)
            else:
                assert len(val_temp) == 0
                no_val += 1
                train_num += len(train_temp)
                train[task] = train_temp
                train_task_list.append(task)

            test_all += len(val_temp)
            train_all += len(train_temp)

        assert len(train.keys()) == len(train_task_list), 'this is not question split'
        assert len(val.keys()) == len(val_task_list), 'this is not question split'
        assert len(train.keys()) + len(val.keys()) == len(task_list), 'this is not question split'
        print('There are {} task dont have training data and {} task doesnt have testing data'.format(no_train,no_val))
        print('{} over {}  testing data didnt train: {}'.format(test_num, test_all,test_num / test_all))
        print('{} over {}  training data didnt have test data: {}'.format(train_num, train_all, train_num / train_all))



        return train, val, test, train_task_list, val_task_list

    split = [0.8, 1.0, 1.0]
    train = {}
    val = {}
    test = {}
    for task in task_list:
        temp = raw[task]
        if seed != -1:
            random.seed(seed)
        random.shuffle(temp)
        length = len(temp)
        train_end = int(split[0] * length)
        train_temp = temp[0:train_end]
        val_temp = temp[train_end:]
        test_temp = val_temp

        train[task] = train_temp
        val[task] = val_temp
        test[task] = test_temp
    return train, val, test

def split_raw_data_finetune(raw, task_list, seed=20, fold=None, n ='train5', n_val = 'train80'):
    train, val = {}, {}
    no_train, no_val, test_num, test_all, train_num, train_all = 0, 0, 0,0,0,0
    for task in task_list:
        temp = raw[task]
        train_temp = [t for t in temp if t[n] == 1]
        val_temp = [t for t in temp if t[n_val] == 0]
        #assert len(train_temp) + len(val_temp) == len(temp)
        if len(train_temp) == 0:
            no_train += 1
            test_num += len(val_temp)
            val[task] = val_temp
            train[task] = val_temp
            assert len(val_temp) != 0
        elif len(val_temp) == 0:
            no_val += 1
            train_num += len(train_temp)
        else:
            train[task] = train_temp
            val[task] = val_temp
        test_all += len(val_temp)
        train_all += len(train_temp)
    print('There are {} task dont have training data and {} task doesnt have testing data'.format(no_train, no_val))
    print('{} over {}  testing data didnt train: {}'.format(test_num, test_all, test_num / test_all))
    print('{} over {}  training data didnt have test data: {}'.format(train_num, train_all, train_num / train_all))
    print('There are {} percentage of the data use for training'.format((train_all-train_num) / (train_all + test_all)))
    return train, val, val


def load_dataset_in_context_tuning_with_meta_learning(debug=False, task_list=[], data_path='small_meta.json',
                                                      submit_mode=False, meta_learning_single=False, alias='', fold=None, **kwargs):
    if (submit_mode or meta_learning_single):
        task_list = task_list
        min_label = -1
        max_label = -1
    else:
        with open("data/tasks_math{}.json".format(alias), "r") as f:
            task_list = json.load(f)

    with open("data/" + data_path, "r") as f:
        raw = dict(json.load(f))
        train, valid, test = split_raw_data(raw, task_list, fold=fold)

    data_meta = {}
    task_list = train.keys()

    for task in task_list:
        data_meta[task] = {}
        examples_train, min_label, max_label = prepare_dataset_in_context_tuning(train[task], raw[task])
        data_meta[task]["train"] = train[task]
        data_meta[task]["valid"] = valid[task]
        data_meta[task]["test"] = test[task]
        # add in-context examples from trainset
        data_meta[task]["examples"] = {}
        for label in range(int(min_label), int(max_label) + 1):
            data_meta[task]["examples"][label] = examples_train[label]
        # add task, min_label and max_label info to each sample
        for set in ["train", "valid", "test"]:
            for sample in data_meta[task][set]:
                sample["min"] = min_label
                sample["max"] = max_label
                sample["task"] = task

    # union of trainsets across tasks
    data_meta["train"] = []
    for task in task_list:
        data_meta["train"] += data_meta[task]["train"]

    if (meta_learning_single):
        return data_meta, min_label, max_label
    else:
        return data_meta, None, None


def load_dataset_in_context_tuning_with_meta_learning_finetune(debug=False, task_list=[], data_path='small_meta.json',
                                                      submit_mode=False, meta_learning_single=False, alias='', fold=None, n_example = 5, n_val='train80',**kwargs):
    assert 'csv' in data_path, 'data path is not csv file for finetune mode'
    df = pd.read_csv(data_path, index_col=0, encoding='utf-8')
    df = df[df['folds'] == fold]

    n = 'train' + str(n_example)
    if n_example not in [5,10,25,50,80]:
        if n_example < 5:
            n = 'train5'
        elif n_example < 10:
            n = 'train10'
        elif n_example < 25:
            n = 'train25'
        else:
            n= 'train80'
    df = df.rename(
        columns={'problem_log_id': 'bl', "problem_id": "id", "clean_full_problem": "p", 'cleaned_answer_text': 'txt',
                 'grade': 'l1'})
    df = df[['id', 'bl', 'p', 'txt', 'l1', 'folds',n, n_val]]
    df['l1'] -= 1
    tr = list(df.groupby('id'))
    raw = {str(l[0]): l[1].to_dict('records') for l in tr}
    task_list = raw.keys()
    train, valid, test = split_raw_data_finetune(raw, task_list, fold = fold,  n = n, n_val = n)

    data_meta = {}
    task_list = train.keys()
    for task in task_list:
        data_meta[task] = {}

        if n_example < 10 and n_example != 5:
            examples_train, min_label, max_label = prepare_dataset_in_context_tuning(train[task], raw[task], n_example = n_example)
        else:
            examples_train, min_label, max_label = prepare_dataset_in_context_tuning(train[task], raw[task])
        data_meta[task]["train"] = train[task]
        data_meta[task]["valid"] = valid[task]
        data_meta[task]["test"] = test[task]
        # add in-context examples from trainset
        data_meta[task]["examples"] = {}
        for label in range(int(min_label), int(max_label) + 1):
            data_meta[task]["examples"][label] = examples_train[label]
        # add task, min_label and max_label info to each sample
        for set in ["train", "valid", "test"]:
            for sample in data_meta[task][set]:
                sample["min"] = min_label
                sample["max"] = max_label
                sample["task"] = task
    # union of trainsets across tasks
    data_meta["train"] = []
    for task in task_list:
        data_meta["train"] += data_meta[task]["train"]

    if (meta_learning_single):
        return data_meta, min_label, max_label
    else:
        return data_meta, None, None
def load_dataset_in_context_tuning_with_meta_learning_question_split(debug=False, task_list=[], data_path='small_meta.json',
                                                      submit_mode=False, meta_learning_single=False, alias='', fold=None, **kwargs):
    if (submit_mode or meta_learning_single):
        task_list = task_list
        min_label = -1
        max_label = -1
    else:
        with open("data/tasks_math{}.json".format(alias), "r") as f:
            task_list = json.load(f)


    with open("data/" + data_path, "r") as f:
        raw = dict(json.load(f))
        train, valid, test, train_list, val_list = split_raw_data_meta(raw, task_list, fold=fold)

    data_meta = {}
    task_list = train.keys()
    for task in task_list:
        data_meta[task] = {}
        examples_train, min_label, max_label = prepare_dataset_in_context_tuning(train[task], raw[task])
        data_meta[task]["train"] = train[task]
        #data_meta[task]["valid"] = valid[task]
        #data_meta[task]["test"] = test[task]
        # add in-context examples from trainset
        data_meta[task]["examples"] = {}
        for label in range(int(min_label), int(max_label) + 1):
            data_meta[task]["examples"][label] = examples_train[label]
        # add task, min_label and max_label info to each sample
        for set in ["train"]: #, "valid", "test"]:
            for sample in data_meta[task][set]:
                sample["min"] = min_label
                sample["max"] = max_label
                sample["task"] = task

    val_list = valid.keys()
    for task in val_list:
        data_meta[task] = {}
        examples_train, min_label, max_label = prepare_dataset_in_context_tuning(valid[task], raw[task])
        data_meta[task]["valid"] = valid[task]
        data_meta[task]["test"] = test[task]
        # add in-context examples from trainset
        data_meta[task]["examples"] = {}
        for label in range(int(min_label), int(max_label) + 1):
            data_meta[task]["examples"][label] = []
        # add task, min_label and max_label info to each sample
        for set in ["valid",'test']:  # , "valid", "test"]:
            for sample in data_meta[task][set]:
                sample["min"] = min_label
                sample["max"] = max_label
                sample["task"] = task

    # union of trainsets across tasks
    data_meta["train"] = []
    for task in task_list:
        data_meta["train"] += data_meta[task]["train"]

    if (meta_learning_single):
        return data_meta, min_label, max_label
    else:
        return data_meta, None, None

def assign_label_to_csv(path, path_all, label_list, fold):
    try:
        df = pd.read_csv(path, index_col=0, encoding='utf-8')
        print('open file {} for fold {}'.format(path, fold))
    except:
        df = pd.read_csv(path_all, index_col=0, encoding='utf-8')
        if 'predict' not in df.head(0):
            df['predict'] = 0
        df['predict'] = 0
        print('open file {} for fold {}'.format(path_all, fold))
    same_question_id_nums = 0
    for id, pre in label_list:
        index_df = df.loc[df['problem_log_id'] == id]
        index = index_df.index.to_list()
        try:
            assert len(index) == 1, 'id {} not one with {} values'.format(id, len(index))
        except:
            same_question_id_nums += len(index) - 1

        final_index = 0
        for i in index:
            if df.loc[i, 'folds'] == fold:
                final_index = i
                df.loc[i, 'predict'] = pre
        assert df.loc[final_index, 'folds'] == fold, 'the fold is {}, but in folder is {}, file path is {}'.format(fold, df.loc[final_index, 'folds'], path_all)

    df = df[df['folds'] == fold]
    if 'finetune' in path:
        df = df[['problem_log_id', 'problem_id', 'clean_full_problem', 'grade', 'predict', 'folds','train5','train10','train25','train50','train80']]
    else:
        df = df[['problem_log_id', 'problem_id', 'clean_full_problem', 'grade','predict','folds']]
    df.to_csv(path, index=True)
    print('save done, there are {} examples have same id'.format(same_question_id_nums))


def for_checking(path, fold = 1, sbert=False):
    data_path = 'meta_full.json'
    with open("data/" + data_path, "r") as f:
        raw = dict(json.load(f))
    list_tuple = []
    for id in raw:
        for r in raw[id]:
            if r['folds'] == fold:
                temp = (r['bl'], -2)
                list_tuple.append(temp)
    assign_label_to_csv(path, label_list=list_tuple, fold=fold)

def quick_check_metric(path, fold=1, sbert=False):
    from utils import auc, convert_to_one_hot, rmse, cohen_kappa_multiclass, acuracy
    import numpy as np
    df = pd.read_csv(path, index_col=0, encoding='utf-8')
    df = df[df['folds'] == fold]
    if sbert:
        df = df.rename(columns={'sbert_predictions':'predict'})
        df['predict'] += 1
    else:
        df = df[df['clean_full_problem'].notnull()]
    df = df[df['cleaned_answer_text'].notnull()]
    #df['predict'] = df.apply(lambda row: row['predict'] - 1, axis=1)
    df['teacher_score'] = df.apply(lambda row: row['grade'] - 1, axis=1)
    actual = convert_to_one_hot(df['teacher_score'])
    prediction = convert_to_one_hot(df['predict'])
    aucm, rmsem, kappam = auc(actual, prediction), rmse(actual, prediction), cohen_kappa_multiclass(actual, prediction)
    acum = acuracy(df['teacher_score'], df['predict'])
    print('the auc score is: {} and rmse is {} and kappa is {} and accuracy is {}'.format(aucm, rmsem, kappam, acum))

def main():
    path = '../data/math/qc_data_small.csv'
    path2 = '../data/math/qc_full.csv'
    path1 = '../data/math/rasch.csv'
    path2 = '../data/math/qc_clean.csv'
    prepare_data(path2, alias='_clean')
    #for_checking(path)
    #quick_check_metric(path1, sbert=True)
    #quick_check_metric(path2)
    #load_data()
    #load_dataset_in_context_tuning_with_meta_learning()



if __name__ == '__main__':
    main()