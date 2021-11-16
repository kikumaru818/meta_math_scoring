import os
import json
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

def human_kappa(data):
    labels = [[d['l1'], d['l2']] for d in data if d['l1']>=0 and d['l2']>=0 ]
    pred_logs = [d[0] for d in labels]
    label_logs = [d[1] for d in labels]
    return cohen_kappa_score(pred_logs, label_logs,weights= 'quadratic')

def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    return x.detach().cpu().numpy()

def flat_predictions(outputs):
    try:
        pred_logs =  np.concatenate([tonp(x['kappa']['preds']).reshape(-1) for x in outputs])
        label_logs = np.concatenate([tonp(x['kappa']['labels']).reshape(-1) for x in outputs])
        return {'preds':pred_logs.tolist(), 'labels':label_logs.tolist()}
    except KeyError:
        return {}

def agg_all_metrics(outputs):
    if len(outputs) == 0:
        return outputs
    res = {}
    keys = [k for k in outputs[0].keys() if not isinstance(outputs[0][k], dict)]
    for k in keys:
        if k in {'epoch', 'loss'}:
            all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
        else:
            try:
                all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs if len(x[k])>0])
            except ValueError:
                all_logs = np.zeros(2)-1.
        if k=='epoch':
            res[k] = all_logs[-1]
        res[k] = np.mean(all_logs)
            
    kappa_keys = [k for k in outputs[0].keys() if 'kappa' in k]
    for kappa_key  in kappa_keys:
        try:
            pred_logs =  np.concatenate([tonp(x[kappa_key]['preds']).reshape(-1) for x in outputs  if len(x[kappa_key])>0])

            label_logs = np.concatenate([tonp(x[kappa_key]['labels']).reshape(-1) for x in outputs if len(x[kappa_key])>0])
            res[kappa_key] = cohen_kappa_score(pred_logs, label_logs,weights= 'quadratic')
        except ValueError:
            res[kappa_key] =  -2.
    return res
    
def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data

def safe_makedirs(path_):
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass