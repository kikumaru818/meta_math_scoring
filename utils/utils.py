import os
import json
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    return x.detach().cpu().numpy()



def agg_all_metrics(outputs):
    if len(outputs) == 0:
        return outputs
    res = {}
    keys = [k for k in outputs[0].keys() if not isinstance(outputs[0][k], dict)]
    for k in keys:
        all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
        if k != 'epoch':
            res[k] = np.mean(all_logs)
        else:
            res[k] = all_logs[-1]
    kappa_keys = [k for k in outputs[0].keys() if 'kappa' in k]
    for kappa_key  in kappa_keys:
        pred_logs =  np.concatenate([tonp(x[kappa_key]['preds']).reshape(-1) for x in outputs])
        label_logs = np.concatenate([tonp(x[kappa_key]['labels']).reshape(-1) for x in outputs])
        res[kappa_key] = cohen_kappa_score(pred_logs, label_logs,weights= 'quadratic')
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