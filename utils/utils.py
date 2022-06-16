import numpy as np
import json
import os
from sklearn.metrics import cohen_kappa_score
import numpy as np
from skll.metrics import kappa as kpa


def tonp(x):
    if isinstance(x, (np.ndarray, float, int, list)):
        return np.array(x)
    return x.detach().cpu().numpy()
# TODO P2: cross check this
def agg_all_metrics(outputs, submit_mode=False, train=False):
    if len(outputs) == 0:
        return outputs
    res = {}
    keys = [k for k in outputs[0].keys() if not isinstance(outputs[0][k], dict)]
    bl_ids = []
    for x in outputs:
        bl_ids += x['bl']

    for k in keys:
        if k == 'bl':
            continue
        all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])

        if k != 'epoch':
            res[k] = float(np.mean(all_logs))
        else:
            res[k] = all_logs[-1]

        if k == 'bl':
            res[k] = all_logs
    if 'loss' in res:
        res['loss'] = float(res['loss'])
    if train:
        return res



    if 'kappa' in outputs[0]:
        pred_logs =  np.concatenate([tonp(x['kappa']['preds']).reshape(-1) for x in outputs])
        label_logs = np.concatenate([tonp(x['kappa']['labels']).reshape(-1) for x in outputs])
        pred_logs_adjust = np.concatenate([(tonp(x['kappa']['preds']) + tonp(x['min'])).reshape(-1) for x in outputs])
        label_logs_adjust = np.concatenate([(tonp(x['kappa']['labels']) + tonp(x['min'])).reshape(-1) for x in outputs])
        bl_pred_pair = zip(bl_ids, pred_logs_adjust)
        res['bl_pair'] = list(bl_pred_pair)

        #if( np.array_equal(pred_logs, label_logs) ):
            # cohen_kappa_score returns a value of NaN when perfect agreement
        #    res['kappa'] = 1
        #else:
            #res['kappa'] = cohen_kappa_score(pred_logs, label_logs, weights= 'quadratic')

        one_hot_true = convert_to_one_hot(label_logs_adjust)
        one_hot_pred = convert_to_one_hot(pred_logs_adjust)

        res['auc'] = float(auc(one_hot_true, one_hot_pred))
        res['mse'] = float(rmse(one_hot_true, one_hot_pred))
        res['kappa'] = float(cohen_kappa_multiclass(one_hot_true, one_hot_pred))


    if( submit_mode ):
        res["pred_logs"] =  pred_logs
        res["label_logs"] =  label_logs
    return res
def build_csv(outputs):
    pass
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
def convert_to_one_hot(vector, cat=4):
    a = np.array(vector)
    a = a.astype(int)
    b = np.zeros((a.size, cat + 1))
    b[np.arange(a.size), a] = 1
    return b
def one_hot_encoding(grades, max=4):
    a_gr = np.array(grades)
    b_gr = np.zeros((a_gr.size, max + 1))
    b_gr[np.arange(a_gr.size), a_gr] = 1
    return b_gr
# auc for multiclass classification
# actual and predicted are one hot encoded values of the labels
def auc(actual, predicted, average_over_labels=True, partition=1024.):
    assert len(actual) == len(predicted), print('actual len is {}, predicted len is {}'.format(len(actual), len(predicted)))

    ac = np.array(actual, dtype=np.float32).reshape((len(actual),-1))
    pr = np.array(predicted, dtype=np.float32).reshape((len(predicted),-1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    ac = ac[na]
    pr = pr[na]

    label_auc = []
    for i in range(ac.shape[-1]):
        a = np.array(ac[:,i])
        p = np.array(pr[:,i])

        val = np.unique(a)

        # if len(val) > 2:
        #     print('AUC Warning - Number of distinct values in label set {} is greater than 2, '
        #           'using median split of distinct values...'.format(i))
        if len(val) == 1:
            # print('AUC Warning - There is only 1 distinct value in label set {}, unable to calculate AUC'.format(i))
            label_auc.append(np.nan)
            continue

        pos = np.argwhere(a[:] >= np.median(val))
        neg = np.argwhere(a[:] < np.median(val))

        # print(pos)
        # print(neg)

        p_div = int(np.ceil(len(pos)/partition))
        n_div = int(np.ceil(len(neg)/partition))

        # print(len(pos), p_div)
        # print(len(neg), n_div)

        div = 0
        for j in range(int(p_div)):
            p_range = list(range(int(j * partition), int(np.minimum(int((j + 1) * partition), len(pos)))))
            for k in range(n_div):
                n_range = list(range(int(k * partition), int(np.minimum(int((k + 1) * partition), len(neg)))))


                eq = np.ones((np.alen(neg[n_range]), np.alen(pos[p_range]))) * p[pos[p_range]].T == np.ones(
                    (np.alen(neg[n_range]), np.alen(pos[p_range]))) * p[neg[n_range]]

                geq = np.array(np.ones((np.alen(neg[n_range]), np.alen(pos[p_range]))) *
                               p[pos[p_range]].T >= np.ones((np.alen(neg[n_range]),
                                                             np.alen(pos[p_range]))) * p[neg[n_range]],
                               dtype=np.float32)
                geq[eq[:, :] == True] = 0.5

                # print(geq)
                div += np.sum(geq)
                # print(np.sum(geq))
                # exit(1)

        label_auc.append(div / (np.alen(pos)*np.alen(neg)))
        # print(label_auc)

    if average_over_labels:
        return np.nanmean(label_auc)
    else:
        return label_auc
def cohen_kappa_multiclass(actual, predicted):
    assert len(actual) == len(predicted)

    ac = np.array(actual,dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted,dtype=np.float32).reshape((len(predicted), -1))

    try:
        na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()
    except:
        for i in ac:
            print(i)

        for i in ac:
            print(np.any(np.isnan(i)))

    if len(na) == 0:
        return np.nan

    aci = np.argmax(np.array(np.array(ac[na]), dtype=np.int32), axis=1)
    pri = np.argmax(np.array(np.array(pr[na]), dtype=np.float32), axis=1)

    # for i in range(len(aci)):
    #     print(aci[i],'--',pri[i],':',np.array(pr[na])[i])

    return kpa(aci,pri)
def acuracy(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    acuracy = sum(actual == predicted)/len(actual)
    return acuracy
def rmse(actual, predicted, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual, dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted, dtype=np.float32).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    ac = ac[na]
    pr = pr[na]

    label_rmse = []
    for i in range(ac.shape[-1]):
        dif = np.array(ac[:, i]) - np.array(pr[:, i])
        sqdif = dif**2
        mse = np.nanmean(sqdif)
        label_rmse.append(np.sqrt(mse))


    if average_over_labels:
        return np.nanmean(label_rmse)
    else:
        return label_rmse


def fill_score(df, row='meta',auc=0, rmse=1,kappa=0, fold=None,num=0, acu = 0):
    print('save result into result.csv', row, ' fold:', fold)
    auc, rmse, kappa = float(auc),float(rmse),float(kappa)
    row_name = row+str(num)
    df_one = df[(df['fold'] == fold) & (df['name'] == row_name)]
    assert len(df_one) == 1, str(df_one)
    index = list(df_one.index)[0]
    if auc + kappa >= df.loc[index,'auc'] + df.loc[index,'kappa'] or df.loc[index,'rmse'] > 0.9:
        print('set value at index')
        df.loc[index,'auc'] = auc
        df.loc[index,'kappa'] = kappa
        df.loc[index,'rmse'] = rmse
        df.loc[index,'rmse'] = rmse
        df.loc[index,'acu'] = acu
    df = df[['name','auc','rmse','kappa','acu','fold']]
    return df
