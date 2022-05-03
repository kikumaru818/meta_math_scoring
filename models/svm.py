import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics


def load_data():
    path = "NAEP_AS_Challenge_Data/Items for Item-Specific Models/Grade 4/"
    filenames = ["2017_DBA_DR04_1715RE1T10_05", "2017_DBA_DR04_1715RE4T05G04_07"]

    train_df = pd.read_csv(path + filenames[0] + "/" + filenames[0] + "_Training.csv")
    val_df = pd.read_csv(path + filenames[0] + "/" + filenames[0] + "_Validation_DS&SS.csv")

    train_X = train_df["WordCount"].to_numpy()
    train_y = train_df["Score1"].to_numpy()
    val_X = val_df["WordCount"].to_numpy()
    val_y = val_df["Score1"].to_numpy()
    
    train_X = np.reshape(train_X, (-1, 1))
    val_X = np.reshape(val_X, (-1, 1))

    #train_y.astype(int)
    #val_y.astype(int)

    print(np.count_nonzero(val_y=='1') / float(val_y.shape[0]))
    print(np.count_nonzero(val_y=='2') / float(val_y.shape[0]))
    print(np.count_nonzero(val_y==' ') / float(val_y.shape[0]))

    items = [train_X, train_y, val_X, val_y]
    for item in items:
        print(item.shape)
    
    return train_X, train_y, val_X, val_y


def svm_classify(train_X, train_y, val_X):
    clf = svm.LinearSVC(C=1, max_iter=100000)
    #clf = linear_model.LinearRegression()
    clf.fit(train_X, train_y)
    pred_y = clf.predict(val_X)
    
    print(np.count_nonzero(pred_y=='1') / float(pred_y.shape[0]))

    return pred_y


def cal_metrics(val_y, pred_y):
    # compute cohen's kappa and accuracy
    acc = metrics.accuracy_score(val_y, pred_y)
    kappa = metrics.cohen_kappa_score(val_y, pred_y)# weights="quadratic")

    return acc, kappa


def main():
    train_X, train_y, val_X, val_y = load_data()
    pred_y = svm_classify(train_X, train_y, val_X)
    acc, kappa = cal_metrics(val_y, pred_y)

    print("accuracy: ", acc)
    print("quadratic weighted kappa: ", kappa)

    
if __name__ == '__main__':
    main()