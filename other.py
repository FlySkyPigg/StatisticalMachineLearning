"""
we can find that a linear regression gives us a 85% mean accuracy.
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
import csv
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from glob import glob
from sklearn.ensemble import AdaBoostClassifier
import torch
import random


def get_test_data():
    paths = list(glob('data/test/*.npy'))
    paths.sort()
    datas = []
    for path in paths:
        datas.append(np.load(path))
    datas = np.stack(datas, axis=0)
    paths = [path.replace('data/test/', '') for path in paths]

    datas = np.reshape(datas, newshape=(datas.shape[0], -1))
    return datas, paths


def get_data():
    paths = []
    labels = []
    with open('data/label_train.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            paths.append(row[0])
            labels.append(int(row[1]))

    data_list = []
    label_list = []
    for path, label in zip(paths, labels):
        data = np.load(f'data/train/{path}')
        data_list.append(data)
        label_list.append(label)

    datas = np.stack(data_list, axis=0)
    labels = np.stack(label_list, axis=0)

    inds = np.arange(datas.shape[0])
    np.random.shuffle(inds)
    datas = datas[inds]
    labels = labels[inds]

    return datas, labels


if __name__ == "__main__":
    datas, labels = get_data()
    datas = np.reshape(datas, newshape=(datas.shape[0], -1))

    train_inds = np.random.choice(np.arange(datas.shape[0]), size=(int(0.8 * datas.shape[0], )), replace=False)
    x_train = datas[train_inds]
    y_train = labels[train_inds]

    val_inds = [i for i in range(datas.shape[0]) if i not in train_inds]
    x_val = datas[val_inds]
    y_val = labels[val_inds]

    # ---- submit logistic.

    # for i in range(2, 64):
    #     pca = PCA(n_components=i + 1)
    #     pca.fit(datas)
    #     datas_reduced = pca.transform(datas)
    #
    #     clf = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    #
    #     scores = cross_val_score(clf, datas_reduced, labels, scoring='accuracy', cv=10)
    #     scores = sum(scores) / len(scores)
    #     print(i, scores)

    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import SGDClassifier

    from sklearn import tree
    #
    # for i in range(2, 64):
    #     i = 16
    #     pca = PCA(n_components=i + 1)
    #     pca.fit(datas)
    #     datas_reduced = pca.transform(datas)
    #     rbf_feature = RBFSampler(gamma=1, random_state=1)
    #     X_features = rbf_feature.fit_transform(datas_reduced)
    #     clf = RandomForestClassifier(n_estimators=100)
    #     scores = cross_val_score(clf, datas_reduced, labels, scoring='accuracy', cv=10)
    #     scores = sum(scores) / len(scores)
    #     print(i, scores)
    #     break
    # exit()

    pca = PCA(n_components=16)
    pca.fit(datas)
    datas_reduced = pca.transform(datas)

    clf = RandomForestClassifier(n_estimators=100)
    # maybe i can normalize the data.
    clf.fit(datas_reduced, labels)

    test_datas, test_paths = get_test_data()
    test_datas_reduced = pca.transform(test_datas)
    test_pred_labels = clf.predict(test_datas_reduced)

    with open('data/pca_tree_label_test.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'category'])
        for path, pred_label in zip(test_paths, test_pred_labels):
            writer.writerow([path, pred_label])
