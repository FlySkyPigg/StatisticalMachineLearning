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

    # ---------- do a simple linear regression.
    # clf = linear_model.LogisticRegression(C=50.0 / datas.shape[0], penalty="l1", solver="saga", tol=0.01)
    # clf.fit(x_train, y_train)
    #
    # score = clf.score(x_train, y_train)
    # print(score)
    # score = clf.score(x_val, y_val)
    # print(score)

    # reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    # reg.fit(x_train, y_train)
    # score = reg.score(x_val, y_val)
    # print(score)

    # we can see that a simple LDA doesn't actually work.
    # maybe because the feature dimension is too fucking high.
    # clf = LinearDiscriminantAnalysis()
    # clf.fit(x_train, y_train)
    # score = clf.score(x_val, y_val)
    # print(score)

    # clf = QuadraticDiscriminantAnalysis()
    # clf.fit(x_train, y_train)
    # score = clf.score(x_val, y_val)
    # print(score)

    # clf = svm.SVC()
    # clf.fit(x_train, y_train)
    # score = clf.score(x_val, y_val)

    # split training data into 10 splits.

    inds = torch.arange(x_train.shape[0])
    chunks = torch.chunk(inds, dim=0, chunks=10)
    chunks = [chunk.numpy() for chunk in chunks]
    train_chunks = []

    for tcid in range(10):
        train_chunk = []
        for cid in range(10):
            if cid == tcid:
                continue
            train_chunk.append(chunks[cid])
        train_chunk = np.concatenate(train_chunk, axis=0)
        train_chunks.append(train_chunk)

    pred_labels_list = []
    for i in range(10):
        # clf = svm.SVC()
        clf = linear_model.LogisticRegression(random_state=1)
        print('haha')
        clf.fit(x_train[train_chunks[i]], y_train[train_chunks[i]])
        pred_labels = clf.predict(x_val)
        pred_labels_list.append(pred_labels)
    pred_labels_list = np.stack(pred_labels_list, axis=0)
    pred_labels_list = [np.bincount(pred_labels_list[:, i]).argmax() for i in range(pred_labels_list.shape[1])]
    pred_labels_list = np.stack(pred_labels_list)
    precision = (pred_labels_list == y_val).mean()
    print(precision)

    # # we can see that the result is amazing.
    #
    # # ------------ submit svm ensemble -----------------
    # inds = torch.arange(datas.shape[0])
    # chunks = torch.chunk(inds, dim=0, chunks=10)
    # chunks = [chunk.numpy() for chunk in chunks]
    # train_chunks = []
    #
    # for tcid in range(10):
    #     train_chunk = []
    #     for cid in range(10):
    #         if cid == tcid:
    #             continue
    #         train_chunk.append(chunks[cid])
    #     train_chunk = np.concatenate(train_chunk, axis=0)
    #     train_chunks.append(train_chunk)
    #
    # test_datas, test_paths = get_test_data()
    #
    # pred_labels_list = []
    # for i in range(10):
    #     clf = svm.SVC()
    #     clf.fit(datas[train_chunks[i]], labels[train_chunks[i]])
    #     pred_labels = clf.predict(test_datas)
    #     pred_labels_list.append(pred_labels)
    # pred_labels_list = np.stack(pred_labels_list, axis=0)
    # pred_labels_list = [np.bincount(pred_labels_list[:, i]).argmax() for i in range(pred_labels_list.shape[1])]
    # test_pred_labels = np.stack(pred_labels_list)
    #
    # with open('data/ensemble_svm_label_test.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['id', 'category'])
    #     for path, pred_label in zip(test_paths, test_pred_labels):
    #         writer.writerow([path, pred_label])
    #
    # exit()
    # # score = clf.score(x_val, y_val)

    # # ---- submit svm.
    # clf = svm.SVC(C=1)
    # scores = cross_val_score(clf, datas, labels, scoring='accuracy', cv=5)
    # print(scores)

    # test_datas, test_paths = get_test_data()
    # test_pred_labels = clf.predict(test_datas)
    #
    # with open('data/svm_label_test.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['id', 'category'])
    #     for path, pred_label in zip(test_paths, test_pred_labels):
    #         writer.writerow([path, pred_label])

    # clf = Pipeline([
    #     ('feature_selection', SelectFromModel(svm.LinearSVC(penalty="l1", dual=False))),
    #     ('classification', svm.SVC())
    # ])
    # clf.fit(x_train, y_train)
    # score = clf.score(x_val, y_val)
    # print(score)

    # clf1 = linear_model.LogisticRegression(random_state=1)
    # clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    # clf3 = svm.SVC()
    # eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    # for clf, label in zip([clf1, clf2, clf3, eclf],
    #                       ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    #     scores = cross_val_score(clf, datas, labels, scoring='accuracy', cv=5)
    #     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    # clf = KNeighborsClassifier(n_neighbors=7)
    # scores = cross_val_score(clf, datas, labels, scoring='accuracy', cv=5)
    # print(scores)

    # for i in range(1, 64):
    #     pca = PCA(n_components=i + 1)
    #     pca.fit(datas)
    #     datas_reduced = pca.transform(datas)
    #
    #     # clf = RandomForestClassifier(n_estimators=50, random_state=1)
    #     # clf = KNeighborsClassifier(n_neighbors=7)
    #     # clf = linear_model.LogisticRegression(random_state=1, C=50.0 / datas.shape[0], solver='saga', tol=0.01)
    #     clf = svm.SVC()
    #
    #     # clf1 = RandomForestClassifier(n_estimators=50, random_state=1)
    #     # clf2 = KNeighborsClassifier(n_neighbors=7)
    #     # clf3 = linear_model.LogisticRegression(random_state=1, C=50.0 / datas.shape[0], solver='saga', tol=0.01)
    #     # clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    #
    #     scores = cross_val_score(clf, datas_reduced, labels, scoring='accuracy', cv=5)
    #     scores = sum(scores) / len(scores)
    #     print(i, scores)

    # # ------------ submit.
    # test_datas, test_paths = get_test_data()
    # pca = PCA(n_components=18)
    # pca.fit(datas)
    # datas_reduced = pca.transform(datas)
    # clf = linear_model.LogisticRegression(random_state=1, C=50.0 / datas.shape[0], solver='saga', tol=0.01)
    # clf.fit(datas_reduced, labels)
    #
    # test_datas_reduced = pca.transform(test_datas)
    # test_pred_labels = clf.predict(test_datas_reduced)
    #
    # with open('data/pca_label_test.csv', 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['id', 'category'])
    #     for path, pred_label in zip(test_paths, test_pred_labels):
    #         writer.writerow([path, pred_label])

    # clf = svm.NuSVC()
    # clf.fit(x_train, y_train)
    # score = clf.score(x_val, y_val)
    # print(score)
    #
    # clf = svm.NuSVR()
    # clf.fit(x_train, y_train)
    # score = clf.score(x_val, y_val)
    # print(score)

    # ------------------ linear methods fail to converge -----------
    # clf = svm.LinearSVC()
    # clf.fit(x_train, y_train)
    # score = clf.score(x_val, y_val)
    # print(score)
    #
    # clf = svm.LinearSVR()
    # clf.fit(x_train, y_train)
    # score = clf.score(x_val, y_val)
    # print(score)
