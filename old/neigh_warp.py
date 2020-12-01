from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import f1_score
from sktime.classifiers.interval_based import TimeSeriesForest
from itertools import permutations
import itertools as it
from sktime.distances.elastic import dtw_distance



train_x, train_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/CBF/CBF_TEST.ts")

ind =2 #class 1
ind = 0 #Class 2
ind = 5 #Class 2
ref = test_x.values[ind,:][0].values
plt.plot(ref)
test_y[ind]


start = random.randrange(len(ref)-20)
len_shift = random.randrange(1,int((len(ref)-start)/2))

ts = ref.copy()
range(start,start+2*len_shift,2)
ts[range(start,start+2*len_shift,2)] = ref[range(start,start+len_shift)]

for t in range(start+1,start+2*len_shift,2):
    ts[t] = np.mean([ts[t-1],ts[t+1]])

plt.plot(ts)


features = np.zeros((test_x.shape[0],2))

for sample in range(0,test_x.shape[0]):

    start = random.randrange(len(ref) - 20)
    len_shift = random.randrange(1, int((len(ref) - start) / 2))

    ts = ref.copy()
    range(start, start + 2 * len_shift, 2)
    ts[range(start, start + 2 * len_shift, 2)] = ref[range(start, start + len_shift)]

    for t in range(start + 1, start + 2 * len_shift, 2):
        ts[t] = np.mean([ts[t - 1], ts[t + 1]])

    test_x.values[sample,:][0]= ts
    features[sample,0] = start
    features[sample,1] = len_shift


#plt.plot(test_x.values[6,:][0])

distance_matrix = np.zeros((test_x.shape[0],train_x.shape[0]))

for i in range(0,test_x.shape[0]):
    for j in range(0, train_x.shape[0]):
        #distance_matrix[i,j] = dtw_distance(test_x.values[i,:][0], train_x.values[j,:][0])
        distance_matrix[i,j] = np.linalg.norm(test_x.values[i,:][0]-train_x.values[j,:][0])



test_y = train_y[np.argmin(distance_matrix,axis=1)]

np.unique(test_y,return_counts=True)
# test_x2 = test_x.copy()
# test_x.values.shape
# test_x2 = test_x2.values[:,0]
# test_x2.shape
test_x3 = np.zeros((900,128))
for i in range(0,900):
    test_x3[i,:] = test_x.values[i,0]

plt.plot(test_x3[100,:])

# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report

clf = LogisticRegression(random_state=0,max_iter=5000)

kf = StratifiedKFold(n_splits=3)
accu = []
for train_index, test_index in kf.split(test_x,test_y):
    X_train, X_test = features[train_index,:], features[test_index,:]
    y_train, y_test = test_y[train_index], test_y[test_index]

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accu.append(np.sum(pred==y_test)/len(pred))
    print(classification_report(y_test, pred))

print(np.mean(accu))

clf.coef_







from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)


kf = StratifiedKFold(n_splits=2)
accu = []
for train_index, test_index in kf.split(test_x,test_y):
    X_train, X_test = features[train_index,:], features[test_index,:]
    y_train, y_test = test_y[train_index], test_y[test_index]

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accu.append(np.sum(pred==y_test)/len(pred))
    print(classification_report(y_test, pred))


print(np.mean(accu))


X_train = features
y_train = test_y

clf.fit(X_train, y_train)
# pred = clf.predict(X_test)
# accu.append(np.sum(pred==y_test)/len(pred))
# print(classification_report(y_test, pred))


from sklearn import tree

#tree.plot_tree(clf)
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf, class_names=np.array(["C1","C2","C3"]),impurity=False, fontsize=10)
plt.show()

plt.scatter(features[:,0],features[:,1],c=test_y.astype(int))



#
#
# def get_lineage(tree, feature_names):
#     left = tree.tree_.children_left
#     right = tree.tree_.children_right
#     threshold = tree.tree_.threshold
#     features = [feature_names[i] for i in tree.tree_.feature]
#
#     # get ids of child nodes
#     idx = np.argwhere(left == -1)[:, 0]
#
#     def recurse(left, right, child, lineage=None):
#         if lineage is None:
#             lineage = [child]
#         if child in left:
#             parent = np.where(left == child)[0].item()
#             split = 'l'
#         else:
#             parent = np.where(right == child)[0].item()
#             split = 'r'
#
#         lineage.append((parent, split, threshold[parent], features[parent]))
#
#         if parent == 0:
#             lineage.reverse()
#             return lineage
#         else:
#             return recurse(left, right, parent, lineage)
#
#     for child in idx:
#         for node in recurse(left, right, child):
#             print(node)
#
#
# get_lineage(clf, np.array([1,2]))