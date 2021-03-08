#from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import f1_score
from sktime.classification.dictionary_based import BOSSEnsemble
import pandas as pd


len_shape1= 5

class1 = np.random.rand(100,150)
class2 = np.random.rand(100,150)

for i in range(0,class1.shape[0]):
    start = random.randrange(1,class1.shape[1]-20)
    class1[i,np.arange(start,start+20)] = 3
    class2[i,np.arange(start,start+20)] = 6

plt.plot(class1[2,:])
plt.ylim(0,6)
plt.plot(class2[2,:])

#
# len_shape1= 25
# class2 = np.random.rand(100,150)
# for i in range(0,class1.shape[0]):
#     start = random.randrange(1,class1.shape[1]-50)
#     class2[i,np.arange(start,start+50)] = 3
#
# plt.plot(class2[4,:])


y = np.repeat([0,1],[100,100])
X = np.concatenate([class1, class2])




#Classification with  DTW
distance_matrix = np.zeros((X.shape[0], X.shape[0]))
for i in range(0, distance_matrix.shape[0]):
    for j in range(0, distance_matrix.shape[0]):
        distance_matrix[i,j] = dtw_distance(X[i,:], X[j,:])

kf = StratifiedKFold(n_splits=5, shuffle=True)
accu = []
for train_index, test_index in kf.split(X,y):
    y_train, y_test = y[train_index], y[test_index]
    distance_matrix_fold = distance_matrix[test_index, :]
    distance_matrix_fold = distance_matrix_fold[:, train_index]

    pred = []
    for i in range(0,len(test_index)):
        ind = np.argmin(distance_matrix_fold[i,:])
        pred.append(y_train[ind])
    accu.append(f1_score(y_test, pred, average="weighted"))

np.mean(accu)


#Classification with BOSS
X = np.concatenate([class1, class2])

tsf = BOSSEnsemble(max_ensemble_size=100)
X_data = pd.DataFrame(X)
X = pd.DataFrame()
X["dim_0"] = [pd.Series(X_data.iloc[x, :]) for x in range(len(X_data))]
accu = []
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    tsf.fit(X_train, y_train)
    y_pred = tsf.predict(X_test)
    accu.append(f1_score(y_test.astype(int), y_pred.astype(int), average="weighted"))

print(np.mean(accu))



y = np.repeat([0,1],[100,100])
X = np.concatenate([class1, class2])

kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index,:], X[test_index,:]
    print("A")

X = np.concatenate([class1, class2])
ind =0
#plt.plot(X[test_index[ind],:])
print(y[test_index[ind]])

from functions import explanation1, explanation2

explanation1(X[test_index[ind],:], y[test_index[ind]], 0.8, X[train_index,:], y[train_index],classifier="boss")
explanation1(X[test_index[ind],:], y[test_index[ind]], 0.8, X[train_index,:], y[train_index],classifier="dtw")

explanation2(X[test_index[ind],:], y[test_index[ind]], 0.8, X[train_index,:], y[train_index],classifier="boss")
