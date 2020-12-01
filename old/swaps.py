import itertools
import numpy as np
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

import seaborn as sns
from sktime.classifiers.shapelet_based import ShapeletTransformClassifier


train_x, train_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")


a = np.array(["1","2","3","4"])
a = np.array( [1,2,3,4])

def f(s):
    result = []
    # This will produce: [0, 1], [0, 2], [1, 2]
    for idx1, idx2 in itertools.combinations(range(len(s)), 2):
        swapped_s = list(s)
        swapped_s[idx1], swapped_s[idx2] = swapped_s[idx2], swapped_s[idx1]
        #result.append(''.join(swapped_s))
        result.append((swapped_s))
    return result


def f_ind(s):
    result = []
    # This will produce: [0, 1], [0, 2], [1, 2]
    for idx1, idx2 in itertools.combinations(range(len(s)), 2):

        result.append(([idx1,idx2]))
    return result




np.unique(train_y)
tsf = ShapeletTransformClassifier(time_contract_in_mins=5)
tsf.fit(train_x, train_y)


indx_test =1
indx_test =0
ts = test_x.iloc[indx_test,:]
test_y[indx_test]
#print(tsf.predict(ts))

ind_list = f_ind(list(range(0,len(ts[0]))))








swaped = f(np.asarray(ts[0]))
swaped_ind = f(list(range(0,len(np.asarray(ts[0])))))
swaped_ind[0]
len(swaped)


pred = []
for i in range(0,len(swaped)):
    a = swaped[i]
    test_x.iloc[0, :][0] = pd.Series(a)
    pred.append(tsf.predict(pd.DataFrame(test_x.iloc[0, :]))[0].astype(int))
    print(pred[-1])

np.sum((np.asarray(pred)!= test_y[indx_test].astype(int))) / len(swaped)

pre2 = np.where(np.asarray(pred) != test_y[indx_test].astype(int))[0]
permus = np.zeros((len(pre2), len(swaped_ind[0])))
for i in range(0, permus.shape[0]):
    permus[i,:] = swaped_ind[pre2[i]]

np.sum(permus, axis=0)
div = np.asarray(list(range(0,150)))
div = div *  permus.shape[0]
cam = np.sum(permus, axis=0)   /div
cam[0]= 1
np.min(cam)
permus_cambian = swaped_ind[pre2]
len(pre2)
len(pred)











pemu_matrix = np.zeros((len(ts[0][0]),len(ts[0][0])))
pemu_matrix = np.zeros((len(ts[0]),len(ts[0])))


pre_array = np.asarray(pred)
for p in range(0, len(ind_list)):
    pemu_matrix[ind_list[p][0], ind_list[p][1]] = pre_array[p]


plt.imshow(pemu_matrix, cmap='hot', interpolation='nearest')
plt.show()


# Tambien tener en cuenta la diferencia
ind_array = np.asarray(ind_list)
permus_change =ind_array[pre_array!=test_y[indx_test].astype(int)]
len(permus_change)

a = np.unique(permus_change[:,0],return_counts=True)
b = np.unique(permus_change[:,1],return_counts=True)



difference = np.zeros((len(a[0])+ len(b[0]),3))
difference[:,0] = np.concatenate([a[0],b[0]])


for i in range(0,len(a[0])):

    aux = permus_change[permus_change[:,0]==difference[i,0],:]
    p = []
    for j in range(0, aux.shape[0]):
        p.append(ts[0][aux[j,1]]- ts[0][aux[j,0]])

    difference[i,1] = np.mean(p)
    difference[i,2] = np.std(p)


for i in range(len(a[0]),difference.shape[0]):

    aux = permus_change[permus_change[:,1]==difference[i,0],:]
    p = []
    for j in range(0, aux.shape[0]):
        p.append(ts[0][aux[j,0]]- ts[0][aux[j,1]])

    difference[i,1] = np.mean(p)
    difference[i,2] = np.std(p)


color_difference = np.zeros(len(ts[0]))
color_difference[difference[:,0].astype(int)] = difference[:,1]


3#plt.scatter(range(0,len(test_x.iloc[indx_test,:][0])),test_x.iloc[indx_test,:][0],c=color_difference)


cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots()
points = ax.scatter(range(0,len(test_x.iloc[indx_test,:][0])),test_x.iloc[indx_test,:][0],c=color_difference, s=50)
f.colorbar(points)



# cuantos = np.zeros((permus.shape[1]))
# for id in range(0,permus.shape[1]):
#     #id = 2
#     cuantos[id]=np.sum(permus[:,id]!=id)
#
# plt.scatter(range(0,len(test_x.iloc[indx_test,:][0])),test_x.iloc[indx_test,:][0],c=cuantos)
#



#
#
# import seaborn as sns
# cmap = sns.cubehelix_palette(as_cmap=True)
#
# f, ax = plt.subplots()
# points = ax.scatter(range(0,len(test_x.iloc[indx_test,:][0])),test_x.iloc[indx_test,:][0],c=cuantos, s=50)
# f.colorbar(points)
#
#
# np.sum(cuantos)
#
#
#
# pre1 = np.where(np.asarray(pred) == 1)[0]
# permus = np.zeros((len(pre1), len(swaped_ind[0])))
# for i in range(0, permus.shape[0]):
#     permus[i,:] = swaped_ind[pre1[i]]
#
# np.sum(permus, axis=0)
# div = np.asarray(list(range(0,150)))
# div = div *  permus.shape[0]
# cam = np.sum(permus, axis=0)   /div
# cam[0]= 1
# np.min(cam)
# permus_cambian = swaped_ind[pre2]
# len(pre2)
# len(pred)