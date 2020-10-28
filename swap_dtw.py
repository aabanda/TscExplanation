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

from sktime.classifiers.shapelet_based import ShapeletTransformClassifier
from sktime.classifiers.distance_based import KNeighborsTimeSeriesClassifier
knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")
knn.fit(train_x, train_y)

indx_test = 1
ts =test_x.iloc[indx_test,:]
print(test_y[indx_test])

plt.plot(ts[0])
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

ind_list = f_ind(list(range(0,len(ts[0]))))


#print(tsf.predict(ts))

swaped = f(np.asarray(ts[0]))
swaped_ind = f(list(range(0,len(np.asarray(ts[0])))))
swaped_ind[0]
len(swaped)


pred = []
for i in range(0,len(swaped)):
    a = swaped[i]
    test_x.iloc[0, :][0] = pd.Series(a)
    pred.append(knn.predict(pd.DataFrame(test_x.iloc[0, :]))[0].astype(int))
    print(pred[-1])

np.sum((np.asarray(pred)  != test_y[indx_test].astype(int))) / len(swaped)
np.sum((np.asarray(pred)  == 1)) / len(swaped)


pre2 = np.where(np.asarray(pred) != test_y[indx_test].astype(int))
#pre1 = np.where(np.asarray(pred) == 1)[0]

permus = np.zeros((len(pre2), len(swaped_ind[0])))


for i in range(0, permus.shape[0]):
    permus[i,:] = swaped_ind[pre2[0][i]]



pemu_matrix = np.zeros((len(ts[0]),len(ts[0])))


pre_array = np.asarray(pred)
for p in range(0, len(ind_list)):
    pemu_matrix[ind_list[p][0], ind_list[p][1]] = pre_array[p]


plt.imshow(pemu_matrix, cmap='hot', interpolation='nearest')
plt.show()










np.sum(permus, axis=0)

div = np.asarray(list(range(0,150)))
div = div *  permus.shape[0]


cam = np.sum(permus, axis=0)   /div
cam[0]= 1
np.min(cam)
permus_cambian = swaped_ind[pre2]
len(pre2)
len(pred)

