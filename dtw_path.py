from dtaidistance import dtw
s1 = [0, 0, 1, 2 ]
s2 = [0, 1, 2, 0]
distance, paths = dtw.warping_paths(s1, s2)
print(distance)
print(paths)



from dtw import *
from sktime.distances.elastic import dtw_distance
import random
import numpy as np
from dtaidistance import dtw_visualisation as dtwvis
from sktime.utils.load_data import load_from_tsfile_to_dataframe

x = np.arange(0, 20, .5)
s1 = np.sin(x)
s2 = np.sin(x - 1)
random.seed(1)
for idx in range(len(s2)):
    if random.random() < 0.05:
        s2[idx] += (random.random() - 0.5) / 2
d, paths = dtw.warping_paths(s1, s2, window=25, psi=2)
paths
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(s1, s2, paths, best_path)



import matplotlib.pyplot as plt

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")


ind_test = 0
test_y[ind_test]


dist = []
for i in range(0,len(train_y)):
    dist.append(dtw_distance(test_x.values[ind_test,:],train_x.values[i,:]))

dist = np.asarray(dist)

ref1 = np.where(train_y=='1')[0][np.argmin(dist[np.where(train_y=='1')[0]])]
ref2 = np.where(train_y=='2')[0][np.argmin(dist[np.where(train_y=='2')[0]])]
ts = np.asarray(test_x.values[0,:])[0]

alignment = dtw(train_x.iloc[ref1,:][0],ts, keep_internals=True)
alignment.plot(type="threeway")
alignment = dtw(train_x.iloc[ref2,:][0], ts, keep_internals=True)
alignment.plot(type="threeway")
## Display the warping curve, i.e. the alignment curve

alignment.costMatrix




train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

ind_test = 0
test_y[ind_test]


dist = []
for i in range(0,len(train_y)):
    dist.append(dtw_distance(test_x.values[ind_test,:],train_x.values[i,:]))

dist = np.asarray(dist)


ref1 = np.where(train_y=='2')[0][np.argmin(dist[np.where(train_y=='2')[0]])]
ref2 = np.where(train_y!='2')[0][np.argmin(dist[np.where(train_y!='2')[0]])]
train_y[ref1]
train_y[ref2]

ts = np.asarray(test_x.values[ind_test,:])[0]

alignment = dtw(train_x.iloc[ref1,:][0],ts, keep_internals=True)
alignment.distance
alignment.plot(type="threeway")
alignment = dtw(train_x.iloc[ref2,:][0], ts, keep_internals=True)
alignment.distance
c = alignment.costMatrix
alignment.plot(type="threeway")



a = np.array([1,3,5,1,3,2])
b = np.array([2,3,2,5,1,3])

plt.plot(a)
plt.plot(b)


alignment = dtw(b,a ,keep_internals=True)
alignment.distance
alignment.plot(type="threeway")


b = np.array([1,3,2,5,1,3])
alignment = dtw(b,a, keep_internals=True)
alignment.distance
alignment.plot(type="threeway")

b = np.array([2,3,3,5,1,3])
alignment = dtw(b,a, keep_internals=True)
alignment.distance
alignment.plot(type="threeway")

b = np.array([2,3,5,6,1,3])
alignment = dtw(b,a, keep_internals=True)
alignment.costMatrix
alignment.distance
alignment.plot(type="threeway")




b = np.array([1,3,2,5,1,2])
alignment = dtw(b,a, keep_internals=True)
alignment.distance
alignment.plot(type="threeway")




a = np.array([1,3,5,1,3,2])
b = np.array([1,3,3,5,1,3])

alignment = dtw(b,a ,keep_internals=True)
alignment.distance
M = alignment.costMatrix
alignment.plot(type="threeway")




len2 = M.shape[1]

b_con = np.zeros((len2))
b_con[0] = a[0]- M[0,0]
b_con[1] = a[1] - M[1,1] + np.min([M[0,1],[M[1,0]],M[0,0]])

b_con[2] = a[2] - M[2,2] + np.min([M[1,2],[M[2,1]],M[2,2]])

b_con[3] = a[3] - M[3,3] + np.min([M[2,3],[M[3,2]],M[3,3]])
b_con[3] = a[3] + M[3,3]- np.min([M[2,3],[M[3,2]],M[3,3]])

b_con[4] = a[4] + M[4,4]- np.min([M[3,4],[M[4,3]],M[4,4]])
b_con[4] = a[4] - M[4,4]+ np.min([M[3,4],[M[4,3]],M[4,4]])

b_con[5] = a[5] + M[5,5]- np.min([M[4,5],[M[5,4]],M[5,5]])
b_con[4] = a[4] - M[4,4]+ np.min([M[3,4],[M[4,3]],M[4,4]])

b
b_con

alignment = dtw(b_con,a ,keep_internals=True)
alignment.distance
M = alignment.costMatrix
alignment.plot(type="threeway")


M



