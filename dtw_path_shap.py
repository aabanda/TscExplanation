from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np
from sktime.distances.elastic import dtw_distance
from dtw import *
import matplotlib.pyplot as plt

# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ShapeletSim/ShapeletSim_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ShapeletSim/ShapeletSim_TEST.ts")


train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

ind_test = 0
test_y[ind_test]
np.unique(train_y)
plt.plot(train_x.iloc[ind_test,:][0])


dist = []
for i in range(0,len(train_y)):
    dist.append(dtw_distance(test_x.values[ind_test,:],train_x.values[i,:]))

dist = np.asarray(dist)

ref1 = np.where(train_y=='2')[0][np.argmin(dist[np.where(train_y=='2')[0]])]
ref2 = np.where(train_y!='2')[0][np.argmin(dist[np.where(train_y!='2')[0]])]

ts = np.asarray(test_x.values[ind_test,:])[0]
plt.plot(ts)
plt.plot(train_x.iloc[ref1,:][0])

plt.plot(ts)
plt.plot(train_x.iloc[ref2,:][0])

alignment = dtw(train_x.iloc[ref1,:][0],ts, keep_internals=True)
alignment.distance
alignment.plot(type="threeway")

alignment = dtw(train_x.iloc[ref2,:][0], ts, keep_internals=True)
alignment.distance
alignment.plot(type="threeway")

d = alignment.costMatrix
