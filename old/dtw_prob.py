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

from sktime.distances.elastic import dtw_distance
from sktime.classifiers.shapelet_based import ShapeletTransformClassifier


train_x, train_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")
np.unique(test_y)




train_x, train_y = load_from_tsfile_to_dataframe(
    "../../datasets/Univariate_ts/ItalyPowerDemand/ItalyPowerDemand_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/ItalyPowerDemand/ItalyPowerDemand_TEST.ts")
np.unique(test_y)


dtw_distance(train_x.values[0,:],train_x.values[1,:])

ind_test = 0
test_y[ind_test]

plt.plot(np.asarray(test_x.values[0,:])[0])

dist = []
for i in range(0,len(train_y)):
    dist.append(dtw_distance(test_x.values[0,:],train_x.values[i,:]))

dist = np.asarray(dist)

# dist[np.where(train_y=='1')[0]]
# np.where(train_y=='1')[0][np.argmin(dist[np.where(train_y=='1')[0]])]
# np.where(train_y=='2')[0][np.argmin(dist[np.where(train_y=='2')[0]])]
ref1 = np.where(train_y=='1')[0][np.argmin(dist[np.where(train_y=='1')[0]])]
ref2 = np.where(train_y=='2')[0][np.argmin(dist[np.where(train_y=='2')[0]])]

# ref1 = np.where(train_y=='0')[0][np.argmin(dist[np.where(train_y=='0')[0]])]
# ref2 = np.where(train_y=='1')[0][np.argmin(dist[np.where(train_y=='1')[0]])]
#

plt.plot(test_x.values[ref1,:][0])
plt.plot(test_x.values[ref2,:][0])


p1 = dist[ref1]
p2 = dist[ref2]

p1 = p2/(p1+p2)
p2 = 1-p1


def f(s):
    result = []
    # This will produce: [0, 1], [0, 2], [1, 2]
    for idx1, idx2 in itertools.combinations(range(len(s)), 2):
        swapped_s = list(s)
        swapped_s[idx1], swapped_s[idx2] = swapped_s[idx2], swapped_s[idx1]
        #result.append(''.join(swapped_s))
        result.append((swapped_s))
    return result

swaped = f(np.asarray(test_x.values[0,:])[0])
len(swaped)
# num_sample = 1000
# ind_random = random.sample(range(0, len(swaped)), num_sample)
num_sample = len(swaped)
ind_random =range(0, len(swaped))

prob_neig = np.zeros((num_sample,2))
for i in range(0,num_sample):
    prob_neig[i,0] = dtw_distance(np.array(swaped[ind_random[i]]),train_x.values[ref1,:][0])
    prob_neig[i,1] = dtw_distance(np.array(swaped[ind_random[i]]), train_x.values[ref2,:][0])

for i in range(0, num_sample):
    prob_neig[i,0] = prob_neig[i,1] / (prob_neig[i,0] + prob_neig[i,1])
    prob_neig[i,1] = 1 -  prob_neig[i,0]


plt.hist(prob_neig[:,0])

np.min(prob_neig[:,0])



plt.plot(swaped[np.argmin(prob_neig[:,0])])