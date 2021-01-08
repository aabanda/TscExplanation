
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np
import matplotlib.pyplot as plt
import random
from sktime.distances.elastic import dtw_distance
import numpy as np
from sktime.classifiers.dictionary_based import BOSSEnsemble, BOSSIndividual
from sklearn.metrics import f1_score


def intersection(intervals):
    start, end = intervals.pop()
    while intervals:
        start_temp, end_temp = intervals.pop()
        start = max(start, start_temp)
        end = min(end, end_temp)
    return [start, end]


def unwrapcircle(z):
    u = np.round(random.uniform(-z, 1), decimals=2)
    return intersection([[0,1],[u, u+z]])


#
# plt.plot(dydx)
# plt.plot(img*100000)

num_ins =500

# inte = np.zeros((num_ins,2))
# for i in range(0,num_ins):
#     inte[i,:]=unwrapcircle(0.8)
#
#
# inte[inte<=0]=0
# inte = inte*100
# inte = inte.astype(int)
# plt.hist(inte[:,1]-inte[:,0])
#
# np.max(inte[:,1]-inte[:,0])
#
# intervals = np.zeros((inte.shape[0],100))
# intervals[:,:] = 0
# for i in range(0,intervals.shape[0]):
#     intervals[i,range(inte[i,0],inte[i,1])] = 1
#
# dydx = np.sum(intervals, axis=0)
# plt.plot(dydx)
#
#
#







from scipy.stats import betaprime
a = 15
p= 0.5
b = a*(1-p)/p



inte = np.zeros((num_ins,2))
for i in range(0,num_ins):
    inte[i,:]=unwrapcircle(betaprime.rvs(a, b, size=1))


long = 128
inte[inte<=0]=0
inte = inte*long
inte = inte.astype(int)
plt.hist(inte[:,1]-inte[:,0])

np.max(inte[:,1]-inte[:,0])

intervals = np.zeros((inte.shape[0],long))
intervals[:,:] = 0
for i in range(0,intervals.shape[0]):
    intervals[i,range(inte[i,0],inte[i,1])] = 1

dydx = np.sum(intervals, axis=0)
plt.plot(dydx)