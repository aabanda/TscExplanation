from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from sklearn.metrics import confusion_matrix
from scipy.stats import betaprime
from scipy.spatial.distance import directed_hausdorff
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.interval_based import TimeSeriesForest
import sys


ind = int(sys.argv[1])
cls = str(sys.argv[2])
db = str(sys.argv[3])
transformation = str(sys.argv[4])


train_x, train_y = load_from_tsfile_to_dataframe("%s_TRAIN.ts" % db)
test_x, test_y = load_from_tsfile_to_dataframe("%s_TEST.ts" %db)

ref = test_x.values[ind,:][0].values
print(test_y[ind])




def warp(ts,start,end,scale):
    ref = ts
    k = scale
    l = len(ref)
    if start>0 and end>0:
        rescaled_t = np.concatenate((np.array(range(start)),
                                     np.array(range(0,end-start)) * k+start,
                                     range(np.int(start+(end-start)*k),(np.int(start+(end-start)*k) + l-end))))
    elif start>0 and end==0:
        rescaled_t = np.concatenate((np.array(range(start)), np.array(range(0, l - start)) * k + start))
    else:
        rescaled_t =  np.array(range(l))*k


    len(rescaled_t)

    new_index = np.array(range(np.int(np.max(rescaled_t))+1))
    t_trasnformed = np.zeros(len(new_index))

    end_ind =0
    for t in range(0,len(new_index)):
        if end==0:
            if t == 0:
                t_trasnformed[t] = ref[t]
            elif t <= start:
                t_trasnformed[t] = ref[t]
            else:
                before = rescaled_t[rescaled_t < t][-1]
                after = rescaled_t[rescaled_t >= t][0]

                t_trasnformed[t] = (ref[np.where(rescaled_t == before)[0][0]] * (k - (t - before)) + ref[
                    np.where(rescaled_t == after)[0][0]] * (
                                            k - (after - t))) / k
        else:
            if t == 0:
                t_trasnformed[t] = ref[t]
            elif t <= start:
                t_trasnformed[t] = ref[t]
            elif t < np.int(start + (end - start) * k):
                before = rescaled_t[rescaled_t < t][-1]
                after = rescaled_t[rescaled_t >= t][0]

                t_trasnformed[t] = (ref[np.where(rescaled_t == before)[0][0]] * (k - (t - before)) + ref[
                    np.where(rescaled_t == after)[0][0]] * (k - (after - t))) / k
            else:
                t_trasnformed[t] = ref[end + end_ind]
                end_ind = end_ind + 1

    return  t_trasnformed




def shift(ref, shift_prefix, shift_sufix):
    shifted_t = ref[shift_prefix:(shift_sufix)]
    return shifted_t





def scale(ref,start, end, k):
    shifted_t = ref.copy()
    shifted_t[range(start,end)] = ref[range(start,end)]*k
    return shifted_t






def noise(ref, start, end, k):
    shifted_t = ref.copy()
    noise = np.random.normal(0, np.abs(np.max(ref)-np.min(ref))*k/100 , len(range(start,end)))
    shifted_t[range(start,end)] = ref[range(start,end)]+noise
    return shifted_t





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





a = 8
p= 0.3
b = a*(1-p)/p


num_neig = 500


start_end = np.zeros((num_neig,2))
for i in range(0,num_neig):
    start_end[i,:]=unwrapcircle(betaprime.rvs(a, b, size=1))


start_end[start_end<=0]=0
start_end = start_end*len(ref)
start_end = start_end.astype(int)



if transformation=="warp":
    neig = []
    inter = np.zeros((num_neig*6,3))
    count = 0
    for i in range(0,num_neig):
         start = start_end[i,0]
         end =  start_end[i,1]
         if end == len(ref):
             end = 0
         if start == 0:
             start = 1
         for k in [0.7,0.8,0.9,1.1,1.2,1.3]:
             inter[count, :] = np.array([start, end, k])
             neig.append(warp(ref, start, end, k))
             count = count+1

elif transformation=="shift":

    start_end = np.zeros((num_neig, 2))
    for i in range(0, num_neig):
        start_end[i, :] = unwrapcircle(betaprime.rvs(a, b, size=1))

    start_end[start_end <= 0] = 0
    start_end = start_end * len(ref)
    start_end = start_end.astype(int)

    for i in range(num_neig):
        while start_end[i, 0] == start_end[i, 1]:
            start_end[i, :] = unwrapcircle(betaprime.rvs(a, b, size=1))
            start_end[i, :][start_end[i, :] <= 0] = 0
            start_end[i, :] = start_end[i, :] * len(ref)
            start_end[i, :] = start_end[i, :].astype(int)

    num_neig = 500
    neig = []
    count = 0
    inter = np.zeros((num_neig, 2))
    for i in range(0, num_neig):
        shi_pre = start_end[i, 0]
        shi_suf = start_end[i, 1]
        inter[i, :] = np.array([shi_pre, shi_suf])
        neig.append(shift(ref, shi_pre, shi_suf))
        count = count + 1

elif transformation=="scale":

    neig = []
    inter = np.zeros((num_neig * 6, 3))
    count = 0
    for i in range(0, num_neig):
        start = start_end[i, 0]
        end = start_end[i, 1]
        if end == len(ref):
            end = 0
        if start == 0:
            start = 1
        for k in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
            inter[count, :] = np.array([start, end, k])
            neig.append(scale(ref, start, end, k))
            count = count + 1



elif transformation=="noise":

    num_neig = 500
    neig = []
    inter = np.zeros((num_neig * 5, 3))
    count = 0
    for i in range(0, num_neig):
        start = start_end[i, 0]
        end = start_end[i, 1]
        for k in [1, 3, 5, 7, 9]:
            inter[count, :] = np.array([start, end, k])
            neig.append(noise(ref, start, end, k))
            count = count + 1

else:

    print("Trasnformation error: warp, shift, scale, noise")





distance_matrix = np.zeros((len(neig), train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i, j] = dtw_distance(neig[i], np.asarray(train_x.values[j, :][0]))

neig_y = train_y[np.argmin(distance_matrix, axis=1)]
print(np.unique(np.asarray(neig_y),return_counts=True))


np.savetxt("%s/%s/%s/neig_%d.txt" % (db,transformation,cls,ind), np.asarray(neig_y).astype(int))
np.savetxt("%s/%s/%s/inter_%d.txt" % (db,transformation,cls,ind), inter)



