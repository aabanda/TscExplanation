#from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance



inter = np.loadtxt("inter_warp_3000.txt")
neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")

weigths1 = np.loadtxt("weigths1.txt")
weigths2 = np.loadtxt("weigths2.txt")

weigths1 = (weigths1-np.min(weigths1))/(np.max(weigths1)-np.min(weigths1))
weigths2 = (weigths2-np.min(weigths2))/(np.max(weigths2)-np.min(weigths2))

plt.plot(weigths1, label="weights 1")
plt.plot(weigths2, label="weights 2")
plt.legend()

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

#CBF
# ind =2 #class 1
# ind = 0 #Class 2
ind = 5 #Class 3
# #
# # Arrowhead
# ind=0
# ind=100
# ind=160

ref = test_x.values[ind,:][0].values
# plt.plot(ref)
print(test_y[ind])


def warp(ts,start,end,scale):
    ref = ts
    k = scale
    l = len(ref)
    # start = 20
    # end = 60
    # k = 0.8
    #
    #
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





#generate intervals 1 : centrado


num_neig=500
neig = []
start_end = np.zeros((num_neig,2))
count = 0
for i in range(0,num_neig):
    center = random.choices(np.arange(len(ref)), weigths1)[0]
    if np.min([len(ref)-center, center-1])<10:
        leng = np.min([len(ref)-center, center-1])
    else:
        leng = random.randint(5,np.min([len(ref)-center, center-1]))
    start_end[i,0] = center-leng
    start_end[i,1] = center+leng


intervals = np.zeros((start_end.shape[0], len(ref)))
intervals[:, :] = np.nan
for i in range(0, intervals.shape[0]):
    intervals[i, range(start_end[i, 0].astype(int), start_end[i, 1].astype(int))] = i
    plt.plot(range(0, len(ref)), intervals[i, :])

intervals = np.zeros((start_end.shape[0], len(ref)))
for i in range(0, intervals.shape[0]):
    intervals[i, range(start_end[i, 0].astype(int), start_end[i, 1].astype(int))] = 1

plt.plot(np.mean(intervals,axis=0), label="Generation")
plt.plot(weigths1, label="weights")
plt.legend()
plt.title("Centered")





#generate intervals 2 : incluido



num_neig=500
neig = []
start_end = np.zeros((num_neig,2))
count = 0
for i in range(0,num_neig):
    center = random.choices(np.arange(len(ref))[2:], weigths1[2:])[0]
    len_left = random.randint(1,np.min([center-1,20]))
    len_right = random.randint(1,np.min([len(ref)-center,20]))
    start_end[i,0] = center-len_left
    start_end[i,1] = center+len_right

intervals = np.zeros((start_end.shape[0], len(ref)))
intervals[:, :] = np.nan
for i in range(0, intervals.shape[0]):
    intervals[i, range(start_end[i, 0].astype(int), start_end[i, 1].astype(int))] = i
    plt.plot(range(0, len(ref)), intervals[i, :])

intervals = np.zeros((start_end.shape[0], len(ref)))
for i in range(0, intervals.shape[0]):
    intervals[i, range(start_end[i, 0].astype(int), start_end[i, 1].astype(int))] = 1

plt.plot(np.mean(intervals,axis=0), label="Generation")
plt.plot(weigths1, label="weights")
plt.legend()
plt.title("Inlcuded")










num_neig= 500
neig = []
inter = np.zeros((num_neig*6,3))
count = 0
for i in range(0,num_neig):
     start = start_end[i,0].astype(int)
     end =  start_end[i,1].astype(int)
     if end == len(ref):
         end = 0
     if start == 0:
         start = 1
     #for k in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
     for k in [0.8]:
         inter[count, :] = np.array([start, end, k])
         neig.append(warp(ref, start, end, k))
         count = count+1



distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))

neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))