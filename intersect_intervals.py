from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance




train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")


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




from scipy.stats import betaprime
a = 8
p=0.5
# a = 8
# p= 0.3
b = a*(1-p)/p


num_neig = 500


start_end = np.zeros((num_neig,2))
for i in range(0,num_neig):
    start_end[i,:]=unwrapcircle(betaprime.rvs(a, b, size=1))


start_end[start_end<=0]=0
start_end = start_end*len(ref)
start_end = start_end.astype(int)


#Comprobación

# plt.hist(start_end[:,1]-start_end[:,0])
#
#
# intervals = np.zeros((start_end.shape[0],len(ref)))
# intervals[:,:] = 0
# for i in range(0,intervals.shape[0]):
#     intervals[i,range(start_end[i,0],start_end[i,1])] = 1
#
# dydx = np.sum(intervals, axis=0)
# plt.plot(dydx)
#




neig = []
inter = np.zeros((num_neig,3))
for i in range(0,num_neig):
     start = start_end[i,0]
     end =  start_end[i,1]
     if end == len(ref):
         end = 0
     if start == 0:
         start = 1
     k = np.round(random.uniform(0.7,1.3), decimals=1)
     while k == 1:
         k = np.round(random.uniform(0.7, 1.3), decimals=1)
     inter[i,:] = np.array([start,end,k])
     neig.append(warp(ref, start, end, k))


# plt.hist(inter[:,0])
# plt.hist(inter[:,1]-inter[:,0])
inter[inter[:,1]==0,1]=len(ref)
variables = inter.copy()
variables = np.delete(variables,2,axis=1)

intersection([variables[0,:], variables[1,:]])