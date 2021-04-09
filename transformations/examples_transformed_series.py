from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from sklearn.metrics import confusion_matrix
from scipy.stats import betaprime
from scipy.spatial.distance import directed_hausdorff
import sys

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")

ind=0

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


def scale(ref,start, end, k):
    shifted_t = ref.copy()
    shifted_t[range(start,end)] = ref[range(start,end)]*k
    return shifted_t


def noise(ref, start, end, k):
    shifted_t = ref.copy()
    noise = np.random.normal(0, np.abs(np.max(ref)-np.min(ref))*k/100 , len(range(start,end)))
    shifted_t[range(start,end)] = ref[range(start,end)]+noise
    return shifted_t



def shift(ref, shift_prefix, shift_sufix):
    shifted_t = ref[shift_prefix:(shift_sufix)]
    return shifted_t

#warp

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(warp(ref,100,200,1.2),linewidth=3, label="Warped TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(warp(ref,30,100,0.7),linewidth=3, label="Warped TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()




#scale

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(scale(ref,100,200,1.2),linewidth=3, label="Scaled TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(scale(ref,30,100,0.7),linewidth=3, label="Scaled TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()


#Noise

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(noise(ref,100,200,2),linewidth=3, label="TS with Noise")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(noise(ref,30,100,9),linewidth=3, label="TS with Noise")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()



#Shift


plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(shift(ref,5,len(ref)),linewidth=3, label="Shifted TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(shift(ref,40,200),linewidth=3, label="Shifted TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()

