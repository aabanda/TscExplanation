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
# #Separado por wapr level
# inter = np.loadtxt("inter_warp_3000.txt")
# neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")

ind = int(sys.argv[1])
cls = str(sys.argv[2])
db = str(sys.argv[3])
transformation = str(sys.argv[4])


train_x, train_y = load_from_tsfile_to_dataframe("%s_TRAIN.ts" % db)
test_x, test_y = load_from_tsfile_to_dataframe("%s_TEST.ts" %db)

clf = TimeSeriesForest(n_estimators=100)
clf.fit(train_x, train_y)


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



for ind in range(len(test_y)):

    ref = test_x.values[ind, :][0].values
    print(test_y[ind])

    inter = np.loadtxt("%s/%s/dtw/inter_%d.txt" % (db, transformation, cls, ind))

    num_neig = 500

    if transformation == "warp":

        neig = []
        for i in range(0, inter.shape[0]):
            neig.append(warp(ref, inter[i, 0], inter[i, 1], inter[i, 2]))


    elif transformation == "shift":

        neig = []
        for i in range(0, inter.shape[0]):
            neig.append(shift(ref, inter[i, 0], inter[i, 1]))

    elif transformation == "scale":

        neig = []
        for i in range(0, inter.shape[0]):
            neig.append(scale(ref, inter[i, 0], inter[i, 1], inter[i, 2]))



    elif transformation == "noise":

        neig = []
        for i in range(0, inter.shape[0]):
            neig.append(noise(ref, inter[i, 0], inter[i, 1], inter[i, 2]))


    else:

        print("Trasnformation error: warp, shift, scale, noise")

    prueba = clf.predict(test_x)

    neig_y = []
    for i in range(len(neig)):
        neig_y.append(clf.predict(neig[i].reshape(1, 1, -1))[0])

    np.savetxt("%s/%s/%s/neig_%d.txt" % (db, transformation, cls, ind), np.asarray(neig_y).astype(int))









