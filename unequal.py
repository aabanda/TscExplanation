from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from sktime.classifiers.dictionary_based import BOSSEnsemble, BOSSIndividual
from sklearn.metrics import f1_score
import pandas as pd


train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")


clf = BOSSIndividual(window_size=50, word_length=8, alphabet_size=4,norm=False)
clf.fit(train_x,train_y)
y_pred = clf.predict(test_x)
f1_score(test_y.astype(int),y_pred,average='weighted')



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

start = 50
end = 0
k = 1.4

ref = test_x.values[0,:][0]

warped = warp(ref, start, end, k)
len(warped)

neig = np.vstack((warped,warped))
neig = pd.DataFrame(neig)



y_pred = clf.predict(test_x)
test_x.shape



for i in range(0,test_x.shape[0]):
    test_x.iloc[i, :][0] = pd.Series(warped)


plt.plot(test_x.values[0,:][0])


train_x.values[0,:]
test_x.iloc[1,:][0] = pd.Series(warped)

test_x.shape[0]

# from sktime.utils.load_data import from_long_to_nested
# X_nested = from_long_to_nested(neig)

y_pred = clf.predict(test_x)


f1_score(test_y.astype(int),y_pred,average='weighted')
