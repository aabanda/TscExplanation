from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.classifiers.dictionary_based import BOSSEnsemble, BOSSIndividual
from sklearn.metrics import f1_score
import pandas as pd
from sktime.classifiers.shapelet_based import ShapeletTransformClassifier




# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")

ind=0
# ind=100
# ind=160

ref = test_x.values[ind,:][0].values
plt.plot(ref)
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

l = len(ref)

start = 20
end = 100
k = 0.8
#
# start = 20
# end = 60
# k = 1.2
#
# start = 50
# end = 0
# k = 1.4

plt.plot(ref)
plt.plot(warp(ref, start, end, k))

start = 60
k = 30



def shift(ref, start, k):
    l = len(ref)
    shifted_t = np.zeros(l+k)
    shifted_t[range(start)] = ref[range(start)]
    shifted_t[range(start+k,len(shifted_t))]= ref[range(start,l)]


    before = ref[start-1]
    after = ref[start]

    for t in range(start,start+k):
        mean =(((start+k-t))* before + (k-(start+k-t)) * after)/k
        std = np.abs(before - after)/5
        shifted_t[t]= np.random.normal(mean,std,1)

    return shifted_t




plt.plot(ref)
plt.plot(shift(ref,start,k))





num_neig = 100
neig = []
inter = np.zeros((num_neig,2))
for i in range(0,num_neig):
     start = random.randint(1,len(ref)-10)
     k = random.randint(1,int(len(ref)*0.3))
     inter[i,:] = np.array([start,k])
     neig.append(shift(ref, start, k))







clf = BOSSEnsemble()
#clf = BOSSIndividual(window_size=50, word_length=8, alphabet_size=4,norm=False)
clf.fit(train_x,train_y.astype(int))
y_pred = clf.predict(test_x)
f1_score(test_y.astype(int),np.asarray(y_pred).astype(int),average='weighted')





pred_neig = []
for i in range(0,len(neig)):
    test_neig = np.transpose(np.column_stack((neig[i],neig[i])))
    pred_neig.append(clf.predict(test_neig)[0])
    #print(pred_neig[-1])

print(np.unique(pred_neig,return_counts=True))




clf = ShapeletTransformClassifier(time_contract_in_mins=5)
clf.fit(train_x,train_y)
y_pred = clf.predict(test_x)
f1_score(test_y.astype(int),y_pred.astype(int),average='weighted')




test_x2 = pd.DataFrame(test_x.values[range(0,2),:])

pred_neig = []
for i in range(0,len(neig)):
    test_x2.iloc[0,:][0] = pd.Series(neig[i])
    pred_neig.append(clf.predict(pd.DataFrame(test_x2.iloc[0,:]))[0])

print(np.unique(pred_neig,return_counts=True))
