from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from sktime.distances.elastic import dtw_distance
# from sktime.classifiers.dictionary_based import BOSSEnsemble, BOSSIndividual
# from sklearn.metrics import f1_score
# import pandas as pd
# from sktime.classifiers.shapelet_based import ShapeletTransformClassifier
#


#WARP 1


train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")



train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GesturePebbleZ1/GesturePebbleZ1_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GesturePebbleZ1/GesturePebbleZ1_TEST.ts")

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")


ind_ref = 0
ref = test_x.values[ind_ref,:][0]
test_y[ind_ref]


plt.plot(ref)

np.unique(train_y)










ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])


ind=0
s= train_x.values[ind,:][0].values
len(s)
plt.plot(s)



per_class =3
matplotlib.use('Agg')  # turn off gui

fig, axs = plt.subplots(len(np.unique(train_y)),per_class, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)


axs = axs.ravel()

i = 0
for c in range(len(np.unique(train_y))):
    for j in range(per_class):
        ind = ind_class[c][j]
        axs[i].plot(train_x.values[ind,:][0].values)
        i = i+1

#plt.plot(train_x.values[3,:][0].values)
# #axs[i].set_title(str(250+i))




per_class =3
fig, axs = plt.subplots(len(np.unique(train_y)),1, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for c in range(len(np.unique(train_y))):
    for j in range(per_class):
        ind = ind_class[c][j]
        axs[c].plot(train_x.values[ind,:][0].values)
