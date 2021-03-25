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
db = "ArrowHead"
#
db = "GunPoint"
db = "Meat"
db = "Car"
db = "BeetleFly"
db = "Coffee"
db = "ECG200"
db = "ECGFiveDays"
db = "GesturePebbleZ1"




#db = "Adiac"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))




ind_ref = 0
ref = test_x.values[ind_ref,:][0]
test_y[ind_ref]


#plt.plot(ref)
print(np.unique(train_y))










ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])


ind=0
s= train_x.values[ind,:][0].values
len(s)
# plt.plot(s)



per_class =5
#matplotlib.use('Agg')  # turn off gui

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




per_class =5
fig, axs = plt.subplots(len(np.unique(train_y)),1, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for c in range(len(np.unique(train_y))):
    for j in range(per_class):
        ind = ind_class[c][j]
        axs[c].plot(train_x.values[ind,:][0].values)






cla =1
plt.plot(train_x.values[ind_class[cla][0],:][0].values)
plt.plot(train_x.values[ind_class[cla][1],:][0].values)
plt.plot(train_x.values[ind_class[cla][2],:][0].values)
plt.plot(train_x.values[ind_class[cla][3],:][0].values)




plt.plot(train_x.values[ind_class[0][0],:][0].values)
plt.plot(train_x.values[ind_class[1][4],:][0].values)


plt.plot(train_x.values[ind_class[0][0],:][0].values)
plt.plot(train_x.values[ind_class[1][5],:][0].values)



plt.plot(train_x.values[ind_class[1][0],:][0].values)

plt.plot(train_x.values[ind_class[0][3],:][0].values)
plt.plot(train_x.values[ind_class[1][3],:][0].values)

plt.plot(train_x.values[ind_class[0][4],:][0].values)
plt.plot(train_x.values[ind_class[1][4],:][0].values)
