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
db = "Adiac"
db = "AllGestureWiimoteX"
db = "AllGestureWiimoteY"
db = "AllGestureWiimoteZ"
db = "CMJ"
db = "Coffee"





ind_ref=0
#db = "Adiac"
if db!="CMJ":
    train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
    test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))
else:
    train_x= np.loadtxt("../datasets/Univariate_ts/%s/%s_TRAIN.txt" % (db, db), delimiter=",")
    test_x= np.loadtxt("../datasets/Univariate_ts/%s/%s_TEST.txt" % (db, db), delimiter=",")
    


ref = test_x.values[ind_ref,:][0]
test_y[ind_ref]

#plt.plot(ref)
print(np.unique(train_y))


ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])

# ind=0
# s= train_x.values[ind,:][0].values
# len(s)
# plt.plot(s)

per_class =5
fig, axs = plt.subplots(len(np.unique(train_y)),1, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for c in range(len(np.unique(train_y))):
    for j in range(per_class):
        ind = ind_class[c][j]
        axs[c].plot(train_x.values[ind,:][0].values)




plt.figure()
plt.plot(train_x.values[ind_class[0][0],:][0].values)
plt.plot(train_x.values[ind_class[1][1],:][0].values)

plt.plot(train_x.values[ind_class[0][1],:][0].values)
plt.plot(train_x.values[ind_class[1][1],:][0].values)




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




#una opcion= 3,0 whichclass 4
#una opcion= 4,0 whichclass 6

k=train_x.values[ind_class[3][0],:][0].values

k=train_x.values[ind_class[8][0],:][0].values
plt.plot(k)

which_class =7
dis = np.zeros((len(ind_class[which_class]), 500))
for t in range(len(ind_class[which_class])):
    for i in range(len(train_x.values[ind_class[which_class][t], :][0].values) - len(k)):
        dis[t, i] = np.linalg.norm(k - train_x.values[ind_class[which_class][t], :][0].values[i:i + len(k)])

dis[dis == 0] = None
print(np.nanmin(dis))

ind_min = np.unravel_index(np.nanargmin(dis), dis.shape)
plt.plot(train_x.values[ind_class[which_class][ind_min[0]],:][0].values)
plt.plot(range(ind_min[1],ind_min[1]+len(k)),k)


plt.plot(train_x.values[ind_class[4][0],:][0].values)
plt.plot(train_x.values[ind_class[4][1],:][0].values)

plt.figure()
plt.plot(train_x.values[ind_class[6][0],:][0].values)
plt.plot(train_x.values[ind_class[6][1],:][0].values)
#dkjh


#plt.plot(train_x.values[3,:][0].values)
# #axs[i].set_title(str(250+i))






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
