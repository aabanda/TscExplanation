from sktime.utils.data_io import load_from_tsfile_to_dataframe
import numpy as np
import matplotlib.pyplot as plt


#WARP 1

#Same class
db = "GesturePebbleZ1"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))


ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])

plt.plot(train_x.values[ind_class[5][0],:][0].values)
plt.plot(train_x.values[ind_class[5][2],:][0].values)





#Other class
db = "ECGFiveDays"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])

plt.plot(train_x.values[ind_class[0][0],:][0].values)
plt.plot(train_x.values[ind_class[1][4],:][0].values)


db = "GunPoint"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])



plt.plot(train_x.values[ind_class[0][18],:][0].values)
plt.plot(train_x.values[ind_class[1][1],:][0].values)






#SCALE
db = "GunPoint"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))
print(np.unique(train_y))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])

plt.plot(train_x.values[ind_class[0][0],:][0].values)
plt.plot(train_x.values[ind_class[1][1],:][0].values)






#Noise

db = "ECG200"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))
print(np.unique(train_y))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])


plt.plot(train_x.values[ind_class[0][6],:][0].values)
plt.plot(train_x.values[ind_class[1][2],:][0].values)



#SHIFT


db = "AllGestureWiimoteY"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))
print(np.unique(train_y))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])


plt.plot(train_x.values[ind_class[0][0],:][0].values)
plt.plot(train_x.values[ind_class[1][1],:][0].values)


#
#
#
# db = "ArrowHead"
# #
#
# db = "Meat"
# db = "Car"
# db = "BeetleFly"
# db = "Coffee"




ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])
print(np.unique(train_y))





#Separadas
per_class =10

fig, axs = plt.subplots(len(np.unique(train_y)),per_class, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()
i = 0
for c in range(len(np.unique(train_y))):
    for j in range(per_class):
        ind = ind_class[c][j+10]
        axs[i].plot(train_x.values[ind,:][0].values)
        i = i+1






for ind_search in range(30):


    k=train_x.values[ind_class[1][ind_search],:][0].values
    #plt.plot(k)


    min_per_class = []
    for c in range(10):
        which_class = c
        dis = np.zeros((len(ind_class[which_class]), 500))
        for t in range(len(ind_class[which_class])):
            for i in range(len(train_x.values[ind_class[which_class][t],:][0].values)-len(k)):
                dis[t,i] = np.linalg.norm(k-train_x.values[ind_class[which_class][t],:][0].values[i:i+len(k)])

        dis[dis==0]= None
        min_per_class.append(np.nanmin(dis))
    print(np.nanargmin(min_per_class))



#una opcion= 3,0 whichclass 4
#una opcion= 4,0 whichclass 6

k=train_x.values[ind_class[3][0],:][0].values
k=train_x.values[ind_class[4][0],:][0].values


which_class = 6
dis = np.zeros((len(ind_class[which_class]), 500))
for t in range(len(ind_class[which_class])):
    for i in range(len(train_x.values[ind_class[which_class][t], :][0].values) - len(k)):
        dis[t, i] = np.linalg.norm(k - train_x.values[ind_class[which_class][t], :][0].values[i:i + len(k)])

dis[dis == 0] = None
print(np.nanmin(dis))

ind_min = np.unravel_index(np.nanargmin(dis), dis.shape)

plt.plot(train_x.values[ind_class[which_class][ind_min[0]],:][0].values)
plt.plot(range(ind_min[1],ind_min[1]+len(k)),k)




#Solapadas
per_class =3
fig, axs = plt.subplots(len(np.unique(train_y)),1, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for c in range(len(np.unique(train_y))):
    for j in range(per_class):
        ind = ind_class[c][j]
        axs[c].plot(train_x.values[ind,:][0].values)



