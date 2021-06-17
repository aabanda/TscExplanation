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

plt.figure()
plt.plot(train_x.values[ind_class[5][0],:][0].values)
plt.plot(train_x.values[ind_class[5][2],:][0].values)





#Other class
db = "ECGFiveDays"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])

plt.figure()
plt.plot(train_x.values[ind_class[0][0],:][0].values)
plt.plot(train_x.values[ind_class[1][4],:][0].values)


db = "GunPoint"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])

plt.figure()
plt.plot(train_x.values[ind_class[0][18],:][0].values, label="Gun class",linewidth=3)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=15)
plt.tight_layout()


plt.figure()
plt.plot(train_x.values[ind_class[1][1],:][0].values, label="Point class",linewidth=3, color="C1")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=15)
plt.tight_layout()



db = "InlineSkate"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])

plt.figure()
plt.plot(train_x.values[ind_class[4][3], :][0].values)
plt.figure()
plt.plot(train_x.values[ind_class[6][3], :][0].values, color="C1")
# dkjh




plt.figure(figsize=(8,6))
plt.plot(train_x.values[ind_class[4][3],:][0].values, label="Class 5",linewidth=3)
plt.plot(train_x.values[ind_class[6][3],:][0].values, label="Class 7",linewidth=3)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()





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


plt.figure(figsize=(8,6))
plt.plot(train_x.values[ind_class[0][0],:][0].values, label="Gun class",linewidth=3)
plt.plot(train_x.values[ind_class[1][1],:][0].values, label="Point class",linewidth=3)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()


db = "Adiac"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))
print(np.unique(train_y))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])

plt.plot(train_x.values[ind_class[0][0],:][0].values)
plt.ylim(-1.5,2)
plt.plot(train_x.values[ind_class[1][0],:][0].values,color="C1")
plt.ylim(-1.5,2)

plt.figure(figsize=(8,6))
plt.plot(train_x.values[ind_class[0][0],:][0].values, label="Class 1",linewidth=3,)
plt.plot(train_x.values[ind_class[1][0],:][0].values, label="Class 2",linewidth=3,)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()


#
# plt.plot(train_x.values[ind_class[0][0],:][0].values)
# plt.plot(train_x.values[ind_class[1][0],:][0].values)


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

# plt.figure()
# plt.plot(train_x.values[ind_class[0][9],:][0].values)
# plt.plot(train_x.values[ind_class[1][0],:][0].values)

plt.figure(figsize=(8,6))
plt.plot(train_x.values[ind_class[0][6],:][0].values, label="Normal",linewidth=3)
plt.plot(train_x.values[ind_class[1][2],:][0].values, label="Myocardial",linewidth=3)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()



#SHIFT


db = "AllGestureWiimoteY"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db, db))
print(np.unique(train_y))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])


k=train_x.values[ind_class[4][0],:][0].values
which_class =6
dis = np.zeros((len(ind_class[which_class]), 500))
for t in range(len(ind_class[which_class])):
    for i in range(len(train_x.values[ind_class[which_class][t], :][0].values) - len(k)):
        dis[t, i] = np.linalg.norm(k - train_x.values[ind_class[which_class][t], :][0].values[i:i + len(k)])

dis[dis == 0] = None
print(np.nanmin(dis))
ind_min= np.unravel_index(np.nanargmin(dis), dis.shape)


plt.figure(figsize=(8,6))
plt.plot(train_x.values[ind_class[which_class][ind_min[0]],:][0].values, label="Class 7",linewidth=3)
#plt.plot(range(ind_min[1],ind_min[1]+len(k)),k, label="Class 5",linewidth=3)
plt.plot(k, label="Class 5",linewidth=3)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.ylim(-1,5)
#plt.ylim(-2,4)
plt.tight_layout()







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

#dkjh


#Solapadas
per_class =3
fig, axs = plt.subplots(len(np.unique(train_y)),1, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for c in range(len(np.unique(train_y))):
    for j in range(per_class):
        ind = ind_class[c][j]
        axs[c].plot(train_x.values[ind,:][0].values)



#kjsadkjsasd