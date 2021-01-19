from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np
import matplotlib.pyplot as plt
import random
from sktime.distances.elastic import dtw_distance




train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

#
# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")

#CBF
# ind =2 #class 1
# ind = 0 #Class 2
ind = 5 #Class 3

#ArrowHead
# ind=0
# ind=100
# ind=160

ref = test_x.values[ind,:][0].values
# plt.plot(ref)
test_y[ind]




def shift(ref, start, k):
    l = len(ref)
    shifted_t = np.zeros(l + k)

    if start ==0:

        start_t = 1
        before = ref[start_t - 1]
        after = ref[start_t]
        shifted_t[0] = ref[0]
        shifted_t[start_t + k:] = ref[1:l]


        for t in range(start_t, start_t + k+1):
            mean = (((start_t + k - t)) * before + (k - (start_t + k - t)) * after) / k
            std = np.abs(before - after) / 5
            shifted_t[t] = np.random.normal(mean, std, 1)

    else :

        start_t = l-1
        before = ref[start_t - 1]
        after = ref[start_t]
        shifted_t[0:l - 1] = ref[0:l-1]

        for t in range(start_t, start_t + k+1):
            mean = (((start_t + k - t)) * before + (k - (start_t + k - t)) * after) / k
            std = np.abs(before - after) / 5
            shifted_t[t] = np.random.normal(mean, std, 1)

    return shifted_t


start = 0
k = 29

start = 1
k = 40


# plt.plot(ref)
# plt.plot(shift(ref,start,k))
#





num_neig = 500
neig = []
inter = np.zeros((num_neig,2))
for i in range(0,num_neig):
     start = random.randint(0,1)
     k = random.randint(int(len(ref)*0.1),int(len(ref)*0.3))
     inter[i,:] = np.array([start,k])
     neig.append(shift(ref, start, k))



distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))


neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))










variables = np.column_stack((inter, neig_y))
inter2 = variables.astype(float)
#
#
# # Separate plot by prefix/sufix
#
# #greater than 1
# x1 = inter2[np.where(inter2[inter2[:,0]>=1,2]==test_y[ind].astype(int))[0],1]
# x2 = inter2[np.where(inter2[inter2[:,0]>=1,2]!=test_y[ind].astype(int))[0],1]
#
#
# plt.figure(figsize=(8,6))
# plt.hist(x1,  range=[10,40], bins=20, alpha=0.5, label="Same Class")
# plt.hist(x2,  range=[10,40], bins=20, alpha=0.5, label="Other Class")
# plt.xlabel("Shift level", size=14)
# plt.ylabel("Count", size=14)
# plt.title("Sufix shift")
# plt.legend(loc='upper right')
#
#
#
# #smaller
# #greater than 1
# x1 = inter2[np.where(inter2[inter2[:,0]<1,2]==test_y[ind].astype(int))[0],1]
# x2 = inter2[np.where(inter2[inter2[:,0]<1,2]!=test_y[ind].astype(int))[0],1]
#
# plt.figure(figsize=(8,6))
# plt.hist(x1, range=[10,40], bins=20, alpha=0.5, label="Same Class")
# plt.hist(x2, range=[10,40], bins=20, alpha=0.5, label="Other Class")
# plt.xlabel("Shift level", size=14)
# plt.ylabel("Count", size=14)
# plt.title("Prefix shift")
# plt.legend(loc='upper right')
#








# Separate plot by same/other

#Same class
x1 = inter2[np.where(inter2[np.where(inter2[:,2]==test_y[ind].astype(int))[0], 0]==0)[0],1]
x2 = inter2[np.where(inter2[np.where(inter2[:,2]==test_y[ind].astype(int))[0], 0]==1)[0],1]


plt.figure(figsize=(8,6))
plt.hist(x1,  range=[10,40], bins=20, alpha=0.5, label="Prefix")
plt.hist(x2,  range=[10,40], bins=20, alpha=0.5, label="Sufix")
plt.xlabel("Shift level", size=14)
plt.ylabel("Count", size=14)
plt.title("Same class")
plt.legend(loc='upper right')



#Other class
x1 = inter2[np.where(inter2[np.where(inter2[:,2]!=test_y[ind].astype(int))[0], 0]==0)[0],1]
x2 = inter2[np.where(inter2[np.where(inter2[:,2]!=test_y[ind].astype(int))[0], 0]==1)[0],1]


plt.figure(figsize=(8,6))
plt.hist(x1,  range=[10,40], bins=20, alpha=0.5, label="Prefix")
plt.hist(x2,  range=[10,40], bins=20, alpha=0.5, label="Sufix")
plt.xlabel("Shift level", size=14)
plt.ylabel("Count", size=14)
plt.title("Other class")
plt.legend(loc='upper right')








#other class
#greater than 1
long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
largest_indices1 = long_ind1[::-1]
largest_indices1 = long_ind1
#smaller
long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
largest_indices2 = long_ind2[::-1]
largest_indices2 = long_ind2



















import matplotlib as mpl
cmap = plt.cm.get_cmap("jet")
norm = mpl.colors.SymLogNorm(2, vmin=inter[neig_y==test_y[ind],:][:,1].min(), vmax=inter[neig_y==test_y[ind],:][:,1].max())
norm = plt.Normalize(inter[neig_y==test_y[ind],:][:,1].min(), inter[neig_y==test_y[ind],:][:,1].max())


sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])








intervals = np.zeros((variables.shape[0],variables[:,1].max().astype(int)))
intervals[:,:] = np.nan
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = i
    plt.plot(range(0, intervals.shape[1]), intervals[i, :], color=cmap(norm(inter[neig_y==test_y[ind],:][i,1].astype(int))), label=inter[neig_y==test_y[ind],:][i,1].astype(int))
#plt.legend(loc=3)

cbar = plt.colorbar(sm, ticks=inter[neig_y==test_y[ind],:][:,1], format=mpl.ticker.ScalarFormatter(),
                     fraction=0.1, pad=0)








intervals = np.zeros((variables.shape[0],variables[:,1].max().astype(int)))
intervals[:,:] = np.nan
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
    #plt.plot(range(0, intervals.shape[1]), intervals[i, :])


import matplotlib as mpl
cmap = plt.cm.get_cmap("jet")
# norm = mpl.colors.SymLogNorm(2, vmin=inter[neig_y==test_y[ind],:][:,1].min(), vmax=inter[neig_y==test_y[ind],:][:,1].max())
# norm = plt.Normalize(inter[neig_y==test_y[ind],:][:,1].min(), inter[neig_y==test_y[ind],:][:,1].max())


sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])


ind_long= (variables[:,1]-variables[:,0]).argsort()
#ind_long= variables[:,0].argsort()
intervals = intervals[ind_long,:]
long_sort = (variables[:,1]-variables[:,0])[ind_long]
norm = plt.Normalize(long_sort.min(),long_sort.max())

for i in range(0,intervals.shape[0]):
    intervals[i,np.where(intervals[i,:]==1)[0]] = i
    plt.plot(range(0, intervals.shape[1]), intervals[i, :],color=cmap(norm(long_sort.astype(int)[i])))

cbar = plt.colorbar(sm, ticks=long_sort, format=mpl.ticker.ScalarFormatter(),
                     fraction=0.1, pad=0)







intervals = np.zeros((variables.shape[0], variables[:,1].max().astype(int)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
intervals = intervals[:,range(0,len(ref))]
np.sum(intervals, axis=0)




from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x = range(0,len(ref))
y = ref
# dydx = np.abs(clf.coef_)[0]
dydx = np.sum(intervals, axis=0)
# dydx = np.zeros((len(clf.coef_[0])))
# dydx[np.where(clf.coef_[0]>0)[0]] = clf.coef_[0][np.where(clf.coef_[0]>0)[0]]


# <dydx = (dydx - dydx.min()) / (dydx.max() - dydx.min())
# dydx = dydx>

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


fig, axs = plt.subplots()

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='jet', norm=norm)
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs.add_collection(lc)


fig.colorbar(line, ax=axs)

axs.set_xlim(0, len(x))
axs.set_ylim(-2.5, 2.5)
axs.set_ylim(-2, 4)
plt.show()


