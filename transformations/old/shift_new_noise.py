from sktime.utils.data_io import load_from_tsfile_to_dataframe
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
            mean = np.mean(ref[range(0,5)])
            std = np.abs(before - after) / 5
            std = np.std(ref[range(0,5)])
            shifted_t[t] = np.random.normal(mean, std, 1)

    else :

        start_t = l-1
        before = ref[start_t - 1]
        after = ref[start_t]
        shifted_t[0:l - 1] = ref[0:l-1]

        for t in range(start_t, start_t + k+1):
            mean = (((start_t + k - t)) * before + (k - (start_t + k - t)) * after) / k
            mean = np.mean(ref[range(len(ref)-5, len(ref))])
            std = np.abs(before - after) / 5
            std = np.std(ref[range(len(ref)-5, len(ref))])

            shifted_t[t] = np.random.normal(mean, std, 1)

    return shifted_t


start = 0
k = 20

start = 1
k = 40


plt.plot(ref)
plt.plot(shift(ref,start,k))






num_neig = 500
neig = []
count=0
inter = np.zeros((num_neig*2,2))
for i in range(0,num_neig):
    k = random.randint(int(len(ref)*0.1),int(len(ref)*0.3))
    for start in [0,1]:
         inter[count,:] = np.array([start,k])
         neig.append(shift(ref, start, k))
         count = count+1



plt.hist(inter[:,1],bins=len(np.arange(int(len(ref)*0.1),int(len(ref)*0.3)+1)))
plt.bar(np.arange(int(len(ref)*0.1),int(len(ref)*0.3)+1),np.unique(inter[:,1],return_counts=True)[1])


distance_matrix = np.zeros((len(neig),train_x.shape[0]))
for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))


neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))

plt.hist(inter[:,1])


np.savetxt("neig_y_shift_1000.txt", neig_y.astype(int))
np.savetxt("inter_shift_1000.txt",inter)


# neig_y = np.loadtxt("neig_y_shift_1000.txt")
# inter = np.loadtxt("inter_shift_1000.txt")



neig_y = np.loadtxt("transformations/weights/CBF_shift_neig_%s.txt" % ind)
inter = np.loadtxt("transformations/weights/CBF_shift_inter_%s.txt" % ind)





variables = np.column_stack((inter, neig_y))
inter2 = variables.astype(float)


# Separate plot by prefix/sufix

x1 = inter2[np.where(inter2[inter2[:,0]==1,2]==test_y[ind].astype(int))[0],1]
x2 = inter2[np.where(inter2[inter2[:,0]==1,2]!=test_y[ind].astype(int))[0],1]


plt.figure(figsize=(8,6))
if len(np.unique(x2))==0:
    plt.hist(x2, range=[10, 40], bins=20, alpha=0.5, label="Other Class")
else:
    plt.hist(x2, bins=len(np.unique(x2)), alpha=0.5, label="Other Class")

if len(np.unique(x1))==0:
    plt.hist(x1, range=[10, 40], bins=20, alpha=0.5, label="Same Class")
else:
    plt.hist(x1, bins=len(np.unique(x1)), alpha=0.5, label="Same Class")

plt.xlabel("Shift level", size=14)
plt.ylabel("Count", size=14)
plt.title("Sufix shift")
plt.legend(loc='upper right')



#smaller
x1 = inter2[np.where(inter2[inter2[:,0]==0,2]==test_y[ind].astype(int))[0],1]
x2 = inter2[np.where(inter2[inter2[:,0]==0,2]!=test_y[ind].astype(int))[0],1]



plt.figure(figsize=(8,6))
if len(np.unique(x2))==0:
    plt.hist(x2, range=[10, 40], bins=20, alpha=0.5, label="Other Class")
else:
    plt.hist(x2, bins=len(np.unique(x2)), alpha=0.5, label="Other Class")

if len(np.unique(x1))==0:
    plt.hist(x1, range=[10, 40], bins=20, alpha=0.5, label="Same Class")
else:
    plt.hist(x1, bins=len(np.unique(x1)), alpha=0.5, label="Same Class")

plt.xlabel("Shift level", size=14)
plt.ylabel("Count", size=14)
plt.title("Prefix shift")
plt.legend(loc='upper right')





presu = 0
inter_presu = inter[inter[:,0]==presu]
neig_presu = neig_y[inter[:,0]==presu]






np.unique(neig_presu, return_counts=True)
same_int = inter_presu[neig_presu==test_y[ind].astype(int)]
other_int = inter_presu[neig_presu!=test_y[ind].astype(int)]





from scipy.spatial.distance import directed_hausdorff
dist_int = np.zeros((same_int.shape[0],other_int.shape[0]))
for i in range(0,same_int.shape[0]):
    for j in range(0,other_int.shape[0]):
        a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
        a = a.reshape(-1, 1)
        b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
        b = b.reshape(-1, 1)
        dist_int[i,j]=max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0])




dist_int = np.zeros((same_int.shape[0],other_int.shape[0]))
for i in range(0,same_int.shape[0]):
    for j in range(0,other_int.shape[0]):
        dist_int[i,j]= np.abs(same_int[i,1]-other_int[j,1])



ran = range(1,30,2)

num = []
for threshold in ran:
    thresh_per_same = []
    for i in range(0,dist_int.shape[0]):
        # min_per_same.append(np.argmin(dist_int[i,:]))
        # perc_per_same.append(np.where(dist_int[i,:]<np.percentile(dist_int,q=2))[0])
        thresh_per_same.append(np.where(dist_int[i,:]<threshold)[0])
    k = np.concatenate(thresh_per_same, axis=0)
    num.append(len(np.unique(k, return_counts=True)[0]))


plt.plot(ran,num)







thresh_per_same = []
for i in range(0,dist_int.shape[0]):
    thresh_per_same.append(np.where(dist_int[i,:]<3)[0])
k = np.concatenate(thresh_per_same, axis=0)

len(np.unique(k))


other_int.shape
same_int.shape


other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
len(other_index)


other_int[other_index,:]

inter2 = inter2[other_index,:]
variables = inter2.copy()

intervals = np.zeros((variables.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
np.sum(intervals, axis=0)




# weigths1 = np.sum(intervals, axis=0)

# w = np.loadtxt("CBF_w_5.txt")
# plot_colormap(ref,w)

plot_colormap(ref,np.sum(intervals, axis=0))






plt.hist(other_int[other_index,1])
#
#
#
# # Separate plot by same/other
#
# #Same class
# x1 = inter2[np.where(inter2[np.where(inter2[:,2]==test_y[ind].astype(int))[0], 0]==0)[0],1]
# x2 = inter2[np.where(inter2[np.where(inter2[:,2]==test_y[ind].astype(int))[0], 0]==1)[0],1]
#
#
# plt.figure(figsize=(8,6))
# plt.hist(x1,  range=[10,40], bins=20, alpha=0.5, label="Prefix")
# plt.hist(x2,  range=[10,40], bins=20, alpha=0.5, label="Sufix")
# plt.xlabel("Shift level", size=14)
# plt.ylabel("Count", size=14)
# plt.title("Same class")
# plt.legend(loc='upper right')
#
#
#
# #Other class
# x1 = inter2[np.where(inter2[np.where(inter2[:,2]!=test_y[ind].astype(int))[0], 0]==0)[0],1]
# x2 = inter2[np.where(inter2[np.where(inter2[:,2]!=test_y[ind].astype(int))[0], 0]==1)[0],1]
#
#
# plt.figure(figsize=(8,6))
# plt.hist(x1,  range=[10,40], bins=20, alpha=0.5, label="Prefix")
# plt.hist(x2,  range=[10,40], bins=20, alpha=0.5, label="Sufix")
# plt.xlabel("Shift level", size=14)
# plt.ylabel("Count", size=14)
# plt.title("Other class")
# plt.legend(loc='upper right')
#
#
#





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


