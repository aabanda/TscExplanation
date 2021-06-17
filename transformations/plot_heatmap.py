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

# ind = int(sys.argv[1])
# cls = str(sys.argv[2])
# db = str(sys.argv[3])
# transformation = str(sys.argv[4])

#warp: gun
ind =52
classifier = "dtw"
db = "GunPoint"
transformation = "warp"
warp_level=0.7


ind =0
classifier = "dtw"
db = "GunPoint"
transformation = "warp"
warp_level=0.7



#Scale: point
ind =1
classifier = "dtw"
db = "GunPoint"
warp_level=1.2
transformation = "scale"


#Noise: arabica
ind =6
classifier = "boss"
db = "Coffee"
warp_level=9
transformation = "noise"





#Shift: robusta
ind =21
classifier = "st"
db = "Coffee"
transformation = "shift"

#
# #Shift: gp
# ind =52
# classifier = "st"
# db = "GunPoint"
# transformation = "shift"



# train_x, train_y = load_from_tsfile_to_dataframe("%s_TRAIN.ts" % db)
# test_x, test_y = load_from_tsfile_to_dataframe("%s_TEST.ts" %db)

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db,db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db,db))




ref = test_x.values[ind, :][0].values
print(test_y[ind])
# plt.plot(ref)
neig_y_total = np.loadtxt("neig/%s/%s/%s/neig_%d.txt" % (db, transformation, classifier, ind))
inter_total = np.loadtxt("neig/%s/%s/dtw/inter_%d.txt" % (db, transformation, ind))
np.unique(neig_y_total, return_counts=True)


inter_total[inter_total[:, 1] == 0, 1] = len(ref)


# for warp_level in np.array([1,3,5,7,9]):

if transformation=="shift":
    neig_y = neig_y_total
    inter = inter_total
else:
    neig_y = neig_y_total[inter_total[:, 2] == warp_level]
    inter = inter_total[inter_total[:, 2] == warp_level, :]
print(ind)
print(np.unique(neig_y, return_counts=True))



# neig_y = neig_y_total
# inter = inter_total
#
# np.unique(neig_y, return_counts=True)




# neig_y = neig_y_total
# inter = inter_total

if transformation!="shift":
    # Same class
    ind_sort = inter[neig_y.astype(int) == test_y[ind].astype(int), 2].argsort()
    inter2 = inter.copy()
    inter2 = inter2[neig_y.astype(int) == test_y[ind].astype(int), :][ind_sort]
    variables = inter2.copy()
    # greater than 1
    long_ind1 = (variables[inter2[:, 2] > 1, 1] - variables[inter2[:, 2] > 1, 0]).argsort()
    largest_indices1 = long_ind1
    # smaller
    long_ind2 = (variables[inter2[:, 2] < 1, 1] - variables[inter2[:, 2] < 1, 0]).argsort()
    largest_indices2 = long_ind2

    same_int = inter2

    # Other class
    ind_sort = inter[neig_y.astype(int) != test_y[ind].astype(int), 2].argsort()
    inter2 = inter.copy()
    inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :][ind_sort]
    variables = inter2.copy()
    # greater than 1
    long_ind1 = (variables[inter2[:, 2] > 1, 1] - variables[inter2[:, 2] > 1, 0]).argsort()
    largest_indices1 = long_ind1[::-1]
    largest_indices1 = long_ind1
    # smaller
    long_ind2 = (variables[inter2[:, 2] < 1, 1] - variables[inter2[:, 2] < 1, 0]).argsort()
    largest_indices2 = long_ind2[::-1]
    largest_indices2 = long_ind2
    # variables = np.delete(variables,2,axis=1)

    # variables = variables[neig_y==test_y[ind],:]
    # variables.shape

    other_int = inter2

    same_int = same_int[:, [0, 1]]
    other_int = other_int[:, [0, 1]]

    same_int = np.delete(same_int, np.where(same_int[:, 0] == same_int[:, 1])[0], 0)
    other_int = np.delete(other_int, np.where(other_int[:, 0] == other_int[:, 1])[0], 0)

elif transformation=="shift":
    # Same class
    inter2 = inter.copy()
    inter2 = inter2[neig_y.astype(int) == test_y[ind].astype(int), :]
    # greater than 1
    # long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
    # largest_indices1 =long_ind1
    # #smaller
    # long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
    # largest_indices2 =long_ind2

    same_int = inter2

    # Other class
    inter2 = inter.copy()
    inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :]
    # greater than 1
    # long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
    # largest_indices1 = long_ind1[::-1]
    # largest_indices1 = long_ind1
    # #smaller
    # long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
    # largest_indices2 = long_ind2[::-1]
    # largest_indices2 = long_ind2
    # #variables = np.delete(variables,2,axis=1)

    # variables = variables[neig_y==test_y[ind],:]
    # variables.shape

    other_int = inter2

    same_int = same_int[:, [0, 1]]
    other_int = other_int[:, [0, 1]]

    same_int = np.delete(same_int, np.where(same_int[:, 0] == same_int[:, 1])[0], 0)
    other_int = np.delete(other_int, np.where(other_int[:, 0] == other_int[:, 1])[0], 0)



dist_int = np.zeros((same_int.shape[0], other_int.shape[0]))
for i in range(0, same_int.shape[0]):
    for j in range(0, other_int.shape[0]):
        a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
        a = a.reshape(-1, 1)
        b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
        b = b.reshape(-1, 1)
        dist_int[i, j] = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])

ran = range(1, 60, 2)

num = []
for threshold in ran:
    thresh_per_same = []
    for i in range(0, dist_int.shape[0]):
        # min_per_same.append(np.argmin(dist_int[i,:]))
        # perc_per_same.append(np.where(dist_int[i,:]<np.percentile(dist_int,q=2))[0])
        thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
    k = np.concatenate(thresh_per_same, axis=0)
    num.append(len(np.unique(k)))

# plt.plot(ran,num)

pendiente = []
for i in range(len(num) - 1):
    pendiente.append(num[i + 1] - num[i])

ind_thre = np.argmin(pendiente)
threshold = np.asarray(ran)[ind_thre]
# plt.plot(pendiente)
#
p_pendiente = []
for i in range(len(pendiente) - 1):
    p_pendiente.append(pendiente[i + 1] - pendiente[i])

# np.column_stack((pendiente[:-1],p_pendiente))
# np.column_stack((ran,num))

p = (np.asarray(p_pendiente) > 0).astype(int)
cambio = []
for pp in range(len(p) - 1):
    cambio.append(p[pp + 1] - p[pp])

if len(np.where(np.asarray(cambio) == 1)[0]) > 0:
    ind_p = np.where(np.asarray(cambio) == 1)[0][0] + 2
    threshold = np.asarray(ran)[ind_p]

    thresh_per_same = []
    for i in range(0, dist_int.shape[0]):
        thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
    k = np.concatenate(thresh_per_same, axis=0)

    len(np.unique(k))
    other_int.shape
    same_int.shape

    other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
    print("len other index")
    len(other_index)

if len(np.where(np.asarray(cambio) == 1)[0]) == 0 or len(other_index) <= 5:
    # middle = (np.where(np.asarray(num) == other_int.shape[0])[0][0] / 2).astype(int)
    middle = (np.where(np.asarray(num) > other_int.shape[0] / 2))[0][0].astype(int)  # percentil 50
    threshold = np.asarray(ran)[middle]

    thresh_per_same = []
    for i in range(0, dist_int.shape[0]):
        thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
    k = np.concatenate(thresh_per_same, axis=0)

    len(np.unique(k))
    other_int.shape
    same_int.shape

    other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
    print("len other index")
    len(other_index)

if len(other_index) == 0:
    # np.savetxt('transformations/weights/threshold/CBF_weights%0.1f_%d.txt' % (warp_level, ind),
    #            np.zeros((len(ref))))
    print("len other index =0")

# Elijo un threshold:

# thresh_per_same = []
# for i in range(0,dist_int.shape[0]):
#     thresh_per_same.append(np.where(dist_int[i,:]<threshold)[0])
# k = np.concatenate(thresh_per_same, axis=0)
#
# len(np.unique(k))
# other_int.shape
# same_int.shape
#
#
#
# other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
# print("len other index")
# len(other_index)

# if len(other_index)<=5 and other_int.shape[0]<=10:
#
#     np.savetxt('transformations/weights/threshold/CBF_weights%0.1f_%d.txt' % (warp_level, ind), np.zeros((len(ref))))

# else:

if transformation!="shift":
    # Other class
    ind_sort = inter[neig_y.astype(int) != test_y[ind].astype(int), 2].argsort()
    inter2 = inter.copy()
    inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :][ind_sort]
    inter2 = inter2[other_index, :]
    variables = inter2.copy()
    # greater than 1
    long_ind1 = (variables[inter2[:, 2] > 1, 1] - variables[inter2[:, 2] > 1, 0]).argsort()
    largest_indices1 = long_ind1[::-1]
    largest_indices1 = long_ind1
    # smaller
    long_ind2 = (variables[inter2[:, 2] < 1, 1] - variables[inter2[:, 2] < 1, 0]).argsort()
    largest_indices2 = long_ind2[::-1]
    largest_indices2 = long_ind2

elif transformation=="shift":

    inter2 = inter.copy()
    inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :]
    inter2 = inter2[other_index, :]
    variables = inter2.copy()
# variables2 = np.concatenate(
#     (variables[inter2[:, 2] > 1, :][largest_indices1, :], variables[inter2[:, 2] < 1, :][largest_indices2, :]))

intervals = np.zeros((variables.shape[0], len(ref)))
for i in range(0, intervals.shape[0]):
    intervals[i, range(variables[i, 0].astype(int), variables[i, 1].astype(int))] = 1
np.sum(intervals, axis=0)

print("Interval lengths")
print(np.mean(np.sum(intervals, axis=1)))
print(np.std(np.sum(intervals, axis=1)))





def plot_colormap(ref, weights):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x = range(0,len(ref)-1)
    y = ref[1:]
    # dydx = np.abs(clf.coef_)[0]
    #dydx = np.sum(intervals, axis=0)/len(other_index)
    dydx = weights
    dydx = dydx[1:]
    dydx = (dydx- np.min(dydx))/(np.max(dydx)-np.min(dydx))


    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)


    fig, axs = plt.subplots(figsize=(6,4))

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)


    cb = fig.colorbar(line, ax=axs)
    cb.ax.tick_params(labelsize=15)
    axs.set_xlim(0, len(x))
    if db=="Coffee":
        axs.set_ylim(-2.5,2.5)
    if db=="GunPoint":
        axs.set_ylim(-1.5, 2.5)
    plt.show()




plot_colormap(ref,np.sum(intervals, axis=0))
#plt.title("Gun",fontsize=15)
#plt.title("Point",fontsize=15)
#plt.title("Arabica",fontsize=15)
plt.title("Robusta",fontsize=15)
plt.xlabel("t",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
    # weigths1 = np.sum(intervals, axis=0)
    # np.savetxt('transformations/weights/threshold/CBF_weights%0.1f_%d.txt' % (warp_level,ind),np.sum(intervals, axis=0) )

