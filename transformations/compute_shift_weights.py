from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from sklearn.metrics import confusion_matrix
from scipy.stats import betaprime
from scipy.spatial.distance import directed_hausdorff
import sys
# #Separado por wapr level
# inter = np.loadtxt("inter_warp_3000.txt")
# neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")

ind = int(sys.argv[1])

ind=5

# train_x, train_y = load_from_tsfile_to_dataframe("CBF_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("CBF_TEST.ts")

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

ref = test_x.values[ind,:][0].values
print(test_y[ind])




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



distance_matrix = np.zeros((len(neig),train_x.shape[0]))
for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))


neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print("importance")
print(np.unique(neig_y,return_counts=True))

np.savetxt("CBF_shift_neig_%d.txt" % ind, neig_y.astype(int))
np.savetxt("CBF_shift_inter_%d.txt" % ind, inter)





















ind=1

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")

ref = test_x.values[ind,:][0].values
print(test_y[ind])
plt.plot(ref)


inter = np.loadtxt('transformations/weights/GP_shift_inter_%s.txt' % ind)
neig_y = np.loadtxt('transformations/weights/GP_shift_neig_%s.txt' % ind)

np.unique(neig_y,return_counts=True)


same_int = inter[neig_y==test_y[ind].astype(int)]
other_int = inter[neig_y!=test_y[ind].astype(int)]

same_int.shape
other_int.shape

from scipy.spatial.distance import directed_hausdorff
dist_int = np.zeros((same_int.shape[0], other_int.shape[0]))
for i in range(0, same_int.shape[0]):
    for j in range(0, other_int.shape[0]):
        a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
        a = a.reshape(-1, 1)
        b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
        b = b.reshape(-1, 1)
        dist_int[i, j] = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])



#por columna
ran = range(1,50,2)
num = []
for threshold in ran:
    thresh_per_same = []
    for i in range(0,dist_int.shape[1]):
        thresh_per_same.append(np.where(dist_int[:,i]<threshold)[0])
    k = np.concatenate(thresh_per_same, axis=0)
    num.append(len(np.unique(k, return_counts=True)[0]))



plt.plot(ran,num)



if np.all(np.asarray(num) == num[0]):

    other_ind = np.arange(same_int.shape[0])

else:


    #Elijo un threshold:
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

    p = ( np.asarray(p_pendiente)>0).astype(int)
    cambio = []
    for pp in range(len(p)-1):
        cambio.append(p[pp+1]-p[pp])

    if len(np.where(np.asarray(cambio)==1)[0])>0:
        ind_p= np.where(np.asarray(cambio)==1)[0][0]+2
        threshold = np.asarray(ran)[ind_p]

        thresh_per_same = []
        for i in range(0, dist_int.shape[0]):
            thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
        k = np.concatenate(thresh_per_same, axis=0)

        len(np.unique(k))
        other_int.shape
        same_int.shape

        other_index = np.setdiff1d(np.arange(same_int.shape[0]), np.unique(k))


    if len(np.where(np.asarray(cambio)==1)[0])==0 or len(same_int)<=5:
        #middle = (np.where(np.asarray(num) == other_int.shape[0])[0][0] / 2).astype(int)
        middle = (np.where(np.asarray(num) >other_int.shape[0]/ 2))[0][0] .astype(int) #percentil 50
        threshold = np.asarray(ran)[middle]

        thresh_per_same = []
        for i in range(0, dist_int.shape[0]):
            thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
        k = np.concatenate(thresh_per_same, axis=0)

        len(np.unique(k))
        other_int.shape
        same_int.shape

        other_index = np.setdiff1d(np.arange(same_int.shape[0]), np.unique(k))


print("len other index")
print(len(other_index))


#variables = other_int[other_index,:]
variables = same_int[other_index,:]




intervals = np.zeros((variables.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
np.sum(intervals, axis=0)


def plot_colormap(ref, weights):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x = range(0,len(ref)-1)
    y = ref[1:]
    # dydx = np.abs(clf.coef_)[0]
    #dydx = np.sum(intervals, axis=0)/len(other_index)
    dydx = weights
    dydx = dydx[1:]


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

    axs.set_xlim(0-5, len(x)+5)
    axs.set_ylim(-2.5, 2.5)
    axs.set_ylim(-1.5,2.5)
    plt.show()


plot_colormap(ref,np.sum(intervals, axis=0))

