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


#
# inter = np.loadtxt('transformations/weights/CBF_inter_54.txt')
# neig_y = np.loadtxt('transformations/weights/CBF_neig_54.txt')

#
#
# inter_seg = inter.copy()
# neig_y_seg = neig_y.copy()
#
# inter = inter_seg
# neig_y = neig_y_seg


# neig_y = neig_y[inter[:,2]==0.7]
# inter = inter[inter[:,2]==0.7,:]
#
#
#
# #Same class
# ind_sort = inter[neig_y.astype(int)== test_y[ind].astype(int), 2].argsort()
# inter2 = inter.copy()
# inter2 = inter2[neig_y.astype(int) == test_y[ind].astype(int), :][ind_sort]
# variables = inter2.copy()
# #greater than 1
# long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
# largest_indices1 =long_ind1
# #smaller
# long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
# largest_indices2 =long_ind2
#
# same_int = inter2
#
#
#
#
# #Other class
# ind_sort = inter[neig_y.astype(int)!= test_y[ind].astype(int), 2].argsort()
# inter2 = inter.copy()
# inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :][ind_sort]
# variables = inter2.copy()
# #greater than 1
# long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
# largest_indices1 = long_ind1[::-1]
# largest_indices1 = long_ind1
# #smaller
# long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
# largest_indices2 = long_ind2[::-1]
# largest_indices2 = long_ind2
# #variables = np.delete(variables,2,axis=1)
#
# # variables = variables[neig_y==test_y[ind],:]
# # variables.shape
#
#
# other_int = inter2
#
# same_int = same_int[:,[0,1]]
# other_int = other_int[:,[0,1]]
#
#
#
# same_int = np.delete(same_int,np.where(same_int[:,0]==same_int[:,1])[0],0 )
#
# other_int = np.delete(other_int,np.where(other_int[:,0]==other_int[:,1])[0],0 )
#
#
#
# np.unique(neig_y,return_counts=True)
#
# dist_int = np.zeros((same_int.shape[0],other_int.shape[0]))
# for i in range(0,same_int.shape[0]):
#     for j in range(0,other_int.shape[0]):
#         a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
#         a = a.reshape(-1, 1)
#         b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
#         b = b.reshape(-1, 1)
#         dist_int[i,j]=max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0])
#
#
#
#
# ran = range(1,30,2)
#
# num = []
# for threshold in ran:
#     thresh_per_same = []
#     for i in range(0,dist_int.shape[0]):
#         # min_per_same.append(np.argmin(dist_int[i,:]))
#         # perc_per_same.append(np.where(dist_int[i,:]<np.percentile(dist_int,q=2))[0])
#         thresh_per_same.append(np.where(dist_int[i,:]<threshold)[0])
#     k = np.concatenate(thresh_per_same, axis=0)
#     num.append(len(np.unique(k, return_counts=True)[0]))
#
#
# plt.plot(ran,num)
#
#
#
# #Elijo un threshold:
#
# thresh_per_same = []
# for i in range(0,dist_int.shape[0]):
#     thresh_per_same.append(np.where(dist_int[i,:]<5)[0])
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
#
#
#
#
#
# #Other class
# ind_sort = inter[neig_y.astype(int)!= test_y[ind].astype(int), 2].argsort()
# inter2 = inter.copy()
# inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :][ind_sort]
# inter2 = inter2[other_index,:]
# variables = inter2.copy()
# #greater than 1
# long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
# largest_indices1 = long_ind1[::-1]
# largest_indices1 = long_ind1
# #smaller
# long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
# largest_indices2 = long_ind2[::-1]
# largest_indices2 = long_ind2
#
#
#
#
#
#
#
#
# variables2 = np.concatenate((variables[inter2[:,2]>1,:][largest_indices1,:],variables[inter2[:,2]<1,:][largest_indices2,:] ))
#
#
# intervals = np.zeros((variables.shape[0],len(ref)))
# for i in range(0,intervals.shape[0]):
#     intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
# np.sum(intervals, axis=0)
#
#
# print("Interval lengths")
# print(np.mean(np.sum(intervals, axis=1)))
# print(np.std(np.sum(intervals, axis=1)))
#
# # weigths1 = np.sum(intervals, axis=0)
# np.savetxt('CBF_weights08_%d.txt' % ind,np.sum(intervals, axis=0) )
