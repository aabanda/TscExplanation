from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from sklearn.metrics import confusion_matrix
from scipy.stats import betaprime
from scipy.spatial.distance import directed_hausdorff

# #Separado por wapr level
# inter = np.loadtxt("inter_warp_3000.txt")
# neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")


# ind = int(sys.argv[1])
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

zerps= []
for ind in range(0,101):
    # ind=1

    ref = test_x.values[ind,:][0].values
    print(test_y[ind])

    neig_y_total= np.loadtxt("transformations/weights/CBF_neig_%d.txt" % ind)
    inter_total =np.loadtxt("transformations/weights/CBF_inter_%d.txt" % ind)

    inter_total[inter_total[:,1]==0,1]=len(ref)

    if len(np.unique(neig_y_total))==1:
        zerps.append(1)


    for warp_level in np.array([0.7,0.8,0.9,1.1,1.2,1.3]):

        neig_y = neig_y_total[inter_total[:,2]==warp_level]
        inter = inter_total[inter_total[:,2]==warp_level,:]



        if len(np.unique(neig_y))==1 or np.unique(neig_y, return_counts=True)[1][np.where(np.unique(neig_y)==test_y[ind].astype(int))[0][0]]>=(0.99*len(neig_y)):

            np.savetxt('transformations/weights/threshold/CBF_weights%0.1f_%d.txt' % (warp_level,ind), np.zeros((len(ref))))

        else:


            #Same class
            ind_sort = inter[neig_y.astype(int)== test_y[ind].astype(int), 2].argsort()
            inter2 = inter.copy()
            inter2 = inter2[neig_y.astype(int) == test_y[ind].astype(int), :][ind_sort]
            variables = inter2.copy()
            #greater than 1
            long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
            largest_indices1 =long_ind1
            #smaller
            long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
            largest_indices2 =long_ind2

            same_int = inter2




            #Other class
            ind_sort = inter[neig_y.astype(int)!= test_y[ind].astype(int), 2].argsort()
            inter2 = inter.copy()
            inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :][ind_sort]
            variables = inter2.copy()
            #greater than 1
            long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
            largest_indices1 = long_ind1[::-1]
            largest_indices1 = long_ind1
            #smaller
            long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
            largest_indices2 = long_ind2[::-1]
            largest_indices2 = long_ind2
            #variables = np.delete(variables,2,axis=1)

            # variables = variables[neig_y==test_y[ind],:]
            # variables.shape


            other_int = inter2

            same_int = same_int[:,[0,1]]
            other_int = other_int[:,[0,1]]



            same_int = np.delete(same_int,np.where(same_int[:,0]==same_int[:,1])[0],0 )
            other_int = np.delete(other_int,np.where(other_int[:,0]==other_int[:,1])[0],0 )


            intersection_matrix = np.zeros((same_int.shape[0], other_int.shape[0]))
            for i in range(0,same_int.shape[0]):
                for j in range(0,other_int.shape[0]):
                    a = set(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
                    b = a.intersection(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
                    intersection_matrix[i,j] = len(b)/ len(a)


            ran = range(1,60,2)
            ran = np.array([0.05,0.1,0.2,0.4,0.5,0.7, 0.9])
            ran = np.array([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05])
            ran = np.array([0.9,0.5,0.01])
            ran = np.array([0.01,0.5,1])

            num = []
            for threshold in ran:
                thresh_per_same = []
                for i in range(0,intersection_matrix.shape[0]):
                    # min_per_same.append(np.argmin(dist_int[i,:]))
                    # perc_per_same.append(np.where(dist_int[i,:]<np.percentile(dist_int,q=2))[0])
                    thresh_per_same.append(np.where(intersection_matrix[i,:]>0.99)[0])
                k = np.concatenate(thresh_per_same, axis=0)
                num.append(len(np.unique(k)))


            plt.plot(num)

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

                other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
                print("len other index")
                len(other_index)

            if len(np.where(np.asarray(cambio)==1)[0])==0 or len(other_index)<=5:
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

                other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
                print("len other index")
                len(other_index)

            if len(other_index)==0:
                np.savetxt('transformations/weights/threshold/CBF_weights%0.1f_%d.txt' % (warp_level, ind),
                           np.zeros((len(ref))))
            #Elijo un threshold:

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


           #Other class
            ind_sort = inter[neig_y.astype(int)!= test_y[ind].astype(int), 2].argsort()
            inter2 = inter.copy()
            inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :][ind_sort]
            inter2 = inter2[other_index,:]
            variables = inter2.copy()
            #greater than 1
            long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
            largest_indices1 = long_ind1[::-1]
            largest_indices1 = long_ind1
            #smaller
            long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
            largest_indices2 = long_ind2[::-1]
            largest_indices2 = long_ind2








            variables2 = np.concatenate((variables[inter2[:,2]>1,:][largest_indices1,:],variables[inter2[:,2]<1,:][largest_indices2,:] ))


            intervals = np.zeros((variables.shape[0],len(ref)))
            for i in range(0,intervals.shape[0]):
                intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
            np.sum(intervals, axis=0)


            print("Interval lengths")
            print(np.mean(np.sum(intervals, axis=1)))
            print(np.std(np.sum(intervals, axis=1)))

            # weigths1 = np.sum(intervals, axis=0)
            np.savetxt('transformations/weights/threshold/CBF_weights%0.1f_%d.txt' % (warp_level,ind),np.sum(intervals, axis=0) )




zerps = []
for ind in range(31,101):
    for warp_level in np.array([0.7,0.8,0.9,1.1,1.2,1.3]):
        w= np.loadtxt('transformations/weights/threshold/CBF_weights%0.1f_%d.txt' % (warp_level, ind))
        if np.sum(w)==0:
            zerps.append(1)

np.sum(zerps)