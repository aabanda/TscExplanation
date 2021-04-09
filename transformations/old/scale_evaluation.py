#from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from transformations.old.warp_function import warp
from scipy.stats import betaprime


def intersection(intervals):
    start, end = intervals.pop()
    while intervals:
        start_temp, end_temp = intervals.pop()
        start = max(start, start_temp)
        end = min(end, end_temp)
    return [start, end]


def unwrapcircle(z):
    u = np.round(random.uniform(-z, 1), decimals=2)
    return intersection([[0, 1], [u, u + z]])

# ind = int(sys.argv[1])
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

ind=5
scale_level= 1.2

class_true = test_y[range(0,101)]

p1 = []
p2 = []
p3 = []
p11 = []
p22 = []
p33 = []
zer = []

for repetition in range(0, 10):

    class_type1 = np.zeros((101, 3))
    class_type2 = np.zeros((101, 3))

    zeros_warp = []
    for ind in range(1,101):

        w = np.loadtxt('transformations/weights/CBF_scale_weights%0.1f_%d.txt' % (scale_level, ind))

        if np.sum(w)==0 :
            zeros_warp.append(ind)

        else:
            print("index")
            print(ind)
            ref = test_x.values[ind, :][0].values

            count1 = 0

            for type1 in np.array([75,50,25]):
                print("deberia cambiar")
                sample_ind = np.arange(len(ref))[np.where(w >= np.percentile(w, q=type1))[0]]
                saltos = []
                for i in range(len(sample_ind) - 1):
                    saltos.append(sample_ind[i + 1] - sample_ind[i])

                num_saltos = np.sum(np.asarray(saltos) > 1)

                if num_saltos  >=  1:

                    ind_saltos = np.where(np.asarray(saltos) > 1)[0]

                    intervals_we = [np.arange(sample_ind[0], sample_ind[ind_saltos[0]] + 1)]

                    for j in range(1, len(np.where(np.asarray(saltos) > 1)[0])):
                        intervals_we.append(np.arange(sample_ind[ind_saltos[j - 1] + 1], sample_ind[ind_saltos[j]] + 1))

                    intervals_we.append(np.arange(sample_ind[ind_saltos[-1] + 1], sample_ind[-1] + 1))

                    len(intervals_we)
                    leng = []
                    for l in range(len(intervals_we)):
                        leng.append(len(intervals_we[l]))

                    selected_set = intervals_we[np.argmax(leng)]
                else:
                    selected_set = sample_ind

                a = 8
                p = 0.5
                b = a * (1 - p) / p



                start_end_sample = unwrapcircle(betaprime.rvs(a, b, size=1))

                if isinstance(start_end_sample[0], np.ndarray):
                    start_end_sample[0] = start_end_sample[0][0]

                if isinstance(start_end_sample[1], np.ndarray):
                    start_end_sample[1] = start_end_sample[1][0]

                while np.abs(start_end_sample[0]-start_end_sample[1])<0.3:

                    a = 8
                    p = 0.5
                    b = a * (1 - p) / p

                    start_end_sample = unwrapcircle(betaprime.rvs(a, b, size=1))

                    if isinstance(start_end_sample[0], np.ndarray):
                        start_end_sample[0] = start_end_sample[0][0]

                    if isinstance(start_end_sample[1], np.ndarray):
                        start_end_sample[1] = start_end_sample[1][0]

                start_end_sample = np.array([start_end_sample[0], start_end_sample[1]])

                int_to_warp = np.round(start_end_sample * len(selected_set)).astype(int)
                if int_to_warp[1] == len(selected_set):
                    int_to_warp[1] = int_to_warp[1] - 1

                while int_to_warp[0] == int_to_warp[1] or int_to_warp[0]-1 == int_to_warp[1]:

                    start_end_sample = unwrapcircle(betaprime.rvs(a, b, size=1))

                    if isinstance(start_end_sample[0], np.ndarray):
                        start_end_sample[0] = start_end_sample[0][0]

                    if isinstance(start_end_sample[1], np.ndarray):
                        start_end_sample[1] = start_end_sample[1][0]

                    start_end_sample = np.array([start_end_sample[0], start_end_sample[1]])

                    int_to_warp = np.round(start_end_sample * len(selected_set)).astype(int)
                    if int_to_warp[1] == len(selected_set):
                        int_to_warp[1] = int_to_warp[1] - 1

                int_to_warp = np.array([selected_set[int_to_warp[0]], selected_set[int_to_warp[1]]])
                print(int_to_warp)
                if int_to_warp[0] == 0:
                    int_to_warp[0] = 1
                warped = warp(ref, int_to_warp[0], int_to_warp[1], scale_level)

                distance_matrix = np.zeros((train_x.shape[0]))

                for i in range(0, train_x.shape[0]):
                    distance_matrix[i] = dtw_distance(warped, np.asarray(train_x.values[i, :][0]))

                print(test_y[ind])
                print(train_y[np.argmin(distance_matrix)])
                class_type1[ind,count1] = train_y[np.argmin(distance_matrix)].astype(int)

                count1= count1+1





            count2 = 0
            for type2 in np.array([25,50,75]):
                print("No deberia cambiar")

                sample_ind = np.arange(len(ref))[np.where(w <= np.percentile(w, q=type2))[0]]
                saltos = []
                for i in range(len(sample_ind) - 1):
                    saltos.append(sample_ind[i + 1] - sample_ind[i])

                num_saltos = np.sum(np.asarray(saltos) > 1)

                if num_saltos >= 1:

                    ind_saltos = np.where(np.asarray(saltos) > 1)[0]

                    intervals_we = [np.arange(sample_ind[0], sample_ind[ind_saltos[0]] + 1)]

                    for j in range(1, len(np.where(np.asarray(saltos) > 1)[0])):
                        intervals_we.append(np.arange(sample_ind[ind_saltos[j - 1] + 1], sample_ind[ind_saltos[j]] + 1))

                    intervals_we.append(np.arange(sample_ind[ind_saltos[-1] + 1], sample_ind[-1] + 1))

                    len(intervals_we)
                    leng = []
                    for l in range(len(intervals_we)):
                        leng.append(len(intervals_we[l]))

                    selected_set = intervals_we[np.argmax(leng)]
                else:
                    selected_set = sample_ind

                a = 8
                p = 0.5
                b = a * (1 - p) / p

                start_end_sample = unwrapcircle(betaprime.rvs(a, b, size=1))

                if isinstance(start_end_sample[0], np.ndarray):
                    start_end_sample[0] = start_end_sample[0][0]

                if isinstance(start_end_sample[1], np.ndarray):
                    start_end_sample[1] = start_end_sample[1][0]
                while np.abs(start_end_sample[0] - start_end_sample[1]) < 0.3:

                    a = 8
                    p = 0.5
                    b = a * (1 - p) / p

                    start_end_sample = unwrapcircle(betaprime.rvs(a, b, size=1))

                    if isinstance(start_end_sample[0], np.ndarray):
                        start_end_sample[0] = start_end_sample[0][0]

                    if isinstance(start_end_sample[1], np.ndarray):
                        start_end_sample[1] = start_end_sample[1][0]



                start_end_sample = np.array([start_end_sample[0], start_end_sample[1]])

                int_to_warp = np.round(start_end_sample * len(selected_set)).astype(int)
                if int_to_warp[1] == len(selected_set):
                    int_to_warp[1] = int_to_warp[1] - 1

                while int_to_warp[0] == int_to_warp[1] or int_to_warp[0]-1 == int_to_warp[1]:

                    start_end_sample = unwrapcircle(betaprime.rvs(a, b, size=1))

                    if isinstance(start_end_sample[0], np.ndarray):
                        start_end_sample[0] = start_end_sample[0][0]

                    if isinstance(start_end_sample[1], np.ndarray):
                        start_end_sample[1] = start_end_sample[1][0]

                    start_end_sample = np.array([start_end_sample[0], start_end_sample[1]])

                    int_to_warp = np.round(start_end_sample * len(selected_set)).astype(int)
                    if int_to_warp[1] == len(selected_set):
                        int_to_warp[1] = int_to_warp[1] - 1

                int_to_warp = np.array([selected_set[int_to_warp[0]], selected_set[int_to_warp[1]]])
                print(int_to_warp)

                if int_to_warp[0]==0:
                    int_to_warp[0]=1

                warped = warp(ref, int_to_warp[0], int_to_warp[1], scale_level)

                distance_matrix = np.zeros((train_x.shape[0]))

                for i in range(0, train_x.shape[0]):
                    distance_matrix[i] = dtw_distance(warped, np.asarray(train_x.values[i, :][0]))

                print(test_y[ind])
                print(train_y[np.argmin(distance_matrix)])

                class_type2[ind, count2] = train_y[np.argmin(distance_matrix)].astype(int)

                count2 = count2 + 1




    class_true = class_true.astype(int)


    p1.append(np.sum(class_true[class_type1[:,1]!=0]==class_type1[class_type1[:,1]!=0,0])/len(class_true[class_type1[:,1]!=0]))
    p2.append(np.sum(class_true[class_type1[:,1]!=0]==class_type1[class_type1[:,1]!=0,1])/len(class_true[class_type1[:,1]!=0]))
    p3.append(np.sum(class_true[class_type1[:,1]!=0]==class_type1[class_type1[:,1]!=0,2])/len(class_true[class_type1[:,1]!=0]))

    p11.append(np.sum(class_true[class_type2[:,1]!=0]==class_type2[class_type2[:,1]!=0,0])/len(class_true[class_type2[:,1]!=0]))
    p22.append(np.sum(class_true[class_type2[:,1]!=0]==class_type2[class_type2[:,1]!=0,1])/len(class_true[class_type2[:,1]!=0]))
    p33.append(np.sum(class_true[class_type2[:,1]!=0]==class_type2[class_type2[:,1]!=0,2])/len(class_true[class_type2[:,1]!=0]))

    zer.append((100-len(zeros_warp))/100)




np.column_stack((class_true,class_type1))
np.column_stack((class_true,class_type2))

plt.plot(np.array(['P1','P2','P3']),np.array([np.mean(p1),np.mean(p2),np.mean(p3)]), marker='o', label="Type1")
plt.plot(np.array([np.mean(p11),np.mean(p22),np.mean(p33)]), marker='o', label="Type2")
plt.legend()
plt.xlabel("Percentiles")
plt.ylabel("Accuracy")
plt.title('Scale %0.1f, robus %s' % (scale_level,np.mean(zer)))

#
#
# idx=2
# plt.plot(np.array(['P1','P2','P3']),np.array([p1[idx],p2[idx],p3[idx]]), marker='o', label="Type1")
# plt.plot(np.array([p11[idx],p22[idx],p33[idx]]), marker='o', label="Type2")
# plt.legend()
# plt.xlabel("Percentiles")
# plt.ylabel("Accuracy")
# plt.title('Warp %0.1f, robus %s' % (warp_level,np.mean(zer)))


# #Plot examples percentil
# plt.plot(ref,c="grey")
# plt.plot(np.arange(len(ref))[np.where(w <= np.percentile(w, q=75))[0]], ref[np.where(w <= np.percentile(w, q=75))[0]],c="red")
#
