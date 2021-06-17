#from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from transformations.old.warp_function import warp
from scipy.stats import betaprime
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.dictionary_based import BOSSEnsemble



db= "CBF"
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db,db))
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/%s/%s_TEST.ts"% (db,db))

classifier = "dtw"
transformation = "warp"
warp_level=0.7



if classifier=="st":
    clf = ShapeletTransformClassifier(time_contract_in_mins=60)
    clf.fit(train_x, train_y)
elif classifier=="boss":
    clf = BOSSEnsemble(max_ensemble_size=100)
    clf.fit(train_x, train_y)


def scale(ref, start, end, k):
    shifted_t = ref.copy()
    shifted_t[range(start, end)] = ref[range(start, end)] * k
    return shifted_t


def noise(ref, start, end, k):
    shifted_t = ref.copy()
    noise = np.random.normal(0, np.abs(np.max(ref) - np.min(ref)) * k / 100, len(range(start, end)))
    shifted_t[range(start, end)] = ref[range(start, end)] + noise
    return shifted_t



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




class_true = test_y

p1 = []
p2 = []
p3 = []
p11 = []
p22 = []
p33 = []

p1a = []
p2a = []
p3a = []
p11a = []
p22a = []
p33a = []

zer = []


for repetition in range(0, 10):

    print("repetition")
    print(repetition)
    class_type1 = np.zeros((len(test_y), 3))
    class_type2 = np.zeros((len(test_y), 3))

    zeros_warp = []
    impor_ind = []
    for ind in range(len(test_y)):

        #w = np.loadtxt('transformations/weights/CBF_weights%0.1f_%d.txt' % (warp_level, ind))
        # w = np.loadtxt('transformations/weights/threshold/CBF_weights%0.1f_%d.txt' % (warp_level, ind))


        w = np.loadtxt('neig/%s/%s/%s/weights%0.1f_%d.txt' % (db, transformation, classifier, warp_level, ind))

        if np.sum(w)==0 :
            zeros_warp.append(ind)

        else:
            print("index")
            print(ind)
            ref = test_x.values[ind, :][0].values


            count1 = 0

            for type1 in np.array([90,50,10]):
            #for type1 in np.array([75,50,25]):
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

                if transformation=="warp":
                    warped = warp(ref, int_to_warp[0], int_to_warp[1], warp_level)
                elif transformation=="scale":
                    warped = scale(ref, int_to_warp[0], int_to_warp[1], warp_level)
                elif transformation == "noise":
                    warped = noise(ref, int_to_warp[0], int_to_warp[1], warp_level)

                if classifier == "st":
                    predic = clf.predict(warped.reshape(1, 1, -1))[0]
                elif classifier == "boss":
                    predic= clf.predict(warped.reshape(1, 1, -1))[0]
                elif classifier=="dtw":
                    distance_matrix = np.zeros((train_x.shape[0]))

                    for i in range(0, train_x.shape[0]):
                        distance_matrix[i] = dtw_distance(warped, np.asarray(train_x.values[i, :][0]))


                    predic = train_y[np.argmin(distance_matrix)].astype(int)

                print(test_y[ind])
                print(predic)
                class_type1[ind,count1] = predic

                count1= count1+1





            count2 = 0
            #for type2 in np.array([25,50,75]):
            for type2 in np.array([10,50,90]):
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

                if transformation == "warp":
                    warped = warp(ref, int_to_warp[0], int_to_warp[1], warp_level)
                elif transformation == "scale":
                    warped = scale(ref, int_to_warp[0], int_to_warp[1], warp_level)
                elif transformation == "noise":
                    warped = noise(ref, int_to_warp[0], int_to_warp[1], warp_level)

                if classifier == "st":
                    predic = clf.predict(warped.reshape(1, 1, -1))[0]
                elif classifier == "boss":
                    predic = clf.predict(warped.reshape(1, 1, -1))[0]
                elif classifier == "dtw":
                    distance_matrix = np.zeros((train_x.shape[0]))

                    for i in range(0, train_x.shape[0]):
                        distance_matrix[i] = dtw_distance(warped, np.asarray(train_x.values[i, :][0]))


                    predic = train_y[np.argmin(distance_matrix)].astype(int)

                print(test_y[ind])
                print(predic)
                class_type2[ind, count2] = predic

                count2 = count2 + 1




    class_true = test_y.astype(int)[np.setdiff1d(np.arange(len(test_y)),zeros_warp)]
    class_type2 = class_type2[np.setdiff1d(range(len(test_y)),zeros_warp),:]
    class_type1 = class_type1[np.setdiff1d(range(len(test_y)),zeros_warp),:]


    p1.append(np.sum(class_true == class_type1[:,0]) / len( class_true))
    p2.append(np.sum(class_true == class_type1[:, 1]) / len(class_true))
    p3.append(np.sum(class_true == class_type1[:, 2]) / len(class_true))

    p11.append(np.sum(class_true == class_type2[:, 0]) / len(class_true))
    p22.append(np.sum(class_true == class_type2[:, 1]) / len(class_true))
    p33.append(np.sum(class_true == class_type2[:, 2]) / len(class_true))

    # np.sum(class_true == class_type2[:, 0]) / len(class_true)
    # np.sum(class_true == class_type2[:, 1]) / len(class_true)
    # np.sum(class_true == class_type2[:, 2]) / len(class_true)
    #
    # p1a.append(np.sum(class_true[class_type1[:, 1] != 0] == class_type1[class_type1[:, 1] != 0, 0]) / len(
    #     class_true[class_type1[:, 1] != 0]))
    # p2a.append(np.sum(class_true[class_type1[:,1]!=0]==class_type1[class_type1[:,1]!=0,1])/len(class_true[class_type1[:,1]!=0]))
    # p3a.append(np.sum(class_true[class_type1[:,1]!=0]==class_type1[class_type1[:,1]!=0,2])/len(class_true[class_type1[:,1]!=0]))
    #
    # p11a.append(np.sum(class_true[class_type2[:,1]!=0]==class_type2[class_type2[:,1]!=0,0])/len(class_true[class_type2[:,1]!=0]))
    # p22a.append(np.sum(class_true[class_type2[:,1]!=0]==class_type2[class_type2[:,1]!=0,1])/len(class_true[class_type2[:,1]!=0]))
    # p33a.append(np.sum(class_true[class_type2[:,1]!=0]==class_type2[class_type2[:,1]!=0,2])/len(class_true[class_type2[:,1]!=0]))

    zer.append((len(zeros_warp))/len(test_y))




np.column_stack((class_true,class_type1))
np.column_stack((class_true,class_type2))

plt.plot(np.array(['P1','P2','P3']),np.array([np.mean(p1),np.mean(p2),np.mean(p3)]), marker='o', label="Type1")
plt.plot(np.array([np.mean(p11),np.mean(p22),np.mean(p33)]), marker='o', label="Type2")
plt.legend()
plt.ylim(0,1)
plt.xlabel("Percentiles")
plt.ylabel("Accuracy")
#plt.title('Scale %0.1f, robus %s' % (warp_level,np.mean(zer)))
#plt.title('Scale %0.1f, robus %s' % (warp_level,np.mean(zer)))
plt.title('Warp %0.1f, robus %s' % (warp_level,np.mean(zer)))

# area = (np.mean(p11)-np.mean(p1))+ (np.mean(p22)-np.mean(p2))+(np.mean(p33)-np.mean(p3))
#
# print(area)



print(np.column_stack((np.array([np.mean(p1),np.mean(p2),np.mean(p3)]), np.array([np.mean(p11),np.mean(p22),np.mean(p33)]))))



np.save('neig/%s/%s/%s/eval%0.1f.txt' % (db, transformation, classifier, warp_level),np.column_stack((np.array([np.mean(p1),np.mean(p2),np.mean(p3)]), np.array([np.mean(p11),np.mean(p22),np.mean(p33)]))))

import numpy as np
a = np.array([np.mean(p11),np.mean(p22), np.mean(p33)])
b = np.array([np.mean(p1),np.mean(p2),np.mean(p3)])



print(  ((a[0]+a[1])/2 +(a[1]+a[2])/2) -   ((b[0]+b[1])/2 +(b[1]+b[2])/2) )
print(np.mean(zer))
