from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import sys


num_neig = 500
warp_level = 0.8

threshold = int(sys.argv[0])
print(threshold)
cutoff = float(sys.argv[0])
print(cutoff)
# threshold = 5
# cutoff = 0.7

inter = np.loadtxt("inter_warp_3000.txt")
neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")


train_x, train_y = load_from_tsfile_to_dataframe("CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("CBF_TEST.ts")
#CBF
# ind =2 #class 1
# ind = 0 #Class 2
ind = 5 #Class 3
ref = test_x.values[ind,:][0].values



inter_seg = inter.copy()
neig_y_seg = neig_y.copy()

neig_y = neig_y[inter[:,2]==warp_level]
inter = inter[inter[:,2]==warp_level,:]







def warp(ts,start,end,scale):
    ref = ts
    k = scale
    l = len(ref)

    if start>0 and end>0:
        rescaled_t = np.concatenate((np.array(range(start)),
                                     np.array(range(0,end-start)) * k+start,
                                     range(np.int(start+(end-start)*k),(np.int(start+(end-start)*k) + l-end))))
    elif start>0 and end==0:
        rescaled_t = np.concatenate((np.array(range(start)), np.array(range(0, l - start)) * k + start))
    else:
        rescaled_t =  np.array(range(l))*k


    len(rescaled_t)

    new_index = np.array(range(np.int(np.max(rescaled_t))+1))
    t_trasnformed = np.zeros(len(new_index))

    end_ind =0
    for t in range(0,len(new_index)):
        if end==0:
            if t == 0:
                t_trasnformed[t] = ref[t]
            elif t <= start:
                t_trasnformed[t] = ref[t]
            else:
                before = rescaled_t[rescaled_t < t][-1]
                after = rescaled_t[rescaled_t >= t][0]

                t_trasnformed[t] = (ref[np.where(rescaled_t == before)[0][0]] * (k - (t - before)) + ref[
                    np.where(rescaled_t == after)[0][0]] * (
                                            k - (after - t))) / k
        else:
            if t == 0:
                t_trasnformed[t] = ref[t]
            elif t <= start:
                t_trasnformed[t] = ref[t]
            elif t < np.int(start + (end - start) * k):
                before = rescaled_t[rescaled_t < t][-1]
                after = rescaled_t[rescaled_t >= t][0]

                t_trasnformed[t] = (ref[np.where(rescaled_t == before)[0][0]] * (k - (t - before)) + ref[
                    np.where(rescaled_t == after)[0][0]] * (k - (after - t))) / k
            else:
                t_trasnformed[t] = ref[end + end_ind]
                end_ind = end_ind + 1

    return  t_trasnformed





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

other_int = inter2
same_int = same_int[:,[0,1]]
other_int = other_int[:,[0,1]]
same_int = np.delete(same_int,np.where(same_int[:,0]==same_int[:,1])[0],0 )
other_int = np.delete(other_int,np.where(other_int[:,0]==other_int[:,1])[0],0 )




dist_int = np.zeros((same_int.shape[0],other_int.shape[0]))
for i in range(0,same_int.shape[0]):
    for j in range(0,other_int.shape[0]):
        a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
        a = a.reshape(-1, 1)
        b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
        b = b.reshape(-1, 1)
        dist_int[i,j]=max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0])



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


#plt.plot(ran,num)


#Elijo un threshold:
thresh_per_same = []
for i in range(0,dist_int.shape[0]):
    thresh_per_same.append(np.where(dist_int[i,:]<threshold)[0])
k = np.concatenate(thresh_per_same, axis=0)


other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
print(len(other_index))





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



intervals = np.zeros((variables.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
np.sum(intervals, axis=0)

w = np.sum(intervals, axis=0)
dydx = (w-np.min(w))/(np.max(w)-np.min(w))
weigths = np.insert(dydx, 0, 0)

plt.plot(np.arange(len(ref))[np.where(weigths>threshold)[0]], ref[np.where(weigths>threshold)[0]], c='red')
plt.plot(np.arange(len(ref))[np.where(weigths>0.9)[0]], ref[np.where(weigths>0.9)[0]], c='red')


red = np.where(weigths>cutoff)[0]
start_end = np.zeros((num_neig, 2))
dis = []
for i in range(0,num_neig/2):
    p = random.sample(list(red),1)[0]
    sign_left = random.sample([0,1],1)[0]
    sign_right = random.sample([0,1],1)[0]

    leng_left= random.randrange(1,5)
    if sign_left==0:
        leng_left = p-red[0]-leng_left
    else:
        leng_left = p - red[0] + leng_left

    leng_rigth = random.randrange(1, 5)
    if sign_right == 0:
        leng_rigth = red[-1] -p - leng_rigth
    else:
        leng_rigth =red[-1] -p + leng_rigth


    start_end[i, 0] = p - leng_left
    start_end[i, 1] = p + leng_rigth


    b = np.asarray(range(start_end[i, 0].astype(int), start_end[i, 1].astype(int)))
    b = b.reshape(-1, 1)
    dis.append(max(directed_hausdorff(red.reshape(-1,1), b)[0], directed_hausdorff(b, red.reshape(-1,1))[0]))

plt.bar(np.unique(dis), np.unique(dis, return_counts=True)[1])

gray = np.where(weigths < threshold)[0]
dis = []
for i in range(int(num_neig / 2), num_neig):
    p = random.sample(list(gray[2:-2]), 1)[0]
    # len_left = random.randint(1, np.min([p - 1, 20]))
    # len_right = random.randint(1, np.min([len(ref) - p, 20]))
    len_left = random.randint(1, np.min([p - 1, 128]))
    len_right = random.randint(1, np.min([len(ref) - p, 128]))
    start_end[i, 0] = p - len_left
    start_end[i, 1] = p + len_right

    b = np.asarray(range(start_end[i, 0].astype(int), start_end[i, 1].astype(int)))
    b = b.reshape(-1, 1)
    dis.append(max(directed_hausdorff(red.reshape(-1, 1), b)[0], directed_hausdorff(b, red.reshape(-1, 1))[0]))

plt.bar(np.unique(dis), np.unique(dis, return_counts=True)[1])

neig = []
inter = np.zeros((num_neig, 3))
count = 0
for i in range(0, num_neig):
    start = start_end[i, 0].astype(int)
    end = start_end[i, 1].astype(int)
    if end == len(ref):
        end = 0
    if start == 0:
        start = 1
    for k in [warp_level]:
        inter[count, :] = np.array([start, end, k])
        neig.append(warp(ref, start, end, k))
        count = count + 1

distance_matrix = np.zeros((len(neig), train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i, j] = dtw_distance(neig[i], np.asarray(train_x.values[j, :][0]))

neig_y = train_y[np.argmin(distance_matrix, axis=1)]

print("f1: warp %s, threshold %s, cutoff %s" % (warp_level, threshold, cutoff))
# print(np.unique(neig_y, return_counts=True))
print(f1_score(neig_y, np.repeat(['1', '3'], [250, 250]), average="weighted"))
print(confusion_matrix(neig_y, np.repeat(['1', '3'], [250, 250])))

