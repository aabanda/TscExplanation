#from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance



inter = np.loadtxt("inter_warp_3000.txt")
neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")

#GunPoint
inter = np.loadtxt("GP_inter_warp_3000.txt")
neig_y = np.loadtxt("GP_neig_warp_3000.txt")




inter.shape



train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")


train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")


#CBF
# ind =2 #class 1
# ind = 0 #Class 2
ind = 5 #Class 3
# #
# # Arrowhead
# ind=0
# ind=100
# ind=160

ref = test_x.values[ind,:][0].values
# plt.plot(ref)
print(test_y[ind])


# dist_prueba = []
# for i in range(0,train_x.shape[0]):
#     dist_prueba.append(dtw_distance(ref,train_x.values[i,:][0].values))
#
#
# print(train_y[np.argmin(dist_prueba)])
#



def warp(ts,start,end,scale):
    ref = ts
    k = scale
    l = len(ref)
    # start = 20
    # end = 60
    # k = 0.8
    #
    #
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


def plot_colormap(ref, weights):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x = range(0, len(ref) - 1)
    y = ref[1:]
    # dydx = np.abs(clf.coef_)[0]
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

    axs.set_xlim(0, len(x))
    axs.set_ylim(-2.5, 2.5)
    axs.set_ylim(-2, 4)
    plt.show()


def intersection(intervals):
    start, end = intervals.pop()
    while intervals:
        start_temp, end_temp = intervals.pop()
        start = max(start, start_temp)
        end = min(end, end_temp)
    return [start, end]


def unwrapcircle(z):
    u = np.round(random.uniform(-z, 1), decimals=2)
    return intersection([[0,1],[u, u+z]])




from scipy.stats import betaprime
# a = 8
# p=0.5
a = 8
p= 0.3
b = a*(1-p)/p


num_neig = 500


start_end = np.zeros((num_neig,2))
for i in range(0,num_neig):
    start_end[i,:]=unwrapcircle(betaprime.rvs(a, b, size=1))


start_end[start_end<=0]=0
start_end = start_end*len(ref)
start_end = start_end.astype(int)


#Comprobación

# plt.hist(start_end[:,1]-start_end[:,0])
#
#
# intervals = np.zeros((start_end.shape[0],len(ref)))
# intervals[:,:] = 0
# for i in range(0,intervals.shape[0]):
#     intervals[i,range(start_end[i,0],start_end[i,1])] = 1
#
# dydx = np.sum(intervals, axis=0)
# plt.plot(dydx)
#




neig = []
inter = np.zeros((num_neig*6,3))
count = 0
for i in range(0,num_neig):
     start = start_end[i,0]
     end =  start_end[i,1]
     if end == len(ref):
         end = 0
     if start == 0:
         start = 1
     for k in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
         inter[count, :] = np.array([start, end, k])
         neig.append(warp(ref, start, end, k))
         count = count+1
     # k = np.round(random.uniform(0.7,1.3), decimals=1)
     # while k == 1:
     #     k = np.round(random.uniform(0.7, 1.3), decimals=1)
     # inter[i,:] = np.array([start,end,k])
     # neig.append(warp(ref, start, end, k))



inter.shape
# plt.hist(inter[:,0])
# plt.hist(inter[:,1]-inter[:,0])
inter[inter[:,1]==0,1]=len(ref)
variables = inter.copy()
variables = np.delete(variables,2,axis=1)
variables[variables[:,1]==0,1]=len(ref)


inter_sum_ones = np.zeros((variables.shape[0],len(ref)))
inter_sum_ones[:,:] = 0
for i in range(0,inter_sum_ones.shape[0]):
    inter_sum_ones[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1




#plt.hist(variables[:,1].astype(int)-variables[:,0].astype(int))
dydx = np.sum(inter_sum_ones, axis=0)
plt.plot(dydx)
#plt.plot(img*1000/2)









#
#
# intervals = np.zeros((inter.shape[0],len(ref)))
# intervals[:,:] = np.nan
# for i in range(0,intervals.shape[0]):
#     intervals[i,range(inter[i,0].astype(int),inter[i,1].astype(int))] = i
#     if inter[i,2]>1:
#         colormp = 'red'
#     else:
#         colormp = 'green'
#     plt.plot(range(0, len(ref)), intervals[i, :],c=colormp)
#
# import matplotlib.patches as mpatches
# red_patch = mpatches.Patch(color='red', label='level > 1')
# plt.legend(handles=[red_patch],loc='upper left')






distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))



neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))

plt.plot(dydx)




np.savetxt("GP_inter_warp_3000.txt", inter)
np.savetxt("GP_neig_warp_3000.txt", neig_y.astype(int))












inter
#
#
# inter_binary = np.zeros((inter.shape[0],len(ref)))
# for i in range(0,inter.shape[0]):
#     inter_binary[i,range(inter[i,0].astype(int),inter[i,1].astype(int))] = inter[i,2]
# np.sum(inter_binary, axis=0)
#
#
# inter_binary.shape
# inter_seg = inter.copy()
# neig_y_seg = neig_y.copy()
#
# neig_y = neig_y[inter[:,2]==0.7]
# inter = inter[inter[:,2]==0.7,:]
#
#
# inter.shape
# len(neig_y)









#
#
# intervals = np.zeros((variables.shape[0],len(ref)))
# intervals[:,:] = np.nan
# for i in range(0,intervals.shape[0]):
#     intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = i
#     if inter[neig_y!=test_y[ind],2][i]>1:
#         colormp = 'red'
#     else:
#         colormp = 'green'
#     plt.plot(range(0, len(ref)), intervals[i, :],c=colormp)
# import matplotlib.patches as mpatches
#
# red_patch = mpatches.Patch(color='red', label='level > 1')
# plt.legend(handles=[red_patch],loc='upper left')
# plt.legend()












#Separado por wapr level
inter = np.loadtxt("inter_warp_3000.txt")
neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")





inter_seg = inter.copy()
neig_y_seg = neig_y.copy()

inter = inter_seg
neig_y = neig_y_seg


neig_y = neig_y[inter[:,2]==0.8]
inter = inter[inter[:,2]==0.8,:]



# neig_y.shape
# inter.shape


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
same_int.shape
other_int.shape



same_int = np.delete(same_int,np.where(same_int[:,0]==same_int[:,1])[0],0 )
np.where(same_int[:,0]==same_int[:,1])[0]

other_int = np.delete(other_int,np.where(other_int[:,0]==other_int[:,1])[0],0 )
np.where(other_int[:,0]==other_int[:,1])[0]




# same_int = np.delete(same_int,175, axis=0)
# same_int = np.delete(same_int,107, axis=0)
# same_int.shape

#
# from scipy.spatial.distance import directed_hausdorff
# a =np.array([2,3,4])
# b =np.array([1,2,3,4])
# a = np.asarray(range(same_int[0,:][0].astype(int),same_int[0,:][1].astype(int)))
# a = a.reshape(-1,1)
# b =  np.asarray(range(other_int[0, :][0].astype(int), other_int[0, :][1].astype(int)))
# b = b.reshape(-1,1)

from scipy.spatial.distance import directed_hausdorff
# a = np.arange(2,26)
# b = np.arange(24,50)
# a = a.reshape(-1,1)
# b = b.reshape(-1,1)
#
#
# max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0])



# ind_int = []
# dist_int = []
# for i in range(0,same_int.shape[0]):
#     for j in range(0,other_int.shape[0]):
#         a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
#         a = a.reshape(-1, 1)
#         b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
#         b = b.reshape(-1, 1)
#         dist_int.append(max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0]))
#         ind_int.append([i,j])
#
# plt.hist(dist_int,bins=20)
# len(dist_int)


from scipy.spatial.distance import directed_hausdorff
dist_int = np.zeros((same_int.shape[0],other_int.shape[0]))
for i in range(0,same_int.shape[0]):
    for j in range(0,other_int.shape[0]):
        a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
        a = a.reshape(-1, 1)
        b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
        b = b.reshape(-1, 1)
        dist_int[i,j]=max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0])



dist_array= dist_int.reshape(1,-1)[0]
len(dist_array)
#plt.hist(dist_array,bins=20)




#
#
# min_per_same = []
# perc_per_same = []
# thresh_per_same = []
# for i in range(0,dist_int.shape[0]):
#     min_per_same.append(np.argmin(dist_int[i,:]))
#     perc_per_same.append(np.where(dist_int[i,:]<np.percentile(dist_int,q=2))[0])
#     thresh_per_same.append(np.where(dist_int[i,:]<1)[0])
#
#
#
# len(thresh_per_same)
# thresh_per_same[50]
# for i in range(0, len(thresh_per_same)):
#     print(i)
#     print(thresh_per_same[i])
#
#
#
# same_int = np.delete(same_int,np.where(same_int[:,0]==same_int[:,1])[0],0 )
# np.where(same_int[:,0]==same_int[:,1])[0]
#
# other_int = np.delete(other_int,np.where(other_int[:,0]==other_int[:,1])[0],0 )
# np.where(other_int[:,0]==other_int[:,1])[0]


#
#
# from scipy.spatial.distance import directed_hausdorff
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
# np.where(same_int[:,0]==same_int[:,1])
#
#
# np.where(other_int[:,0]==other_int[:,1])
#
#
# same_int[242,:]
# other_int[60,:]


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




#Elijo un threshold:



thresh_per_same = []
for i in range(0,dist_int.shape[0]):
    thresh_per_same.append(np.where(dist_int[i,:]<5)[0])
k = np.concatenate(thresh_per_same, axis=0)

len(np.unique(k))
other_int.shape
same_int.shape


#
#
# np.unique(min_per_same, return_counts=True)
# len(np.unique(min_per_same))
#
# # len(perc_per_same[107])
# k = np.concatenate(perc_per_same, axis=0 )
# np.unique(k, return_counts=True)
# len(np.unique(k, return_counts=True)[0])


#
#
# k = np.concatenate(thresh_per_same, axis=0 )
# np.unique(k, return_counts=True)
# len(np.unique(k, return_counts=True)[0])



#
#
# np.unique(neig_y, return_counts=True)[1][0]
#
# len(np.unique(min_per_same))

# other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(min_per_same))


other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
len(other_index)





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




# ind_int[np.where(dist_int>np.percentile(dist_int,q=30))[0]]
# ind_int[np.array([0,1])]










variables2 = np.concatenate((variables[inter2[:,2]>1,:][largest_indices1,:],variables[inter2[:,2]<1,:][largest_indices2,:] ))


intervals = np.zeros((variables2.shape[0],len(ref)))
intervals[:,:] = np.nan
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables2[i,0].astype(int),variables2[i,1].astype(int))] = i
    if variables2[i,2]>1:
        colormp = 'red'
    else:
        colormp = 'green'
    plt.plot(range(0, len(ref)), intervals[i, :],c=colormp)

import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='level > 1')
plt.legend(handles=[red_patch],loc='upper left')












intervals = np.zeros((variables.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
np.sum(intervals, axis=0)


print("Interval lengths")
print(np.mean(np.sum(intervals, axis=1)))
print(np.std(np.sum(intervals, axis=1)))

weigths1 = np.sum(intervals, axis=0)


plot_colormap(ref,np.sum(intervals, axis=0))
plot_colormap(ref,f1['mean']/f3['mean'])






#Aqui amaia



inter
# point_other = np.sum(intervals, axis=0)
# point_same = np.sum(intervals, axis=0)

# point_other = np.sum(intervals, axis=0)/(500*p)
# point_same = np.sum(intervals, axis=0)/(500*p)
#
# point_other-point_same

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

    axs.set_xlim(0, len(x))
    axs.set_ylim(-2.5, 2.5)
    axs.set_ylim(-2, 4)
    plt.show()









plt.plot(ref, c='grey')

weigths = np.insert(dydx, 0, 0)

plt.plot(np.arange(len(ref))[weigths>0.5], ref[weigths>0.5], c='red')
plt.plot(np.arange(len(ref))[weigths>0.7], ref[weigths>0.7], c='red')


#0.7
vec1=warp(ref,60,100, 0.7)
vec2=warp(ref,75,100, 0.7)

#1.3
vec1=warp(ref,80,100, 1.3)
vec2=warp(ref,90,110, 1.3)



#0.7 GunPoint
vec1=warp(ref,10,60, 0.7)
vec2=warp(ref,20,65,0.7)

#1.3 GunPoint
vec1=warp(ref,40,100, 1.3)
vec2=warp(ref,20,120,1.3)

# plt.plot(ref)
vec = []
vec.append(vec1)
vec.append(vec2)


distance_matrix = np.zeros((len(vec),train_x.shape[0]))

for i in range(0, len(vec)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(vec[i], np.asarray(train_x.values[j,:][0]))


print(test_y[ind])
print(train_y[np.argmin(distance_matrix,axis=1)])


print(np.argmin(distance_matrix,axis=1))
plt.plot(np.asarray(train_x.values[np.argmin(distance_matrix,axis=1)[0],:][0]))
print(train_y[np.argmin(distance_matrix,axis=1)[0]])



plt.plot(ref,c='grey')
plt.plot(vec1)
plt.plot(vec2)
plt.xlim(0,len(ref))


#
#
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# inter[inter[:,1]==0,1] = len(ref)
# labels = neig_y
#
# co = 0
# for lbl in np.unique(labels):
#     indices = np.where(labels == lbl)
#     ax.scatter(inter[indices,0], inter[indices,1], inter[indices,2], s=50, alpha=0.6, label=str(lbl), cmap='rainbow')
#     co = co+1
#     print(inter[:,0], inter[:,1], inter[:,2],lbl)
#
# ax.set_xlabel('start')
# ax.set_ylabel('end')
# ax.set_zlabel('level')
# ax.legend()

# plt.show()















#
# X= inter_binary
#
# X = inter_seg
# y = neig_y_seg
#
#
#
# # Logistic regression
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.metrics import f1_score
#
# clf = LogisticRegression(random_state=0,max_iter=5000)
#
# kf = StratifiedKFold(n_splits=3)
# accu = []
# for train_index, test_index in kf.split(X,y):
#     X_train, X_test = X[train_index,:], X[test_index,:]
#     y_train, y_test = y[train_index], y[test_index]
#
#     clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
#     accu.append(f1_score(y_test.astype(int),pred.astype(int),average="weighted"))
#     print(classification_report(y_test, pred))
#
# clf.fit(X,y)
# print(np.mean(accu))
#
#
#
#
#
# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
#
# x = range(0,len(ref))
# y = ref
# # dydx = np.abs(clf.coef_)[0]
# dydx = clf.coef_[0]
# # dydx = np.exp(clf.coef_[0])
# len(dydx)
#
#
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
#
#
# fig, axs = plt.subplots()
#
# # Create a continuous norm to map from data points to colors
# norm = plt.Normalize(dydx.min(), dydx.max())
# lc = LineCollection(segments, cmap='jet', norm=norm)
# # Set the values used for colormapping
# lc.set_array(dydx)
# lc.set_linewidth(2)
# line = axs.add_collection(lc)
#
#
# fig.colorbar(line, ax=axs)
#
# axs.set_xlim(0, len(x))
# axs.set_ylim(-2.5, 2.5)
# axs.set_ylim(-2, 4)
# plt.show()
#









inter = np.loadtxt("inter_warp_3000.txt")
neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")

inter = np.loadtxt("GP_inter_warp_3000.txt")
neig_y = np.loadtxt("GP_neig_warp_3000.txt")






inter.shape

X = inter
y = neig_y


intervals = np.zeros((inter.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(inter[i,0].astype(int),inter[i,1].astype(int))] = inter[i,2]




from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.datasets import load_arrow_head
from sktime.utils.slope_and_trend import _slope
#
steps = [
    (
        "extract",
        RandomIntervalFeatureExtractor(
            n_intervals="sqrt", features=[np.mean, np.std, _slope]
        ),
    ),
    ("clf", DecisionTreeClassifier()),
]
time_series_tree = Pipeline(steps)

#
#
# kf = StratifiedKFold(n_splits=5)
# accu = []
# X = intervals
#
# for train_index, test_index in kf.split(X,y):
#     X_train, X_test = X[train_index,:], X[test_index,:]
#     y_train, y_test = y[train_index], y[test_index]
#
#     time_series_tree.fit(X_train.reshape(X_train.shape[0],1,X_train.shape[1]), y_train)
#     accu.append(time_series_tree.score(X_test.reshape(X_test.shape[0],1,X_test.shape[1]), y_test))
#
#     # clf.fit(X_train, y_train)
#     # pred = clf.predict(X_test)
#     # accu.append(f1_score(y_test.astype(int),pred.astype(int),average="weighted"))
#     # print(classification_report(y_test, pred))
#
# # clf.fit(X,y)
# print(np.mean(accu))



tsf = TimeSeriesForestClassifier(
    estimator=time_series_tree,
    n_estimators=100,
    criterion="entropy",
    bootstrap=True,
    oob_score=True,
    random_state=1,
    n_jobs=-1,
)


tsf = TimeSeriesForestClassifier(
    estimator=time_series_tree,
    n_estimators=1,
    criterion="entropy",
    bootstrap=True,
    oob_score=True,
    random_state=1,
    n_jobs=-1,
)


# tsf = TimeSeriesForestClassifier(n_estimators=5)



intervals = np.zeros((inter.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(inter[i,0].astype(int),inter[i,1].astype(int))] = inter[i,2]




y = neig_y


X_data_intervals = pd.DataFrame(intervals)
X = pd.DataFrame()
X["dim_0"] = [pd.Series(X_data_intervals.iloc[x, :]) for x in range(len(X_data_intervals))]
accu = []
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    tsf.fit(X_train, y_train)
    y_pred = tsf.predict(X_test)
    accu.append(f1_score(y_test.astype(int), y_pred.astype(int), average="weighted"))

print(np.mean(accu))
fi = tsf.feature_importances_


f1 = fi[0]
f3 = fi[2]
plt.plot(f3)
plt.plot(f1)
plt.plot((f1/f3)['mean']*10)

plt.plot(f1['mean']*10, label="Imp mean")
plt.plot((f1/f3)['mean']*10, label="Imp mean Norm")
plt.plot(intervals[y==test_y[ind].astype(int),:].mean(axis=0), label ="Mean same class")
plt.plot(intervals[y!=test_y[ind].astype(int),:].mean(axis=0), label ="Mean other class")
plt.legend(loc='upper right')


plot_colormap(ref, (f1/f3)['mean'])












#Mas pequeño que uno
inter_less = inter[inter[:,2]<1,:]
intervals = np.zeros((inter_less.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(inter_less[i,0].astype(int),inter_less[i,1].astype(int))] = inter_less[i,2]
    #intervals[i,range(inter_less[i,0].astype(int),inter_less[i,1].astype(int))] = 1
y = neig_y[inter[:,2]<1]


X_data_intervals = pd.DataFrame(intervals)
X = pd.DataFrame()
X["dim_0"] = [pd.Series(X_data_intervals.iloc[x, :]) for x in range(len(X_data_intervals))]
accu = []
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    tsf.fit(X_train, y_train)
    y_pred = tsf.predict(X_test)
    accu.append(f1_score(y_test.astype(int), y_pred.astype(int), average="weighted"))

print(np.mean(accu))
fi = tsf.feature_importances_
f1 = fi[0]
f3 = fi[2]
plt.plot(f1['mean']*10, label="Imp mean")
plt.plot((f1/f3)['mean']*10, label="Imp mean Norm")
plt.plot(intervals[y==test_y[ind].astype(int),:].mean(axis=0), label ="Mean same class")
plt.plot(intervals[y!=test_y[ind].astype(int),:].mean(axis=0), label ="Mean other class")
plt.legend(loc='upper right')


plot_colormap(ref, (f1/f3)['mean'])

plt.figure()

for i in np.where(y==test_y[ind].astype(int))[0][range(20)]:
    plt.plot(intervals[i,:],c ="red")

for i in np.where(y!=test_y[ind].astype(int))[0][range(20)]:
    plt.plot(intervals[i,:],c ="blue")









#Mas grande que uno
inter_more = inter[inter[:,2]>1,:]
intervals = np.zeros((inter_more.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(inter_more[i,0].astype(int),inter_more[i,1].astype(int))] = inter_more[i,2]

y = neig_y[inter[:,2]>1]
X_data_intervals = pd.DataFrame(intervals)
X = pd.DataFrame()
X["dim_0"] = [pd.Series(X_data_intervals.iloc[x, :]) for x in range(len(X_data_intervals))]



accu = []
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    tsf.fit(X_train, y_train)
    y_pred = tsf.predict(X_test)
    accu.append(f1_score(y_test.astype(int), y_pred.astype(int), average="weighted"))

print(np.mean(accu))

fi = tsf.feature_importances_
f1 = fi[0]
f3 = fi[2]
plt.plot(f1['mean']*10, label="Imp mean")
plt.plot((f1/f3)['mean']*10, label="Imp mean Norm")
plt.plot(intervals[y==test_y[ind].astype(int),:].mean(axis=0), label ="Mean same class")
plt.plot(intervals[y!=test_y[ind].astype(int),:].mean(axis=0), label ="Mean other class")
plt.legend(loc='upper right')

plot_colormap(ref, (f1/f3)['mean'])






#Separado por warp level
warp_level = 0.8
inter_less = inter[inter[:,2]==warp_level,:]
y = neig_y[inter[:,2]==warp_level]
intervals = np.zeros((inter_less.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(inter_less[i,0].astype(int),inter_less[i,1].astype(int))] = inter_less[i,2]

np.unique(y, return_counts=True)

# tsf = TimeSeriesForestClassifier(n_estimators=1)
X_data_intervals = pd.DataFrame(intervals)
X = pd.DataFrame()
X["dim_0"] = [pd.Series(X_data_intervals.iloc[x, :]) for x in range(len(X_data_intervals))]
accu = []
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    tsf.fit(X_train, y_train)
    y_pred = tsf.predict(X_test)
    accu.append(f1_score(y_test.astype(int), y_pred.astype(int), average="weighted"))

print(np.mean(accu))

# tsf.__dict__ 
tsf.fit(X, y)

fi = tsf.feature_importances_
f1 = fi[0]
f3 = fi[2]
# plt.figure()
# plt.plot(f3['mean'], label="Intervals associated")
# plt.legend()
plt.plot(f1['mean'] ,label="Imp mean")
plt.plot(f1['mean']/f3['mean'], label="Imp mean Norm")
plt.plot(intervals[y==test_y[ind].astype(int),:].mean(axis=0), label ="Mean same class")
plt.plot(intervals[y!=test_y[ind].astype(int),:].mean(axis=0), label ="Mean other class")
# plt.plot(intervals[y==test_y[ind].astype(int),:].mean(axis=0)*200/len(np.where(y==test_y[ind].astype(int))[0]), label ="Mean same class")
# plt.plot(intervals[y!=test_y[ind].astype(int),:].mean(axis=0)*200/len(np.where(y!=test_y[ind].astype(int))[0]), label ="Mean other class")
plt.legend(loc='upper left')
plt.xlabel("t")
plt.ylabel("Importance")
intervals[y==test_y[ind].astype(int),:].shape
intervals[y!=test_y[ind].astype(int),:].shape




np.unique(y, return_counts=True)
# plt.plot(f1*10)
# plt.plot(f3)
# w = (f1 / f3)['mean']
plt.plot(f1['mean']/f3['mean'], label="Imp mean Norm")
plot_colormap(ref, f1['mean']/f3['mean'])
plot_colormap(ref, (f1)['mean'])

weigths2 = np.asarray(f1['mean'] / f3['mean'])

np.savetxt("weigths1.txt", weigths1)
np.savetxt("weigths2.txt", weigths2)

plt.figure()
for i in np.where(y==test_y[ind].astype(int))[0][range(10)]:
    plt.plot(intervals[i,:],c ="red")

plt.figure()
for i in np.where(y!=test_y[ind].astype(int))[0][range(20)]:
    plt.plot(intervals[i,:],c ="blue")


