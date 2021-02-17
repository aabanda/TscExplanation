from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance




train_x, train_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/CBF/CBF_TEST.ts")

# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")


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

l = len(ref)

start = 20
end = 100
k = 0.8

start = 20
end = 60
k = 1.2

start = 50
end = 0
k = 1.4


# plt.plot(ref)
# plt.plot(warp(ref, start, end, k))
#
#


num_neig = 500
neig = []
inter = np.zeros((num_neig,3))
for i in range(0,num_neig):
     start = random.randint(1,len(ref)-10)
     end = random.randint(start+1, len(ref))
     if end == len(ref):
         end = 0
     k = np.round(random.uniform(0.7,1.3), decimals=1)
     while k == 1:
         k = np.round(random.uniform(0.7, 1.3), decimals=1)
     inter[i,:] = np.array([start,end,k])
     neig.append(warp(ref, start, end, k))


# plt.hist(inter[:,0])


distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))



neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))





inter[inter[:,1]==0,1]=len(ref)
(inter[:,1]-inter[:,0])>0
inter2 = inter.copy()
neig_y



#
#
# test_y[ind]
# plt.scatter(inter[neig_y!=test_y[ind],0],inter[neig_y!=test_y[ind],1],color='r',label="Other class")
# plt.scatter(inter[neig_y==test_y[ind],0],inter[neig_y==test_y[ind],1],color='b',label="Same class")
# plt.xlabel("start")
# plt.ylabel("end")
# plt.legend()
#
#
#
# plt.scatter(inter[neig_y!=test_y[ind],0],inter[neig_y!=test_y[ind],2],color='r',label="Other class")
# plt.scatter(inter[neig_y==test_y[ind],0],inter[neig_y==test_y[ind],2],color='b',label="Same class")
# plt.xlabel("start")
# plt.ylabel("level")
# plt.legend()
#
# plt.scatter(inter[neig_y!=test_y[ind],1],inter[neig_y!=test_y[ind],2],color='r',label="Other class")
# plt.scatter(inter[neig_y==test_y[ind],1],inter[neig_y==test_y[ind],2],color='b',label="Same class")
# plt.xlabel("end")
# plt.ylabel("level")
# plt.legend()




# type(inter)
variables = inter.copy()
variables = np.delete(variables,2,axis=1)
variables.shape
variables = variables[neig_y!=test_y[ind],:]
variables.shape


# plt.scatter(inter[neig_y=='2',0],inter[neig_y=='2',1], marker="x", c=inter[neig_y=='1',2],label="Other class")
# plt.scatter(inter[neig_y=='0',0],inter[neig_y=='0',1], marker="o", c=inter[neig_y=='3',2],label="Same class")
# plt.xlabel("start")
# plt.ylabel("end")
# plt.legend()


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








#Class change sorted

ind_sort = inter[neig_y != test_y[ind], 2].argsort()
inter2 = inter.copy()
inter2 = inter2[neig_y != test_y[ind], :][ind_sort]


variables = inter2.copy()
variables.shape
#variables = np.delete(variables,2,axis=1)

# variables = variables[neig_y==test_y[ind],:]
# variables.shape

#greater than 1
long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()

variables[inter2[:,2]>1,:][long_ind1,:].shape

#smaller
long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()

variables[inter2[:,2]<1,:][long_ind2,:].shape


variables2 = np.concatenate((variables[inter2[:,2]>1,:][long_ind1,:],variables[inter2[:,2]<1,:][long_ind2,:] ))

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




#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# inter[inter[:,1]==0,1] = l
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
#
# plt.show()


weigths = np.sum(intervals, axis=0)


# correct solution:
# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)

# p = softmax(weigths)
# plt.hist(p)
# p[p.argsort()]



dis = []
for i in range(0,1000):
     #print(np.random.choice(range(0,len(ref)),weights= weigths))
    dis.append(random.choices(range(0,len(ref)),weights= weigths)[0])
plt.hist(dis)



center = random.choices(range(0,len(ref)),weights= weigths)[0]

long_intervals =(inter2[:,1]-inter2[:,0])
# np.unique(long_intervals, return_counts=True)[1]

len_inter = random.choices(np.unique(long_intervals, return_counts=True)[0],weights= np.unique(long_intervals, return_counts=True)[1])[0]

inter_new = np.array([np.max([center-int(len_inter/2),0]), np.min([center+int(len_inter/2),len(ref)])])

level = random.choices(np.unique(inter2[:,2], return_counts=True)[0],weights= np.unique(inter2[:,2], return_counts=True)[1])[0]

ts_new =warp(ref,inter_new[0],inter_new[1],level)


distance_matrix2 = np.zeros((train_x.shape[0]))


for i in range(0, train_x.shape[0]):
    distance_matrix2[i] = dtw_distance(ts_new, np.asarray(train_x.values[i,:][0]))

    neig_y = train_y[np.argmin(distance_matrix2)]

print(np.unique(neig_y,return_counts=True))
