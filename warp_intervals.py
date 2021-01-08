from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance




train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

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
a = 8
p=0.5
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
inter = np.zeros((num_neig,3))
for i in range(0,num_neig):
     start = start_end[i,0]
     end =  start_end[i,1]
     if end == len(ref):
         end = 0
     if start == 0:
         start = 1
     k = np.round(random.uniform(0.7,1.3), decimals=1)
     while k == 1:
         k = np.round(random.uniform(0.7, 1.3), decimals=1)
     inter[i,:] = np.array([start,end,k])
     neig.append(warp(ref, start, end, k))


# plt.hist(inter[:,0])
# plt.hist(inter[:,1]-inter[:,0])

variables = inter.copy()
variables = np.delete(variables,2,axis=1)
variables[variables[:,1]==0,1]=len(ref)


inter_sum_ones = np.zeros((variables.shape[0],len(ref)))
inter_sum_ones[:,:] = 0
for i in range(0,inter_sum_ones.shape[0]):
    inter_sum_ones[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1




plt.hist(variables[:,1].astype(int)-variables[:,0].astype(int))
dydx = np.sum(inter_sum_ones, axis=0)
plt.plot(dydx)
#plt.plot(img*1000/2)














distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))



neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))



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





#Class change sorted

ind_sort = inter[neig_y == test_y[ind], 2].argsort()
inter2 = inter.copy()
inter2 = inter2[neig_y == test_y[ind], :][ind_sort]


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
# plt.plot(dydx)
# plt.plot(img)
# plt.plot(dydx/img)
# len(dydx)
# img[0]= 0.000000001
# len(img)
# dydx = dydx/img
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




















fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
inter[inter[:,1]==0,1] = l
labels = neig_y

co = 0
for lbl in np.unique(labels):
    indices = np.where(labels == lbl)
    ax.scatter(inter[indices,0], inter[indices,1], inter[indices,2], s=50, alpha=0.6, label=str(lbl), cmap='rainbow')
    co = co+1
    print(inter[:,0], inter[:,1], inter[:,2],lbl)

ax.set_xlabel('start')
ax.set_ylabel('end')
ax.set_zlabel('level')
ax.legend()

plt.show()



X= inter
y = neig_y

# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

clf = LogisticRegression(random_state=0,max_iter=5000)

kf = StratifiedKFold(n_splits=3)
accu = []
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index,:], X[test_index,:]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accu.append(f1_score(y_test.astype(int),pred.astype(int),average="weighted"))
    print(classification_report(y_test, pred))

print(np.mean(accu))

clf.coef_







from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()


kf = StratifiedKFold(n_splits=3)
accu = []
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index,:], X[test_index,:]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accu.append(f1_score(y_test.astype(int),pred.astype(int),average="weighted"))
    print(classification_report(y_test, pred))


print(np.mean(accu))




import pandas as pd
X1 = pd.DataFrame(X)

clf.fit(X1, y)



from sklearn import tree

tree.plot_tree(clf,feature_names=np.array(['start','end','level']))
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf, class_names=np.array(["C1","C3"]),impurity=False, fontsize=10)
#tree.plot_tree(clf, class_names=np.array(["C1","C2","C3"]),impurity=False, fontsize=10)
plt.show()


L = 128
import numpy as np
def f(x):
    suma = np.sum(1/(L-np.asarray(range(0,x))))
    return ((L-x)/L)*suma


img = np.zeros(L)
for i in range(0,L):
    img[i] = f(i)

import matplotlib.pyplot as plt

plt.plot(img)
np.argmax(img)
np.sum(img)


