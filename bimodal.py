from pylab import *
from scipy.optimize import curve_fit
import random
from sktime.distances.elastic import dtw_distance




data=concatenate((normal(1,.2,5000),normal(2,.2,2500)))
y,x,_=hist(data,100,alpha=.3,label='data')

x=(x[1:]+x[:-1])/2 # for len(x)==len(y)



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

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)




expected=(1,.2,250,2,.2,125)
params,cov=curve_fit(bimodal,x,y,expected)


ts = bimodal(x,*params)
plot(range(0,100),bimodal(x,*params),color='red')



same_ind = range(20,40)
other_ind = range(60,80)

num_class1= 100
num_class2= 100

class1 = []
for class1_ind in range(0,num_class1):
    k = np.round(random.uniform(0.7, 1.3), decimals=1)
    while k == 1:
        k = np.round(random.uniform(0.7, 1.3), decimals=1)
    class1.append(warp(ts,20,40,k))

import matplotlib.pyplot as plt
plt.plot(ts)
plt.plot(class1[5])


other_reference = warp(ts,60,80,1.3)
plt.plot(ts)
plt.plot(other_reference)




class2 = []
for class2_ind in range(0,num_class2):
    k = np.round(random.uniform(0.7, 1.3), decimals=1)
    while k == 1:
        k = np.round(random.uniform(0.7, 1.3), decimals=1)
    class2.append(warp(other_reference,20,40,k))

import matplotlib.pyplot as plt
plt.plot(other_reference)
plt.plot(class2[5])




labels1 = np.zeros((num_class1))
labels1[labels1==0]=1
labels2 = np.zeros((num_class2))
labels2[labels2==0]=2

y = np.concatenate((labels1,labels2))
y_labels = y.copy()
len(y)


len(class1)
len(class2)

data = class1
for i in range(0,len(class2)):
    data.append(class2[i])

len(data)








distance_matrix = np.zeros((num_class1+num_class2,num_class1+num_class2))
distance_matrix.shape

for i in range(0, distance_matrix.shape[0]):
    for j in range(0, distance_matrix.shape[0]):
        distance_matrix[i,j] = dtw_distance(data[i], data[j])




from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True)
accu = []
for train_index, test_index in kf.split(distance_matrix,y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    y_train, y_test = y[train_index], y[test_index]

    distance_matrix_fold = distance_matrix[test_index, :]
    distance_matrix_fold = distance_matrix_fold[:, train_index]


    pred = []
    for i in range(0,len(test_index)):
        ind = np.argmin(distance_matrix_fold[i,:])
        pred.append(y_train[train_index[ind]])

    accu.append((np.sum(pred==y_test)/len(y_test)))

np.mean(accu)



ind_explanation=20
# y_labels[test_index]
y_labels[test_index[ind_explanation]]
num_neig = 500
ref = data[test_index[ind_explanation]]
plt.plot(ref)


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
p= 0.3
b = a*(1-p)/p
start_end = np.zeros((num_neig,2))
for i in range(0,num_neig):
    start_end[i,:]=unwrapcircle(betaprime.rvs(a, b, size=1))


start_end[start_end<=0]=0
start_end = start_end*len(ref)
start_end = start_end.astype(int)


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




distance_matrix_neig = np.zeros((len(neig),len(train_index)))

for i in range(0, len(neig)):
    for j in range(0, len(train_index)):
        distance_matrix_neig[i,j] = dtw_distance(neig[i], data[train_index[j]])
    print(i)


distance_matrix_neig.shape
neig_y = y[train_index][np.argmin(distance_matrix_neig,axis=1)]
print(np.unique(neig_y,return_counts=True))







inter[inter[:,1]==0,1]=len(ref)



#Same class
ind_sort = inter[neig_y== y_labels[test_index[ind_explanation]], 2].argsort()
inter2 = inter.copy()
inter2 = inter2[neig_y ==y_labels[test_index[ind_explanation]], :][ind_sort]
variables = inter2.copy()




#Other class
ind_sort = inter[neig_y!= y_labels[test_index[ind_explanation]], 2].argsort()
inter2 = inter.copy()
inter2 = inter2[neig_y != y_labels[test_index[ind_explanation]], :][ind_sort]
variables = inter2.copy()

#variables = np.delete(variables,2,axis=1)

# variables = variables[neig_y==test_y[ind],:]
# variables.shape


#same class
#greater than 1
long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
largest_indices1 =long_ind1
#smaller
long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
largest_indices2 =long_ind2



#other class
#greater than 1
long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
largest_indices1 = long_ind1[::-1]
largest_indices1 = long_ind1
#smaller
long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
largest_indices2 = long_ind2[::-1]
largest_indices2 = long_ind2





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


# plt.plot(ref)
# plt.plot(neig[11])

intervals = np.zeros((variables.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
np.sum(intervals, axis=0)


# point_other = np.sum(intervals, axis=0)
# point_same = np.sum(intervals, axis=0)

# point_other = np.sum(intervals, axis=0)/(500*p)
# point_same = np.sum(intervals, axis=0)/(500*p)
#
# point_other-point_same
#

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x = range(0,len(ref))
y = ref
# dydx = np.abs(clf.coef_)[0]
dydx = np.sum(intervals, axis=0)/(500*p)
# dydx = np.sum(intervals, axis=0)
# dydx = point_other-point_same
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
axs.set_ylim(0, 250)
plt.show()
