#from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import f1_score
from sktime.classification.dictionary_based import BOSSEnsemble
import pandas as pd


len_shape1= 5

class1 = np.random.rand(100,150)
class2 = np.random.rand(100,150)


start1 = 20
for i in range(0,class1.shape[0]):
    class1[i,np.arange(start1,start1+20)] = np.random.rand(1)*3

start2 = 100
for i in range(0,class1.shape[0]):
    class2[i,np.arange(start2,start2+20)] = np.random.rand(1)*3

plt.plot(class1[0,:])
plt.plot(class1[1,:])
plt.plot(class1[2,:])

plt.plot(class2[0,:])
plt.plot(class2[1,:])
plt.plot(class2[2,:])

#
# len_shape1= 25
# class2 = np.random.rand(100,150)
# for i in range(0,class1.shape[0]):
#     start = random.randrange(1,class1.shape[1]-50)
#     class2[i,np.arange(start,start+50)] = 3
#
# plt.plot(class2[4,:])


y = np.repeat([0,1],[100,100])
X = np.concatenate([class1, class2])




#Classification with  DTW
distance_matrix = np.zeros((X.shape[0], X.shape[0]))
for i in range(0, distance_matrix.shape[0]):
    for j in range(0, distance_matrix.shape[0]):
        distance_matrix[i,j] = dtw_distance(X[i,:], X[j,:])

kf = StratifiedKFold(n_splits=5, shuffle=True)
accu = []
for train_index, test_index in kf.split(X,y):
    y_train, y_test = y[train_index], y[test_index]
    distance_matrix_fold = distance_matrix[test_index, :]
    distance_matrix_fold = distance_matrix_fold[:, train_index]

    pred = []
    for i in range(0,len(test_index)):
        ind = np.argmin(distance_matrix_fold[i,:])
        pred.append(y_train[ind])
    accu.append(f1_score(y_test, pred, average="weighted"))

np.mean(accu)


#Classification with BOSS
X = np.concatenate([class1, class2])

tsf = BOSSEnsemble(max_ensemble_size=100)
X_data = pd.DataFrame(X)
X = pd.DataFrame()
X["dim_0"] = [pd.Series(X_data.iloc[x, :]) for x in range(len(X_data))]
accu = []
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    tsf.fit(X_train, y_train)
    y_pred = tsf.predict(X_test)
    accu.append(f1_score(y_test.astype(int), y_pred.astype(int), average="weighted"))

print(np.mean(accu))



y = np.repeat([0,1],[100,100])
X = np.concatenate([class1, class2])

kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index,:], X[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
    print("A")

X = np.concatenate([class1, class2])
ind =38
ref= X_test[ind,:]
plt.plot(ref)
y_ref= y_test[ind]
#plt.plot(X[test_index[ind],:])
print(y[test_index[ind]])







def shift(ref, shift_prefix, shift_sufix):
    shifted_t = ref[shift_prefix:(shift_sufix)]
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
    return intersection([[0,1],[u, u+z]])




from scipy.stats import betaprime
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



for i in range(num_neig):
    while (start_end[i, 1]- start_end[i, 0])<10 :
        start_end[i, :] = unwrapcircle(betaprime.rvs(a, b, size=1))
        start_end[i,:][start_end[i,:] <= 0] = 0
        start_end[i,:] = start_end[i,:] * len(ref)
        start_end [i,:]= start_end[i,:].astype(int)

for i in range(num_neig):
    while (start_end[i, 1]- start_end[i, 0])<20 :
        start_end[i, :] = unwrapcircle(betaprime.rvs(a, b, size=1))
        start_end[i,:][start_end[i,:] <= 0] = 0
        start_end[i,:] = start_end[i,:] * len(ref)
        start_end [i,:]= start_end[i,:].astype(int)





num_neig = 500
neig = []
count=0
inter = np.zeros((num_neig,2))
for i in range(0,num_neig):
    shi_pre = start_end[i,0]
    shi_suf = start_end[i,1]
    inter[i,:] = np.array([shi_pre,shi_suf])
    neig.append(shift(ref, shi_pre, shi_suf))
    count = count+1




#DTW
distance_matrix = np.zeros((len(neig),X_train.shape[0]))
for i in range(0, len(neig)):
    for j in range(0, X_train.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(X_train[j,:]))


neig_y = y_train[np.argmin(distance_matrix,axis=1)]
print(np.unique(neig_y,return_counts=True))



#BOSS


X_data = pd.DataFrame(X_train)
X = pd.DataFrame()
X["dim_0"] = [pd.Series(X_data.iloc[x, :]) for x in range(len(X_data))]

lens=[]
for i in range(num_neig):
    lens.append(len(neig[i]))
np.min(lens)


clf = BOSSEnsemble(max_ensemble_size=100)
clf.fit(X, y_train)

neig_y = []
for i in range(len(neig)):
    neig_y.append(clf.predict(neig[i].reshape(1, 1, -1))[0])

neig_y = np.ravel(neig_y)
print(np.unique(neig_y, return_counts=True))







#SEGUMOS
same_int = inter[neig_y==y_test[ind].astype(int)]
other_int = inter[neig_y!=y_test[ind].astype(int)]

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
ran = range(1,60,2)
num = []
for threshold in ran:
    thresh_per_same = []
    for i in range(0,dist_int.shape[1]):
        # min_per_same.append(np.argmin(dist_int[i,:]))
        # perc_per_same.append(np.where(dist_int[i,:]<np.percentile(dist_int,q=2))[0])
        thresh_per_same.append(np.where(dist_int[:,i]<threshold)[0])
    k = np.concatenate(thresh_per_same, axis=0)
    num.append(len(np.unique(k, return_counts=True)[0]))



plt.plot(ran,num)

np.column_stack((ran,num))
y_test[ind]





#por columna
thresh_per_same = []
for i in range(0,dist_int.shape[1]):
    thresh_per_same.append(np.where(dist_int[:,i]<5)[0])
k = np.concatenate(thresh_per_same, axis=0)

len(np.unique(k))

other_int.shape
same_int.shape


same_index = np.setdiff1d(np.arange(same_int.shape[0]), np.unique(k))
len(same_index)





variables = same_int[same_index,:]

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

    axs.set_xlim(0, len(x))
    axs.set_ylim(-2.5, 2.5)
    axs.set_ylim(-2, 4)
    plt.show()



plot_colormap(ref,np.sum(intervals, axis=0))