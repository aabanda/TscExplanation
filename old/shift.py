from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np
import matplotlib.pyplot as plt
import random
from sktime.distances.elastic import dtw_distance




train_x, train_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/CBF/CBF_TEST.ts")

#
# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")

#CBF
# ind =2 #class 1
# ind = 0 #Class 2
ind = 5 #Class 3

#ArrowHead
# ind=0
# ind=100
# ind=160

ref = test_x.values[ind,:][0].values
# plt.plot(ref)
test_y[ind]




def shift(ref, start, k):
    l = len(ref)
    shifted_t = np.zeros(l+k)
    shifted_t[range(start)] = ref[range(start)]
    shifted_t[range(start+k,len(shifted_t))]= ref[range(start,l)]


    before = ref[start-1]
    after = ref[start]

    for t in range(start,start+k):
        mean =(((start+k-t))* before + (k-(start+k-t)) * after)/k
        std = np.abs(before - after)/5
        shifted_t[t]= np.random.normal(mean,std,1)

    return shifted_t


start = 15
k = 20

start = 60
k = 30


# plt.plot(ref)
# plt.plot(shift(ref,start,k))
#
#




num_neig = 500
neig = []
inter = np.zeros((num_neig,2))
for i in range(0,num_neig):
     start = random.randint(1,len(ref)-10)
     k = random.randint(1,int(len(ref)*0.3))
     inter[i,:] = np.array([start,k])
     neig.append(shift(ref, start, k))



distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))


neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))




plt.scatter(inter[neig_y=='1',0],inter[neig_y=='1',1],color='r',label="Other class")
plt.scatter(inter[neig_y=='3',0],inter[neig_y=='3',1],color='b',label="Same class")
plt.xlabel("start")
plt.ylabel("level")
plt.legend()


variables = inter.copy()
variables = variables[neig_y!=test_y[ind],:]
variables[:,1] = variables[:,0]+variables[:,1]


import matplotlib as mpl
cmap = plt.cm.get_cmap("jet")
norm = mpl.colors.SymLogNorm(2, vmin=inter[neig_y==test_y[ind],:][:,1].min(), vmax=inter[neig_y==test_y[ind],:][:,1].max())
norm = plt.Normalize(inter[neig_y==test_y[ind],:][:,1].min(), inter[neig_y==test_y[ind],:][:,1].max())


sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])








intervals = np.zeros((variables.shape[0],variables[:,1].max().astype(int)))
intervals[:,:] = np.nan
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = i
    plt.plot(range(0, intervals.shape[1]), intervals[i, :], color=cmap(norm(inter[neig_y==test_y[ind],:][i,1].astype(int))), label=inter[neig_y==test_y[ind],:][i,1].astype(int))
#plt.legend(loc=3)

cbar = plt.colorbar(sm, ticks=inter[neig_y==test_y[ind],:][:,1], format=mpl.ticker.ScalarFormatter(),
                     fraction=0.1, pad=0)








intervals = np.zeros((variables.shape[0],variables[:,1].max().astype(int)))
intervals[:,:] = np.nan
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
    #plt.plot(range(0, intervals.shape[1]), intervals[i, :])


import matplotlib as mpl
cmap = plt.cm.get_cmap("jet")
# norm = mpl.colors.SymLogNorm(2, vmin=inter[neig_y==test_y[ind],:][:,1].min(), vmax=inter[neig_y==test_y[ind],:][:,1].max())
# norm = plt.Normalize(inter[neig_y==test_y[ind],:][:,1].min(), inter[neig_y==test_y[ind],:][:,1].max())


sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])


ind_long= (variables[:,1]-variables[:,0]).argsort()
#ind_long= variables[:,0].argsort()
intervals = intervals[ind_long,:]
long_sort = (variables[:,1]-variables[:,0])[ind_long]
norm = plt.Normalize(long_sort.min(),long_sort.max())

for i in range(0,intervals.shape[0]):
    intervals[i,np.where(intervals[i,:]==1)[0]] = i
    plt.plot(range(0, intervals.shape[1]), intervals[i, :],color=cmap(norm(long_sort.astype(int)[i])))

cbar = plt.colorbar(sm, ticks=long_sort, format=mpl.ticker.ScalarFormatter(),
                     fraction=0.1, pad=0)







intervals = np.zeros((variables.shape[0], variables[:,1].max().astype(int)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
intervals = intervals[:,range(0,len(ref))]
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