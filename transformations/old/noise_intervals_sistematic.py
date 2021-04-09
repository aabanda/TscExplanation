from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import matplotlib.pyplot as plt
import random
from sktime.distances.elastic import dtw_distance
import numpy as np
from sklearn.metrics import f1_score



train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")


train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")


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




def noise(ref, start, end, k):
    shifted_t = ref.copy()
    noise = np.random.normal(0, np.abs(np.max(ref)-np.min(ref))*k/100 , len(range(start,end)))
    shifted_t[range(start,end)] = ref[range(start,end)]+noise
    return shifted_t



#
# plt.plot(ref)
# plt.plot(noise(ref,2,20,1))
# plt.plot(noise(ref,2,20,9))



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



num_neig =500
neig = []
inter = np.zeros((num_neig*5,3))
count = 0
for i in range(0,num_neig):
     start = start_end[i, 0]
     end = start_end[i, 1]
     for k in [1,3,5,7,9]:
         inter[count, :] =  np.array([start,end,k])
         neig.append(noise(ref, start,end,k))
         count = count + 1



distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))


neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))
















inter[inter[:,1]==0,1]=len(ref)

variables = inter.copy()
#variables = np.delete(variables,2,axis=1)
variables.shape
variables = variables[neig_y==test_y[ind],:]
variables = variables[neig_y!=test_y[ind],:]



intervals = np.zeros((variables.shape[0],len(ref)))
intervals[:,:] = np.nan
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = i
    #plt.plot(range(0, len(ref)), intervals[i, :])





import matplotlib as mpl
cmap = plt.cm.get_cmap("jet")
# norm = mpl.colors.SymLogNorm(2, vmin=inter[neig_y==test_y[ind],:][:,1].min(), vmax=inter[neig_y==test_y[ind],:][:,1].max())
# norm = plt.Normalize(inter[neig_y==test_y[ind],:][:,1].min(), inter[neig_y==test_y[ind],:][:,1].max())



ind_long= (variables[:,1]-variables[:,0]).argsort()
ind_long= variables[:,2].argsort()
#ind_long= variables[:,0].argsort()
variables = variables[ind_long,:]
# long_sort = (variables[:,1]-variables[:,0])[ind_long]
norm = plt.Normalize(variables[:,2].min(),variables[:,2].max())
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])




intervals = np.zeros((variables.shape[0],len(ref)))
intervals[:,:] = np.nan
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = i
    plt.plot(range(0, intervals.shape[1]), intervals[i, :],color=cmap(norm(variables[i,2].astype(int))))

cbar = plt.colorbar(sm, ticks=variables[:,2], format=mpl.ticker.ScalarFormatter(),
                     fraction=0.1, pad=0)



















import matplotlib.patches as mpatches
# red_patch = mpatches.Patch(color='red', label='level > 1')
# plt.legend(handles=[red_patch],loc='upper left')



intervals = np.zeros((variables.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
np.sum(intervals, axis=0)




from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x = range(0,len(ref))
y = ref
# dydx = np.abs(clf.coef_)[0]
dydx = np.sum(intervals, axis=0)/(500*p)
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





















scatter = plt.scatter(range(0,len(inter)),inter,c=neig_y.astype(int))
plt.legend(handles=scatter.legend_elements()[0], labels=['Other class', 'Same class'])


import pandas as pd
# pivot and plot
y_plot = neig_y.copy()
y_plot[y_plot=='3'] = 'Same class'
y_plot[y_plot=='1'] = 'Other class'

df = pd.DataFrame({'level': inter.reshape(1,-1)[0], 'class': y_plot})
ax = df.pivot(columns="class", values="level").reindex(columns=["S","O"]).plot.hist(bins=9, alpha=0.5)
ax.set_xlabel("level")
ax.legend()




clf = BOSSEnsemble()
clf.fit(train_x,train_y.astype(int))
y_pred = clf.predict(test_x)
f1_score(test_y.astype(int),np.asarray(y_pred).astype(int),average='weighted')






pred_neig = []
for i in range(0,len(neig)):
    test_neig = np.transpose(np.column_stack((neig[i],neig[i])))
    pred_neig.append(clf.predict(test_neig)[0])

print(np.unique(pred_neig,return_counts=True))







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

print(np.mean(accu))

clf.coef_






from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)


X= inter
y = neig_y

kf = StratifiedKFold(n_splits=2)
accu = []
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index,:], X[test_index,:]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accu.append(f1_score(y_test, pred,average="weighted"))


print(np.mean(accu))



clf.fit(X, y)



from sklearn import tree
inter[neig_y=="1"]
neig_y[inter.reshape(1,-1)[0]==3]
tree.plot_tree(clf,feature_names=np.array(['level']),class_names=np.array(["C1","C3"]))
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf, class_names=np.array(["C1","C3","C3"]),impurity=False, fontsize=10)
plt.show()




import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 2
plot_colors = "br"
plot_step = 0.02

# Load data

# X= inter_seg
# y = neig_y_seg

# np.savetxt("inter_noise_3000.txt", inter)
# np.savetxt("neig_y_seg_noise_3000.txt", neig_y.astype(int))
#

inter = np.loadtxt("inter_noise_3000.txt")
neig_y = np.loadtxt("neig_y_seg_noise_3000.txt")

neig_y[neig_y==3]=0
np.unique(neig_y,return_counts=True)

col_names = ['start', 'end', 'noise level']
labels_names =['same', 'other']

for pairidx, pair in enumerate([[0, 1], [0, 2], [1, 2]]):
    # We only take the two corresponding features

    X = inter[:, pair]
    y = neig_y

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    np.unique(y, return_counts=True)
    # Standardize
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # X = (X - mean) / std
    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(1, 3, pairidx + 1)
    if pair[1]==2:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    else:
        x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:, 1].max() + 1
    # z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    # xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, plot_step),
    #                      np.arange(y_min, y_max, plot_step), np.arange(z_min, z_max, plot_step))
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(col_names[pair[0]])
    plt.ylabel(col_names[pair[1]])
    plt.axis("tight")

    sizes_arr = [40,10]
    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)[0]
        print(len(idx))
        plt.scatter(X[idx, 0], X[idx, 1], c=color, s=sizes_arr[i], label=labels_names[i],
                    cmap=plt.cm.Paired)
    plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()
