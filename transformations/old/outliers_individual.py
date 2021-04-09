from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np
import matplotlib.pyplot as plt
import random
from sktime.distances.elastic import dtw_distance
import numpy as np
from sktime.classifiers.dictionary_based import BOSSEnsemble, BOSSIndividual
from sklearn.metrics import f1_score
from matplotlib.collections import LineCollection



train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")


# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")

#CBF
# ind =2 #class 1
# ind = 0 #Class 2
ind = 5 #Class 3

#ArrowHead
# ind=0
# ind q=100
# ind=160

ref = test_x.values[ind,:][0].values
# plt.plot(ref)
test_y[ind]




# def outliers(ref, num_outliers, k):
#     add = np.zeros(len(ref))
#     positions = []
#     [positions.append(random.randint(0, len(ref)-1)) for l in range(0,num_outliers)]
#     for j in positions:
#         add[j] = np.random.normal(0, np.abs(np.max(ref)-np.min(ref))*k/10 , 1)[0]
#     shifted_t = ref+add
#     return shifted_t

def outliers(ref, positions):
    pos_ind = np.where(positions>0)[0]
    shifted_t = ref.copy()
    for j in pos_ind:
        shifted_t[j] = shifted_t[j]+ np.random.normal(0, np.abs(np.max(ref)-np.min(ref))*positions[j]/10 , 1)[0]
    return shifted_t



# plt.plot(ref)
# plt.plot(shifted_t)
# plt.plot(ref)
# plt.plot(outliers(ref,5,120))
#



num_neig = 10000
neig = []
inter = np.zeros((num_neig,len(ref)))
for i in range(0,num_neig):
     num_outliers = int(len(ref) * random.randint(1,5) / 100)
     num_outliers = 1
     positions = np.zeros(len(ref))
     outliers_ind = random.sample(range(0, len(ref)), num_outliers)
     for out_ind in range(num_outliers):
        positions[outliers_ind[out_ind]] =  random.randrange(5, 7)
     inter[i,:] =positions
     neig.append(outliers(ref, positions))


plt.plot(ref)
plt.plot(neig[0])
# kk=3
# plt.plot(neig[kk])
# inter[kk,:]



inter.shape

inter[inter>0]=1

plt.plot(np.sum(inter,axis=0))



distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))


neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))


variables = np.column_stack((inter, neig_y.astype(int)))


# Analisis univariado:  num outliers
other = variables[variables[:,-1]!=test_y[ind].astype(int),:][:,:-1]
other[other>0]=1
plt.hist(np.sum(other, axis=1),bins=4)
plt.bar(np.unique(np.sum(other, axis=1)),np.unique(np.sum(other, axis=1), return_counts=True)[1])
plt.xticks(np.unique(np.sum(other, axis=1)), np.unique(np.sum(other, axis=1)))

same = variables[variables[:,-1]==test_y[ind].astype(int),:][:,:-1]
same[same>0]=1
plt.hist(np.sum(same, axis=1),bins=4)




# Analisis univariado:  level
other = variables[variables[:,-1]!=test_y[ind].astype(int),:][:,:-1]
np.mean(other[other>0])
np.std(other[other>0])


same = variables[variables[:,-1]==test_y[ind].astype(int),:][:,:-1]
np.mean(same[same>0])
np.std(same[same>0])



inter2 = inter.copy()
# a = inter2[:,:-1]
a = inter2
a.shape
a[a>0]=1

plt.plot(np.sum(a,axis=0))

np.where(variables[:,-1]==test_y[ind].astype(int))[0]
same = variables[variables[:,-1]==test_y[ind].astype(int),:-1]



np.std(same)
same[same>0] = 1
np.sum(same,axis=0)





x = range(0,len(ref))
y = ref
dydx = np.sum(same,axis=0)/np.sum(a,axis=0)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
fig, axs = plt.subplots()
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='jet', norm=norm)
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs.add_collection(lc)
fig.colorbar(line, ax=axs)
axs.set_xlim(0, len(x))
axs.set_ylim(-2.5, 2.5)
axs.set_ylim(-2, 4)
plt.show()




for i in range(0, len(np.where(variables[:,-1]!=test_y[ind].astype(int))[0])):
    plt.plot(neig[np.where(variables[:,-1]!=test_y[ind].astype(int))[0][i]])

a =variables[variables[:,-1]!=test_y[ind].astype(int),:-1]
other = variables[variables[:,-1]!=test_y[ind].astype(int),:-1]
other[other>0] = 1
np.sum(other,axis=0)


np.mean(other)
np.std(other)



x = range(0,len(ref))
y = ref
dydx =np.sum(other, axis=0)/np.sum(a,axis=0)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
fig, axs = plt.subplots()
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='jet', norm=norm)
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs.add_collection(lc)
fig.colorbar(line, ax=axs)
axs.set_xlim(0, len(x))
axs.set_ylim(-2.5, 2.5)
axs.set_ylim(-2, 4)
plt.show()






x = range(0,len(ref))
y = ref
dydx = (np.sum(other,axis=0)/np.sum(a,axis=0))-(np.sum(same, axis=0)/np.sum(a,axis=0))

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
fig, axs = plt.subplots()
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='jet', norm=norm)
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs.add_collection(lc)
fig.colorbar(line, ax=axs)
axs.set_xlim(0, len(x))
axs.set_ylim(-2.5, 2.5)
axs.set_ylim(-2, 4)
plt.show()











inter[:,-1].shape
inter2 = inter[:,:-1]
variables = inter2.copy()
variables = variables[neig_y!=test_y[ind],:]
variables.shape
intervals = np.zeros((variables.shape[0],len(ref)))
intervals[:,:] = np.nan
for i in range(0,intervals.shape[0]):
    intervals[i,np.where(variables[i,:]==1)[0]] = i
    plt.plot(range(0, len(ref)), intervals[i, :])






from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x = range(0,len(ref))
y = ref
# dydx = np.abs(clf.coef_)[0]
dydx = np.sum(variables, axis=0)
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









plt.plot(variables.sum(axis=0))
plt.scatter(inter[neig_y=='1',0],inter[neig_y=='1',1],color='r',label="Other class")
plt.scatter(inter[neig_y=='3',0],inter[neig_y=='3',1],color='b',label="Same class")
plt.xlabel("start")
plt.ylabel("level")
plt.legend()






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







from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)


kf = StratifiedKFold(n_splits=3)
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

#tree.plot_tree(clf)
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf, class_names=np.array(["C1","C2","C3"]),impurity=False, fontsize=10)
plt.show()