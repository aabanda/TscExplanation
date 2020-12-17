from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np
import matplotlib.pyplot as plt
import random
from sktime.distances.elastic import dtw_distance
import numpy as np
from sktime.classifiers.dictionary_based import BOSSEnsemble, BOSSIndividual
from sklearn.metrics import f1_score




train_x, train_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/CBF/CBF_TEST.ts")


# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")

# CBF
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




def scale(ref, k):
    shifted_t = ref*k
    return shifted_t




# plt.plot(ref)
# plt.plot(scale(ref,1.3))






num_neig = 100
neig = []
inter = np.zeros((num_neig,1))
for i in range(0,num_neig):
     k = np.round(random.uniform(0.7,1.3), decimals=1)
     while k == 1:
         k = np.round(random.uniform(0.7, 1.3), decimals=1)
     inter[i,:] = k
     neig.append(scale(ref, k))



distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))


neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))





clf = BOSSEnsemble()
clf.fit(train_x,train_y.astype(int))
y_pred = clf.predict(test_x)
f1_score(test_y.astype(int),np.asarray(y_pred).astype(int),average='weighted')






pred_neig = []
for i in range(0,len(neig)):
    test_neig = np.transpose(np.column_stack((neig[i],neig[i])))
    pred_neig.append(clf.predict(test_neig)[0])

print(np.unique(pred_neig,return_counts=True))





scatter = plt.scatter(range(0,len(inter)),inter,c=neig_y.astype(int))
plt.legend(handles=scatter.legend_elements()[0], labels=['Other class', 'Same class'])


import pandas as pd
# pivot and plot
y_plot = neig_y.copy()
y_plot[y_plot=='1'] = 'Other class'
y_plot[y_plot=='3'] = 'Same class'
df = pd.DataFrame({'level': inter.reshape(1,-1)[0], 'class': y_plot})
ax = df.pivot(columns="class", values="level").plot.hist(bins=5)
ax.set_xlabel("level")
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

print(np.mean(accu))

clf.coef_







from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1)


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

#tree.plot_tree(clf)
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf, class_names=np.array(["C1","C3","C3"]),impurity=False, fontsize=10)
plt.show()