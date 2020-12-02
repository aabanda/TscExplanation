from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np
import matplotlib.pyplot as plt
import random
from sktime.distances.elastic import dtw_distance




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


plt.plot(ref)
plt.plot(shift(ref,start,k))






num_neig = 100
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