from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance




# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")


#CBF
#ind =2 #class 1
# ind = 0 #Class 2
# ind = 5 #Class 3
# #
# # Arrowhead
ind=0
# ind=100
# ind=160

ref = test_x.values[ind,:][0].values
plt.plot(ref)
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


plt.plot(ref)
plt.plot(warp(ref, start, end, k))




num_neig = 100
neig = []
inter = np.zeros((num_neig,3))
for i in range(0,num_neig):
     start = random.randint(1,len(ref)-10)
     end = random.randint(start, len(ref))
     if end == len(ref):
         end = 0
     k = np.round(random.uniform(0.7,1.3), decimals=1)
     while k == 1:
         k = np.round(random.uniform(0.7, 1.3), decimals=1)
     inter[i,:] = np.array([start,end,k])
     neig.append(warp(ref, start, end, k))




distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))



neig_y = train_y[np.argmin(distance_matrix,axis=1)]

print(np.unique(neig_y,return_counts=True))




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

clf = LogisticRegression(random_state=0,max_iter=5000)

kf = StratifiedKFold(n_splits=3)
accu = []
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index,:], X[test_index,:]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accu.append(np.sum(pred==y_test)/len(pred))
    print(classification_report(y_test, pred))

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
    accu.append(np.sum(pred==y_test)/len(pred))
    print(classification_report(y_test, pred))


print(np.mean(accu))







clf.fit(X, y)



from sklearn import tree

#tree.plot_tree(clf)
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf, class_names=np.array(["C1","C2","C3"]),impurity=False, fontsize=10)
plt.show()



