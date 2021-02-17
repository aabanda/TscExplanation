from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

X = inter
y = neig_y

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
#
#
# L = 128
# import numpy as np
# def f(x):
#     suma = np.sum(1/(L-np.asarray(range(0,x))))
#     return ((L-x)/L)*suma
#
#
# img = np.zeros(L)
# for i in range(0,L):
#     img[i] = f(i)
#
# import matplotlib.pyplot as plt
#
# plt.plot(img)
# np.argmax(img)
# np.sum(img)







import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 2
plot_colors = "br"
plot_step = 0.02

# Load data

X= inter_seg
y = neig_y_seg




# np.savetxt("inter_warp_3000.txt", inter_seg)
# np.savetxt("neig_y_seg_warp_3000.txt", neig_y_seg.astype(int))


inter = np.loadtxt("inter_warp_3000.txt")
neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")

inter.shape


neig_y[neig_y==3]=0
col_names = ['start', 'end', 'warp level']
labels_names =['same', 'other']

# np.sum(inter[:,1]==0)
# inter[np.where(inter[:,1]<20)[0],:]
inter[inter[:,1]==0,1]=len(ref)
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












