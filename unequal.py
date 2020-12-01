from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from sktime.classifiers.dictionary_based import BOSSEnsemble, BOSSIndividual
from sklearn.metrics import f1_score
import pandas as pd
from sktime.classifiers.shapelet_based import ShapeletTransformClassifier

# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")
# ind_ref = 0
# ref = test_x.values[ind_ref,:][0]
# test_y[ind_ref]
#
clf = BOSSEnsemble()
#clf = BOSSIndividual(window_size=50, word_length=8, alphabet_size=4,norm=False)
clf.fit(train_x,train_y.astype(int))
y_pred = clf.predict(test_x)
f1_score(test_y.astype(int),np.asarray(y_pred).astype(int),average='weighted')






pred_neig = []
for i in range(0,len(neig)):
    test_neig = np.transpose(np.column_stack((neig[i],neig[i])))
    pred_neig.append(clf.predict(test_neig)[0])
    #print(pred_neig[-1])

print(np.unique(pred_neig,return_counts=True))






clf = ShapeletTransformClassifier(time_contract_in_mins=5)
clf.fit(train_x,train_y)
y_pred = clf.predict(test_x)
f1_score(test_y.astype(int),y_pred.astype(int),average='weighted')




test_x2 = pd.DataFrame(test_x.values[range(0,2),:])

pred_neig = []
for i in range(0,len(neig)):
    test_x2.iloc[0,:][0] = pd.Series(neig[i])
    pred_neig.append(clf.predict(pd.DataFrame(test_x2.iloc[0,:]))[0])
    #print(pred_neig[-1])
#
# plt.plot(test_x2.iloc[0,:][0])
# plt.plot(neig[i])
# clf.predict(pd.DataFrame(test_x2.iloc[0,:]))
print(np.unique(pred_neig,return_counts=True))






plt.hist(pred_neig)
print(inter)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
inter[inter[:,1]==0,1] = l
labels = np.asarray(pred_neig).astype(int)

co = 0
for lbl in np.unique(labels):
    indices = np.where(labels == lbl)
    ax.scatter(inter[indices,0], inter[indices,1], inter[indices,2], s=50, alpha=0.6, label=str(lbl), cmap='rainbow')

ax.set_xlabel('start')
ax.set_ylabel('end')
ax.set_zlabel('level')
ax.legend()

plt.show()
