from sktime.utils.data_io import load_from_tsfile_to_dataframe
import numpy as np
import matplotlib.pyplot as plt

# Warp
db = "GunPoint"
train_x, train_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TEST.ts" % (db, db))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y == i)[0])

plt.figure()
plt.plot(train_x.values[ind_class[0][18], :][0].values, label="Gun class", linewidth=3)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=15)
plt.tight_layout()

plt.figure()
plt.plot(train_x.values[ind_class[1][1], :][0].values, label="Point class", linewidth=3, color="C1")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=15)
plt.tight_layout()

# Scale
db = "Adiac"
train_x, train_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TEST.ts" % (db, db))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y == i)[0])

plt.figure(figsize=(8, 6))
plt.plot(train_x.values[ind_class[0][0], :][0].values, label="Class 1", linewidth=3, )
plt.plot(train_x.values[ind_class[1][0], :][0].values, label="Class 2", linewidth=3, )
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()


# Noise
db = "ECG200"
train_x, train_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TEST.ts" % (db, db))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y == i)[0])

plt.figure(figsize=(8, 6))
plt.plot(train_x.values[ind_class[0][6], :][0].values, label="Normal", linewidth=3)
plt.plot(train_x.values[ind_class[1][2], :][0].values, label="Myocardial", linewidth=3)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()




# Slice
db = "AllGestureWiimoteY"
train_x, train_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TRAIN.ts" % (db, db))
test_x, test_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TEST.ts" % (db, db))

ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y == i)[0])

plt.figure(figsize=(8, 6))
plt.plot(train_x.values[ind_class[6][1], :][0].values, label="Class 7", linewidth=3)
plt.plot(train_x.values[ind_class[4][0], :][0].values, label="Class 5", linewidth=3)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.ylim(-1, 5)
plt.tight_layout()

