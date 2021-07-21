import numpy as np
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from functions.neighbours import create_neighbours, label_neighbours, compute_robustness
from functions.plot_explanation import plot_colormap

# 1. Select transformation:

#Warp
ts_index = 52
classifier = "dtw"
db_name = "GunPoint"
transformation = "warp"
level = 0.7


#Scale
ts_index = 1
classifier = "dtw"
db_name = "GunPoint"
level = 1.2
transformation = "scale"


#Noise
ts_index = 6
classifier = "boss"
db_name = "Coffee"
level = 9
transformation = "noise"


#Slice
ts_index = 21
classifier = "st"
db_name = "Coffee"
transformation = "slice"






# 2. Load data:

train_x, train_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TRAIN.ts" % (db_name,db_name))
test_x, test_y = load_from_tsfile_to_dataframe("data/Univariate2018/%s/%s_TEST.ts" % (db_name,db_name))
ts = test_x.iloc[ts_index,:][0].values





# 3. Explanation:

# Option 1: plot explanation with pre-computed weights
if transformation=="slice":
    w = np.loadtxt('data/neig/%s/%s/%s/weights_%d.txt' % (db_name, transformation, classifier, ts_index))
else:
    w = np.loadtxt('data/neig/%s/%s/%s/weights%0.1f_%d.txt' % (db_name, transformation, classifier, level, ts_index))
plot_colormap(ts, w)



# Option 2: compute weights and plot explanation
ts_y = test_y[ts_index]
interval, neig = create_neighbours(ts, transformation, num_neig=500)
neig_y = label_neighbours(neig, classifier, train_x, train_y)
r, w = compute_robustness(ts, ts_y, transformation, level, interval, neig_y)
plot_colormap(ts, w)
