from sktime.utils.data_io import load_from_tsfile_to_dataframe
from functions.neighbours import create_neighbours, label_neighbours, compute_robustness
from functions.plot_explanation import plot_colormap

input_dir = "../Univariate2018"
db_name = "GunPoint"
ts_index = 3
transformation = "slice"
level = 0.7
classifier = "dtw"

train_x, train_y = load_from_tsfile_to_dataframe("%s/%s/%s_TRAIN.ts" % (input_dir, db_name, db_name))
test_x, test_y = load_from_tsfile_to_dataframe("%s/%s/%s_TRAIN.ts" % (input_dir, db_name, db_name))


ts = test_x.iloc[ts_index,:][0].values
ts_y = test_y[ts_index]

#Sample random intervals ana create neighbours (in interval and time domain)
interval, neig = create_neighbours(ts, transformation, num_neig=500)


#Label neighbours:
neig_y = label_neighbours(neig, classifier, train_x, train_y)


#High-level explanation(Robustness) and weigths:
r, w = compute_robustness(ts, ts_y, transformation, level, interval, neig_y)

#Low-level explanation visualization
plot_colormap(ts,w)