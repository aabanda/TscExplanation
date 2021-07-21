from sktime.utils.data_io import load_from_tsfile_to_dataframe
from functions.consensus import compute_consensus,compute_evaluation

input_dir = "data/Univariate2018"
db_name = "GunPoint"
transformation = "slice"
level = 0.7
classifier = "st"

train_x, train_y = load_from_tsfile_to_dataframe("%s/%s/%s_TRAIN.ts" % (input_dir, db_name, db_name))
test_x, test_y = load_from_tsfile_to_dataframe("%s/%s/%s_TRAIN.ts" % (input_dir, db_name, db_name))

c1, c2, z = compute_consensus(db_name, train_x,train_y,test_x,test_y,classifier,transformation,level)

compute_evaluation(c1, c2, z)
