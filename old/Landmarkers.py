from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier
from sktime.transformers.shapelets import ContractedShapeletTransform
from sklearn.pipeline import Pipeline
from sktime.classifiers.dictionary_based import  BOSSIndividual
from sktime.classifiers.compose import TimeSeriesForestClassifier
from sktime.classifiers.interval_based import TimeSeriesForest
import time
import random
from sklearn.cluster import AgglomerativeClustering
from random import sample



from sys import argv
db_name = argv[1]
print(db_name)

# db_name = 'Coffee'
# db_name = 'CBF'
db_name = 'Phoneme'


start_total_land = time.time()

#import os
#UCR_list = os.listdir("/home/aabanda/PycharmProjects/datasets/Univariate_ts/")

train_x, train_y = load_from_tsfile_to_dataframe("/home/aabanda/PycharmProjects/datasets/Univariate_ts/%s/%s_TRAIN.ts" % (db_name,db_name))
test_x, test_y = load_from_tsfile_to_dataframe("/home/aabanda/PycharmProjects/datasets/Univariate_ts/%s/%s_TEST.ts" % (db_name,db_name))

data = np.zeros((len(train_y)+len(test_y),len(train_x.iloc[1,0])))

for i in range(0,len(train_y)):
    data[i,:] = train_x.iloc[i,:][0]

k = 0
for i in range(len(train_y),len(train_y)+len(test_y)):
    data[i,:] = test_x.iloc[k,:][0]
    k = k +1

classes = np.concatenate((train_y,test_y))
classes = list(map(int, classes))


# general features
num_classes = len(np.unique(classes))
len_series = data.shape[1]
num_series = data.shape[0]





# RandF
start = time.time()
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(data[:len(train_y),:],train_y)
end_build = time.time()
preds = rfc.predict(data[len(train_y):,:])
rf_ac = sum(preds == test_y)/len(test_y)
end_test = time.time()

print("Correct:")
print(rf_ac)
print("\nTiming:")
print("\tTo build:   " + str(end_build - start) + " secs")
print("\tTo predict: " + str(end_test - end_build) + " secs")






#1NN with FastDTW with train/test

def myfastdtw(x,y):
    return fastdtw(x,y)[0]


start = time.time()
knn = KNeighborsClassifier(n_neighbors=1, metric=myfastdtw)

clus_data = np.transpose(data[:len(train_y)])

num_clusters = np.int(data[:len(train_y)].shape[0]*0.3)
cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
cluster.fit(np.transpose((clus_data)))
cluster_labels = cluster.labels_

representative = np.zeros((num_clusters))
for c in range(0,num_clusters):
    representative[c] = int(random.sample(list(np.where(cluster_labels==c)[0]),1)[0])


train_x_reduced = np.zeros((len(representative),len_series))
for r in range(0, len(representative)):
    train_x_reduced[r,:] = data[int(representative[r]),:]

representative = representative.astype(int)
train_y_reduced = train_y[representative]

knn.fit(train_x_reduced,train_y_reduced)
end_build = time.time()
kn_ac = knn.score(data[len(train_y):,:], test_y)
end_test = time.time()

print("Correct:")
print(kn_ac)
print("\nTiming:")
print("\tTo build:   " + str(end_build - start) + " secs")
print("\tTo predict: " + str(end_test - end_build) + " secs")









#ST with limited time = 10 mins


start = time.time()
pipeline = Pipeline([
    ('st', ContractedShapeletTransform(time_limit_in_mins=5,
                                       num_candidates_to_sample_per_case=10,
                                       verbose=True)),
    ('rf', RandomForestClassifier(n_estimators=10)),
])

try :
    pipeline.fit(train_x, train_y)
    end_build = time.time()
    preds = pipeline.predict(test_x)
    num_shapelets_found = len(pipeline.named_steps['st'].shapelets)
    st_ac = sum(preds == test_y) / len(test_y)
except :
    st_ac = 0
    num_shapelets_found = 0

end_test = time.time()

print("Correct:")
print(st_ac)
print("\nTiming:")
print("\tTo build:   " + str(end_build - start) + " secs")
print("\tTo predict: " + str(end_test - end_build) + " secs")









#Boss
from sktime.utils import all_estimators
all_estimators(estimator_type="classifier")



start = time.time()
number_of_windows = 3
windows = np.linspace(10, len_series, num=number_of_windows, endpoint=False)
accu = np.zeros((number_of_windows))
k=0

for w in windows:
    mod = BOSSIndividual(window_size=int(w),word_length=random.sample([16, 14, 12, 10, 8],1)[0],alphabet_size=3,norm=True)
    mod.fit(train_x,train_y)
    preds= mod.predict(test_x)
    test_y=np.asarray(test_y).astype(int)
    correct= sum(preds == test_y)
    print(w)
    print("\t" + str(correct / len(test_y)))
    accu[k] = correct / len(test_y)
    k = k+1

bs_ac = accu.max()
win = windows[np.argmax(accu)]/len_series

end_test = time.time()

print("bs_ac:")
print(kn_ac)
print("\nTiming:")
print("\tTodo: " + str(end_test - start) + " secs")










#TSF



start = time.time()
tsf = TimeSeriesForest(n_trees=10)
tsf.fit(train_x, train_y)
end_build = time.time()
preds = np.asarray(tsf.predict(test_x))
preds = preds.astype(int)
tsf_ac = sum(preds == test_y)/len(test_y)

end_test = time.time()

print("Correct:")
print(tsf_ac)
print("\nTiming:")
print("\tTo build:   " + str(end_build - start) + " secs")
print("\tTo predict: " + str(end_test - end_build) + " secs")



end_total_land = time.time()


print("\tTotal Landmarkers: " + str(end_total_land - start_total_land) + " secs")


features = np.array([num_classes,num_series,len_series,rf_ac,kn_ac,num_shapelets_found,st_ac,bs_ac,win,tsf_ac])
print(features)

np.savetxt("landmarkers_reduced/land_%s.csv" % db_name, features, delimiter=",")
