from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt




train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")



np.unique(train_y)
ind =2 #class 1
ind = 0 #Class 2
ind = 5 #Class 3

ref = test_x.values[ind,:][0].values
# plt.plot(ref)
test_y[ind]

l = len(ref)
start = 15
k = 20

start = 60
k = 30

shifted_t = np.zeros(l+k)
shifted_t[range(start)] = ref[range(start)]
shifted_t[range(start+k,len(shifted_t))]= ref[range(start,l)]


before = ref[start-1]
after = ref[start]
std = np.abs(before - after) / 2

# fill = np.zeros(k)
# for i in range(len(fill)):
#     mean = ((k-i)/k)* before + (1-(k-i)/k) * after
#     std = np.abs(before- after)/2
#     fill[i] = np.random.normal(mean,std,1)
# shifted_t[range(start, start + k)] = fill

for t in range(start,start+k):
    mean =(((start+k-t))* before + (k-(start+k-t)) * after)/k
    std = np.abs(before - after)/5
    shifted_t[t]= np.random.normal(mean,std,1)

shifted_t[40]
t=40


plt.plot(ref)
plt.plot(shifted_t)

