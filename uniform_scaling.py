from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt




train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

ind =2 #class 1
ind = 0 #Class 2
# ind = 5 #Class 2
ref = test_x.values[ind,:][0].values
plt.plot(ref)
test_y[ind]

l = len(ref)
start = 50
end = 100
k = 0.7


# rescaled_t = np.zeros((len(ref),2))
# rescaled_t[:,0] = np.array(range(len(ref)))
if start>0 and end>0:
    rescaled_t = np.concatenate((np.array(range(start)),
                                 np.array(range(0,end-start)) * k+start,
                                 range(np.int(start+(end-start)*k),(np.int(start+(end-start)*k) + l-end)+1)))
elif start>0 and end==0:
    rescaled_t = np.concatenate((np.array(range(start)), np.array(range(0, l - start)) * k + start))
else:
    rescaled_t =  np.array(range(l))*k



new_index = np.array(range(np.int(np.max(rescaled_t))+1))
t_trasnformed = np.zeros(len(new_index))

end_ind =0
for t in range(0,len(new_index)):
    if end==0:
        if t == 0:
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
        elif t <= np.int(start + (end - start) * k):
            before = rescaled_t[rescaled_t < t][-1]
            after = rescaled_t[rescaled_t >= t][0]

            t_trasnformed[t] = (ref[np.where(rescaled_t == before)[0][0]] * (k - (t - before)) + ref[
                np.where(rescaled_t == after)[0][0]] * (
                                        k - (after - t))) / k
        else:
            t_trasnformed[t] = ref[end + end_ind]
            end_ind = end_ind + 1
plt.plot(ref)
plt.plot(t_trasnformed)
