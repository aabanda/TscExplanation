from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
from sktime.distances.elastic import dtw_distance
from sklearn.metrics import confusion_matrix
from scipy.stats import betaprime
from scipy.spatial.distance import directed_hausdorff
import sys
# #Separado por wapr level
# inter = np.loadtxt("inter_warp_3000.txt")
# neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")

ind = int(sys.argv[1])

ind=0

# train_x, train_y = load_from_tsfile_to_dataframe("CBF_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("CBF_TEST.ts")

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/ArrowHead/ArrowHead_TEST.ts")


ref = test_x.values[ind,:][0].values
print(test_y[ind])
plt.plot(ref)


def warp(ts,start,end,scale):
    ref = ts
    k = scale
    l = len(ref)
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



def intersection(intervals):
    start, end = intervals.pop()
    while intervals:
        start_temp, end_temp = intervals.pop()
        start = max(start, start_temp)
        end = min(end, end_temp)
    return [start, end]


def unwrapcircle(z):
    u = np.round(random.uniform(-z, 1), decimals=2)
    return intersection([[0,1],[u, u+z]])





a = 8
p= 0.3
b = a*(1-p)/p


num_neig = 500


start_end = np.zeros((num_neig,2))
for i in range(0,num_neig):
    start_end[i,:]=unwrapcircle(betaprime.rvs(a, b, size=1))


start_end[start_end<=0]=0
start_end = start_end*len(ref)
start_end = start_end.astype(int)



neig = []
inter = np.zeros((num_neig*6,3))
count = 0
for i in range(0,num_neig):
     start = start_end[i,0]
     end =  start_end[i,1]
     if end == len(ref):
         end = 0
     if start == 0:
         start = 1
     for k in [0.7,0.8,0.9,1.1,1.2,1.3]:
         inter[count, :] = np.array([start, end, k])
         neig.append(warp(ref, start, end, k))
         count = count+1


# #
# #
# neig = []
# inter = np.zeros((num_neig,3))
# count = 0
# for i in range(0,num_neig):
#      start = start_end[i,0]
#      end =  start_end[i,1]
#      if end == len(ref):
#          end = 0
#      if start == 0:
#          start = 1
#      for k in [1.2]:
#          inter[count, :] = np.array([start, end, k])
#          neig.append(warp(ref, start, end, k))
#          count = count+1
#

inter[inter[:,1]==0,1]=len(ref)

plt.plot(ref)
plt.xlabel("t")

ind_pr=0
ind_pr=2
ind_pr=6
ind_pr=4
ind_pr=18



plt.figure(figsize=(6,4))
plt.plot(ref,linewidth=3)
plt.xlim(0,len(ref))
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()


plt.figure(figsize=(6,4))
plt.plot(np.arange(inter[ind_pr,0], inter[ind_pr,1]), np.repeat(0,len(np.arange(inter[ind_pr,0], inter[ind_pr,1]))),linewidth=3,color="black")
plt.plot(np.arange(1, 100), np.repeat(0,len(np.arange(1,100))),linewidth=3,color="black")
plt.xlim(0,len(ref))
plt.ylim(-1.5,1.5)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(warp(ref,100,200,1.2),linewidth=3, label="Warped TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(warp(ref,30,100,0.7),linewidth=3, label="Warped TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()



distance_matrix = np.zeros((len(neig),train_x.shape[0]))

for i in range(0, len(neig)):
    for j in range(0, train_x.shape[0]):
        distance_matrix[i,j] = dtw_distance(neig[i], np.asarray(train_x.values[j,:][0]))



neig_y = train_y[np.argmin(distance_matrix,axis=1)]
print("importance")
print(np.unique(neig_y,return_counts=True))

np.savetxt("CBF_neig_%d.txt" % ind, neig_y)
np.savetxt("CBF_inter_%d.txt" % ind, inter)


inter = np.loadtxt('transformations/weights/CBF_inter_54.txt')
neig_y = np.loadtxt('transformations/weights/CBF_neig_54.txt')



inter = np.loadtxt('GP_inter_warp_3000.txt')
neig_y = np.loadtxt('GP_neig_warp_3000.txt')







#Load gunpoint neig and compute saliency map

ind=1
train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")

ref = test_x.values[ind,:][0].values
print(test_y[ind])
plt.plot(ref)


# inter = np.loadtxt('transformations/weights/GunPoint_inter_%s.txt' % ind)
# neig_y = np.loadtxt('transformations/weights/GunPoint_neig_%s.txt' % ind)


inter = np.loadtxt('transformations/weights/GP_scale_inter_%s.txt' % ind)
neig_y = np.loadtxt('transformations/weights/GP_scale_neig_%s.txt' % ind)


np.unique(neig_y,return_counts=True)

#
#
# inter_seg = inter.copy()
# neig_y_seg = neig_y.copy()
#
# inter = inter_seg
# neig_y = neig_y_seg

warp_level=1.2
neig_y = neig_y[inter[:,2]==warp_level]
inter = inter[inter[:,2]==warp_level,:]


inter[inter[:,1]==0,1] = len(ref)

np.unique(neig_y,return_counts=True)


#Same class
ind_sort = inter[neig_y.astype(int)== test_y[ind].astype(int), 2].argsort()
inter2 = inter.copy()
inter2 = inter2[neig_y.astype(int) == test_y[ind].astype(int), :][ind_sort]
variables = inter2.copy()
#greater than 1
long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
largest_indices1 =long_ind1
#smaller
long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
largest_indices2 =long_ind2

same_int = inter2




#Other class
ind_sort = inter[neig_y.astype(int)!= test_y[ind].astype(int), 2].argsort()
inter2 = inter.copy()
inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :][ind_sort]
variables = inter2.copy()
#greater than 1
long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
largest_indices1 = long_ind1[::-1]
largest_indices1 = long_ind1
#smaller
long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
largest_indices2 = long_ind2[::-1]
largest_indices2 = long_ind2
#variables = np.delete(variables,2,axis=1)

# variables = variables[neig_y==test_y[ind],:]
# variables.shape


other_int = inter2

same_int = same_int[:,[0,1]]
other_int = other_int[:,[0,1]]



same_int = np.delete(same_int,np.where(same_int[:,0]==same_int[:,1])[0],0 )

other_int = np.delete(other_int,np.where(other_int[:,0]==other_int[:,1])[0],0 )



np.unique(neig_y,return_counts=True)

dist_int = np.zeros((same_int.shape[0],other_int.shape[0]))
for i in range(0,same_int.shape[0]):
    for j in range(0,other_int.shape[0]):
        a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
        a = a.reshape(-1, 1)
        b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
        b = b.reshape(-1, 1)
        dist_int[i,j]=max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0])




ran = range(1,80,2)

num = []
for threshold in ran:
    thresh_per_same = []
    for i in range(0,dist_int.shape[0]):
        # min_per_same.append(np.argmin(dist_int[i,:]))
        # perc_per_same.append(np.where(dist_int[i,:]<np.percentile(dist_int,q=2))[0])
        thresh_per_same.append(np.where(dist_int[i,:]<threshold)[0])
    k = np.concatenate(thresh_per_same, axis=0)
    num.append(len(np.unique(k, return_counts=True)[0]))


plt.plot(ran,num)


if np.all(np.asarray(num) == num[0]):

    other_index = np.arange(other_int.shape[0])

else:


    #Elijo un threshold:
    pendiente = []
    for i in range(len(num) - 1):
        pendiente.append(num[i + 1] - num[i])

    ind_thre = np.argmin(pendiente)
    threshold = np.asarray(ran)[ind_thre]
    # plt.plot(pendiente)
    #
    p_pendiente = []
    for i in range(len(pendiente) - 1):
        p_pendiente.append(pendiente[i + 1] - pendiente[i])

    # np.column_stack((pendiente[:-1],p_pendiente))
    # np.column_stack((ran,num))

    p = ( np.asarray(p_pendiente)>0).astype(int)
    cambio = []
    for pp in range(len(p)-1):
        cambio.append(p[pp+1]-p[pp])

    if len(np.where(np.asarray(cambio)==1)[0])>0:
        ind_p= np.where(np.asarray(cambio)==1)[0][0]+2
        threshold = np.asarray(ran)[ind_p]

        thresh_per_same = []
        for i in range(0, dist_int.shape[0]):
            thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
        k = np.concatenate(thresh_per_same, axis=0)

        len(np.unique(k))
        other_int.shape
        same_int.shape

        other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
        print("len other index")
        len(other_index)

    if len(np.where(np.asarray(cambio)==1)[0])==0 or len(other_index)<=5:
        #middle = (np.where(np.asarray(num) == other_int.shape[0])[0][0] / 2).astype(int)
        middle = (np.where(np.asarray(num) >other_int.shape[0]/ 2))[0][0] .astype(int) #percentil 50
        threshold = np.asarray(ran)[middle]

        thresh_per_same = []
        for i in range(0, dist_int.shape[0]):
            thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
        k = np.concatenate(thresh_per_same, axis=0)

        len(np.unique(k))
        other_int.shape
        same_int.shape

        other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))


print("len other index")
print(len(other_index))
#
# other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))
# print("len other index")
# len(other_index)





#Other class
ind_sort = inter[neig_y.astype(int)!= test_y[ind].astype(int), 2].argsort()
inter2 = inter.copy()
inter2 = inter2[neig_y.astype(int) != test_y[ind].astype(int), :][ind_sort]
inter2 = inter2[other_index,:]
variables = inter2.copy()
#greater than 1
long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
largest_indices1 = long_ind1[::-1]
largest_indices1 = long_ind1
#smaller
long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
largest_indices2 = long_ind2[::-1]
largest_indices2 = long_ind2








variables2 = np.concatenate((variables[inter2[:,2]>1,:][largest_indices1,:],variables[inter2[:,2]<1,:][largest_indices2,:] ))


intervals = np.zeros((variables.shape[0],len(ref)))
for i in range(0,intervals.shape[0]):
    intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1
np.sum(intervals, axis=0)


print("Interval lengths")
print(np.mean(np.sum(intervals, axis=1)))
print(np.std(np.sum(intervals, axis=1)))

# weigths1 = np.sum(intervals, axis=0)
# np.savetxt('CBF_weights08_%d.txt' % ind,np.sum(intervals, axis=0) )



def plot_colormap(ref, weights):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x = range(0,len(ref)-1)
    y = ref[1:]
    # dydx = np.abs(clf.coef_)[0]
    #dydx = np.sum(intervals, axis=0)/len(other_index)
    dydx = weights
    dydx = dydx[1:]


    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)


    fig, axs = plt.subplots()

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)


    fig.colorbar(line, ax=axs)

    axs.set_xlim(0-5, len(x)+5)
    axs.set_ylim(-2.5, 2.5)
    axs.set_ylim(-2,1.7)
    plt.show()





plt.figure(figsize=(6,4))
plt.plot(ref,linewidth=3)
plt.plot(neig[ind_pr],linewidth=3)
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()

we_example = np.zeros(len(ref))
we_example[range(100,180)]=1
we_example[range(180,185)]=2
we_example[range(185,200)]=3
we_example[range(200,235)]=4
we_example[range(235,245)]=3
we_example[range(245,249)]=1



from matplotlib.collections import LineCollection

x = range(0,len(ref)-1)
y = ref[1:]
# dydx = np.abs(clf.coef_)[0]
#dydx = np.sum(intervals, axis=0)/len(other_index)
dydx = we_example
dydx = dydx[1:]
dydx= (dydx - np.min(dydx))/(np.max(dydx)-np.min(dydx))


points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)


fig, axs = plt.subplots(figsize=(8,4))

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='jet', norm=norm, lw=2)

# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(5)
line = axs.add_collection(lc)


cbar=fig.colorbar(line, ax=axs)
cbar.ax.tick_params(labelsize=20)

axs.set_xlim(0-5, len(x)+5)
axs.set_ylim(-2,1.7)
axs.tick_params(axis="both", labelsize=20)
plt.show()




plot_colormap(ref,np.sum(intervals, axis=0))



plot_colormap(ref,we_example)
