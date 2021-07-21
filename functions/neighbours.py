import numpy as np
from functions.random_intervals import unwrapcircle
from functions.transformations import warp, scale, noise, slice
from scipy.stats import betaprime
from sktime.distances.elastic import dtw_distance
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from scipy.spatial.distance import directed_hausdorff

def create_neighbours(ts, transformation, num_neig):

    a = 8
    p = 0.3
    b = a * (1 - p) / p


    start_end = np.zeros((num_neig, 2))
    for i in range(0, num_neig):
        start_end[i, :] = unwrapcircle(betaprime.rvs(a, b, size=1))

    start_end[start_end <= 0] = 0
    start_end = start_end * len(ts)
    start_end = start_end.astype(int)

    if transformation == "warp":
        neig = []
        inter = np.zeros((num_neig * 6, 3))

        count=0
        for i in range(0, num_neig):
            start = start_end[i, 0]
            end = start_end[i, 1]
            if end == len(ts):
                end = 0
            if start == 0:
                start = 1

            for k in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
                inter[count, :] = np.array([start, end, k])
                neig.append(warp(ts, start, end, k))
                count= count+1

    elif transformation == "scale":

        neig = []
        inter = np.zeros((num_neig * 6, 3))
        count = 0
        for i in range(0, num_neig):
            start = start_end[i, 0]
            end = start_end[i, 1]
            if end == len(ts):
                end = 0
            if start == 0:
                start = 1
            for k in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
                inter[count, :] = np.array([start, end, k])
                neig.append(warp(ts, start, end, k))
                count = count + 1

    elif transformation == "noise":

        neig = []
        inter = np.zeros((num_neig * 5, 3))
        count=0
        for i in range(0, num_neig):
            start = start_end[i, 0]
            end = start_end[i, 1]
            for  k in [1, 3, 5, 7, 9]:
                inter[count, :] = np.array([start, end, k])
                neig.append(noise(ts, start, end, k))
                count = count + 1


    elif transformation == "slice":

        start_end = np.zeros((num_neig, 2))
        for i in range(0, num_neig):
            start_end[i, :] = unwrapcircle(betaprime.rvs(a, b, size=1))

        start_end[start_end <= 0] = 0
        start_end = start_end * len(ts)

        for i in range(num_neig):
            while np.abs(start_end[i, 1]- start_end[i, 0])<len(ts)*0.5 or np.abs(start_end[i, 1]- start_end[i, 0])==len(ts):
                start_end[i, :] = unwrapcircle(betaprime.rvs(a, b, size=1))
                start_end[i, :][start_end[i, :] <= 0] = 0
                start_end[i, :] = start_end[i, :] * len(ts)
                start_end[i, :] = start_end[i, :].astype(int)

        start_end = start_end.astype(int)


        neig = []
        inter = np.zeros((num_neig, 2))
        for i in range(0, num_neig):
            shi_pre = start_end[i, 0]
            shi_suf = start_end[i, 1]
            inter[i, :] = np.array([shi_pre, shi_suf])
            neig.append(slice(ts, shi_pre, shi_suf))


    else:

        print("Invalid transformation. Valid transformations are: warp, scale, noise or slice")

    return inter, neig


def label_neighbours(neig, classifier, train_x, train_y):

    if classifier == "dtw":

        distance_matrix = np.zeros((len(neig), train_x.shape[0]))

        for i in range(0, len(neig)):
            for j in range(0, train_x.shape[0]):
                distance_matrix[i, j] = dtw_distance(neig[i], np.asarray(train_x.values[j, :][0]))

        neig_y = train_y[np.argmin(distance_matrix, axis=1)]


    elif classifier == "boss":
        clf = BOSSEnsemble(max_ensemble_size=100)
        clf.fit(train_x, train_y)

        neig_y = []
        for i in range(len(neig)):
            neig_y.append(clf.predict(neig[i].reshape(1, 1, -1))[0])

    elif classifier == "st":

        clf = ShapeletTransformClassifier(time_contract_in_mins=60)
        clf.fit(train_x, train_y)
        neig_y = []
        for i in range(len(neig)):
            neig_y.append(clf.predict(neig[i].reshape(1, 1, -1))[0])

    return np.asarray(neig_y).astype(int)






def compute_robustness(ts,  ts_y, transformation, level, inter, neig_y):


    inter[inter[:, 1] == 0, 1] = len(ts)


    if transformation == "warp" or transformation == "scale" or transformation=="noise":

        neig_y = neig_y[inter[:, 2] == level]
        inter = inter[inter[:, 2] == level, :]
        ts_y = ts_y.astype(int)

        if len(np.intersect1d(np.unique(neig_y), [ts_y])) == 0 or \
                np.unique(neig_y, return_counts=True)[1][
                    np.where(np.unique(neig_y) == ts_y)[0][0]] >= (0.99 * len(neig_y)):

            print("High-level explanation:")
            print("Robustness = 1")
            return [1, 0]

        else:
            same = np.unique(neig_y, return_counts=True)[1][np.where(np.unique(neig_y) == ts_y)[0][0]]
            robus = same / len(neig_y)
            print("High-level explanation: Robustness = %.2f" % robus)

            # Same class
            ind_sort = inter[neig_y.astype(int) == ts_y, 2].argsort()
            inter2 = inter.copy()
            inter2 = inter2[neig_y.astype(int) == ts_y, :][ind_sort]
            same_int = inter2

            # Other class
            ind_sort = inter[neig_y.astype(int) != ts_y.astype(int), 2].argsort()
            inter2 = inter.copy()
            inter2 = inter2[neig_y.astype(int) != ts_y.astype(int), :][ind_sort]

            other_int = inter2
            same_int = same_int[:, [0, 1]]
            other_int = other_int[:, [0, 1]]

            same_int = np.delete(same_int, np.where(same_int[:, 0] == same_int[:, 1])[0], 0)
            other_int = np.delete(other_int, np.where(other_int[:, 0] == other_int[:, 1])[0], 0)

            dist_int = np.zeros((same_int.shape[0], other_int.shape[0]))
            for i in range(0, same_int.shape[0]):
                for j in range(0, other_int.shape[0]):
                    a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
                    a = a.reshape(-1, 1)
                    b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
                    b = b.reshape(-1, 1)
                    dist_int[i, j] = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])

            ran = range(1, 60, 2)

            num = []
            for threshold in ran:
                thresh_per_same = []
                for i in range(0, dist_int.shape[0]):
                    thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
                k = np.concatenate(thresh_per_same, axis=0)
                num.append(len(np.unique(k)))

            # plt.figure(figsize=(7, 5))
            # #plt.plot(ran, np.repeat(28,len(num)) - num,label="$\gamma$",linewidth=3)
            # plt.plot(ran, num,label="$\gamma$",linewidth=3)
            # plt.xlabel("s", size=25)
            # plt.ylabel("$\gamma$", size=25, rotation='horizontal',labelpad=30)
            # plt.xticks(np.arange(0, 20, step=2),size=25)
            # plt.yticks(size=25)
            # # plt.legend(fontsize=25)
            # plt.tight_layout()

            pendiente = []
            for i in range(len(num) - 1):
                pendiente.append(num[i + 1] - num[i])

            p_pendiente = []
            for i in range(len(pendiente) - 1):
                p_pendiente.append(pendiente[i + 1] - pendiente[i])

            p = (np.asarray(p_pendiente) > 0).astype(int)
            cambio = []
            for pp in range(len(p) - 1):
                cambio.append(p[pp + 1] - p[pp])

            if len(np.where(np.asarray(cambio) == 1)[0]) > 0:
                ind_p = np.where(np.asarray(cambio) == 1)[0][0] + 2
                threshold = np.asarray(ran)[ind_p]

                thresh_per_same = []
                for i in range(0, dist_int.shape[0]):
                    thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
                k = np.concatenate(thresh_per_same, axis=0)

                other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))


            if len(np.where(np.asarray(cambio) == 1)[0]) == 0 or len(other_index) <= 5:
                middle = (np.where(np.asarray(num) > other_int.shape[0] / 2))[0][0].astype(int)  # percentil 50
                threshold = np.asarray(ran)[middle]

                thresh_per_same = []
                for i in range(0, dist_int.shape[0]):
                    thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
                k = np.concatenate(thresh_per_same, axis=0)

                other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))


            if len(other_index) == 0:

                print("The isolation of the intervals of interest is not feasible")

            ind_sort = inter[neig_y.astype(int) != ts_y, 2].argsort()
            inter2 = inter.copy()
            inter2 = inter2[neig_y.astype(int) != ts_y, :][ind_sort]
            inter2 = inter2[other_index, :]
            variables = inter2.copy()

            intervals = np.zeros((variables.shape[0], len(ts)))
            for i in range(0, intervals.shape[0]):
                intervals[i, range(variables[i, 0].astype(int), variables[i, 1].astype(int))] = 1

            print("Low-level explanation: w=")
            print(np.sum(intervals, axis=0))

            return [robus,  np.sum(intervals, axis=0)]

    elif transformation=="slice":

        ts_y = ts_y.astype(int)
        same = np.unique(neig_y, return_counts=True)[1][np.where(np.unique(neig_y) == ts_y)[0][0]]
        robus = same / len(neig_y)
        print("High-level explanation: Robustness = %.2f" % robus)

        inter[inter[:, 1] == 0, 1] = len(ts)
        ts_y = ts_y.astype(int)

        if len(np.intersect1d(np.unique(neig_y), [ts_y])) == 0 or np.unique(neig_y, return_counts=True)[1][
            np.where(np.unique(neig_y) == ts_y)[0][0]] >= (0.99 * len(neig_y)):


            print("High-level explanation:")
            print("Robustness = 1")
            return [1, 0]


        else:

            # Same class
            inter2 = inter.copy()
            inter2 = inter2[neig_y.astype(int) == ts_y, :]
            same_int = inter2

            # Other class
            inter2 = inter.copy()
            inter2 = inter2[neig_y.astype(int) != ts_y, :]
            other_int = inter2

            same_int = same_int[:, [0, 1]]
            other_int = other_int[:, [0, 1]]

            same_int = np.delete(same_int, np.where(same_int[:, 0] == same_int[:, 1])[0], 0)
            other_int = np.delete(other_int, np.where(other_int[:, 0] == other_int[:, 1])[0], 0)

            dist_int = np.zeros((same_int.shape[0], other_int.shape[0]))
            for i in range(0, same_int.shape[0]):
                for j in range(0, other_int.shape[0]):
                    a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
                    a = a.reshape(-1, 1)
                    b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
                    b = b.reshape(-1, 1)
                    dist_int[i, j] = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])

            ran = range(1, 60, 2)

            num = []
            for threshold in ran:
                thresh_per_same = []
                for i in range(0, dist_int.shape[0]):
                    thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
                k = np.concatenate(thresh_per_same, axis=0)
                num.append(len(np.unique(k)))

            pendiente = []
            for i in range(len(num) - 1):
                pendiente.append(num[i + 1] - num[i])

            p_pendiente = []
            for i in range(len(pendiente) - 1):
                p_pendiente.append(pendiente[i + 1] - pendiente[i])

            p = (np.asarray(p_pendiente) > 0).astype(int)
            cambio = []
            for pp in range(len(p) - 1):
                cambio.append(p[pp + 1] - p[pp])

            if len(np.where(np.asarray(cambio) == 1)[0]) > 0:
                ind_p = np.where(np.asarray(cambio) == 1)[0][0] + 2
                threshold = np.asarray(ran)[ind_p]

                thresh_per_same = []
                for i in range(0, dist_int.shape[0]):
                    thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
                k = np.concatenate(thresh_per_same, axis=0)

                other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))

            if len(np.where(np.asarray(cambio) == 1)[0]) == 0 or len(other_index) <= 5:
                middle = (np.where(np.asarray(num) > other_int.shape[0] / 2))[0][0].astype(int)  # percentil 50
                threshold = np.asarray(ran)[middle]

                thresh_per_same = []
                for i in range(0, dist_int.shape[0]):
                    thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
                k = np.concatenate(thresh_per_same, axis=0)

                other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))

            if len(other_index) == 0:

                print("The isolation of the intervals of interest is not feasible")

            # Other class

            inter2 = inter.copy()
            inter2 = inter2[neig_y.astype(int) != ts_y, :]
            inter2 = inter2[other_index, :]
            variables = inter2.copy()

            intervals = np.zeros((variables.shape[0], len(ts)))
            for i in range(0, intervals.shape[0]):
                intervals[i, range(variables[i, 0].astype(int), variables[i, 1].astype(int))] = 1


            print("Low-level explanation: w=")
            print(np.sum(intervals, axis=0))

            return [robus,  np.sum(intervals, axis=0)]

