from transformations.old.warp_function import warp, unwrapcircle
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import numpy as np
from sktime.distances.elastic import dtw_distance
from scipy.stats import betaprime
from sktime.classification.compose import TimeSeriesForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.utils.slope_and_trend import _slope
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pandas as pd
from sktime.classification.dictionary_based import BOSSEnsemble


train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")



def scale(ref,start, end, k):
    shifted_t = ref.copy()
    shifted_t[range(start,end)] = ref[range(start,end)]*k
    return shifted_t


def explanation1(ref, y_ref, warp_level, train_x, train_y, classifier):

    ref= X[test_index[ind], :]
    y_ref= y[test_index[ind]]
    warp_level=0.8
    train_x= X[train_index, :]
    train_y= y[train_index]
    classifier = "boss"
    classifier = "dtw"

    a = 8
    p = 0.3
    b = a * (1 - p) / p

    num_neig = 500

    start_end = np.zeros((num_neig, 2))
    for i in range(0, num_neig):
        start_end[i, :] = unwrapcircle(betaprime.rvs(a, b, size=1))

    start_end[start_end <= 0] = 0
    start_end = start_end * len(ref)
    start_end = start_end.astype(int)

    neig = []
    inter = np.zeros((num_neig * 6, 3))
    count = 0
    for i in range(0, num_neig):
        start = start_end[i, 0]
        end = start_end[i, 1]
        if end == len(ref):
            end = 0
        if start == 0:
            start = 1
        for k in [0.3, 0.5, 0.9, 1.1, 1.2, 1.3]:
            # neig.append(warp(ref, start, end, k))
            inter[count, :] = np.array([start, end, k])
            neig.append(scale(ref, start, end, k))
            count = count + 1

    inter[inter[:, 1] == 0, 1] = len(ref)

    if classifier=="dtw":

        distance_matrix = np.zeros((len(neig), train_x.shape[0]))

        for i in range(0, len(neig)):
            for j in range(0, train_x.shape[0]):
                #distance_matrix[i, j] = dtw_distance(neig[i], np.asarray(train_x.values[j, :][0]))
                distance_matrix[i, j] = dtw_distance(neig[i], np.asarray(train_x[j, :]))

        neig_y = train_y[np.argmin(distance_matrix, axis=1)]

        np.savetxt("sinteticas_inter_dtw.txt",inter)
        np.savetxt("sinteticas_y_dtw.txt",neig_y)

    elif classifier=="boss":

        X_data = pd.DataFrame(train_x)
        X = pd.DataFrame()
        X["dim_0"] = [pd.Series(X_data.iloc[x, :]) for x in range(len(X_data))]
        clf = BOSSEnsemble(max_ensemble_size=100)
        clf.fit(X,train_y)

        neig_y = []
        for i in range(len(neig)):
            neig_y.append(clf.predict(neig[i].reshape(1,1,-1))[0])

        neig_y = np.ravel(neig_y)
        # np.savetxt("sinteticas_inter_boss2_c105.txt", inter)
        # np.savetxt("sinteticas_y_boss2_c105.txt", neig_y)

    print(np.unique(neig_y, return_counts=True))



    neig_cop = neig_y.copy()
    inter_cop = inter.copy()

    scale_level = 0.3
    neig_y = neig_cop[inter_cop[:,2]==scale_level]
    inter = inter_cop[inter_cop[:,2]==scale_level,:]

    print(np.unique(neig_y, return_counts=True))

    #
    # inter = np.loadtxt("sinteticas_inter_boss2_c105.txt")
    # neig_y = np.loadtxt("sinteticas_y_boss2_c105.txt", )
    #
    # neig_y = neig_y[inter[:, 2] == 0.3]
    # inter = inter[inter[:, 2] == 0.3, :]
    # #np.unique(neig_y, return_counts=True)

    # Same class
    ind_sort = inter[neig_y.astype(int) == y_ref.astype(int), 2].argsort()
    inter2 = inter.copy()
    inter2 = inter2[neig_y.astype(int) == y_ref.astype(int), :][ind_sort]
    variables = inter2.copy()
    # greater than 1
    long_ind1 = (variables[inter2[:, 2] > 1, 1] - variables[inter2[:, 2] > 1, 0]).argsort()
    # smaller
    long_ind2 = (variables[inter2[:, 2] < 1, 1] - variables[inter2[:, 2] < 1, 0]).argsort()

    same_int = inter2


    #Other class
    ind_sort = inter[neig_y.astype(int)!= y_ref.astype(int), 2].argsort()
    inter2 = inter.copy()
    inter2 = inter2[neig_y.astype(int) != y_ref.astype(int), :][ind_sort]
    variables = inter2.copy()
    #greater than 1
    long_ind1 =( variables[inter2[:,2]>1,1]-variables[inter2[:,2]>1,0]).argsort()
    #smaller
    long_ind2 =( variables[inter2[:,2]<1,1]-variables[inter2[:,2]<1,0]).argsort()
    other_int = inter2


    same_int = same_int[:,[0,1]]
    other_int = other_int[:,[0,1]]

    same_int = np.delete(same_int,np.where(same_int[:,0]==same_int[:,1])[0],0 )
    other_int = np.delete(other_int,np.where(other_int[:,0]==other_int[:,1])[0],0 )

    from scipy.spatial.distance import directed_hausdorff
    dist_int = np.zeros((same_int.shape[0], other_int.shape[0]))
    for i in range(0, same_int.shape[0]):
        for j in range(0, other_int.shape[0]):
            a = np.asarray(range(same_int[i, :][0].astype(int), same_int[i, :][1].astype(int)))
            a = a.reshape(-1, 1)
            b = np.asarray(range(other_int[j, :][0].astype(int), other_int[j, :][1].astype(int)))
            b = b.reshape(-1, 1)
            dist_int[i, j] = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])



    ran = range(1, 30, 2)
    num = []
    for threshold in ran:
        thresh_per_same = []
        for i in range(0, dist_int.shape[0]):
            thresh_per_same.append(np.where(dist_int[i, :] < threshold)[0])
        k = np.concatenate(thresh_per_same, axis=0)
        num.append(len(np.unique(k, return_counts=True)[0]))

    # plt.plot(ran,num)
    thresh_per_same = []
    for i in range(0, dist_int.shape[0]):
        thresh_per_same.append(np.where(dist_int[i, :] < 5)[0])
    k = np.concatenate(thresh_per_same, axis=0)

    other_index = np.setdiff1d(np.arange(other_int.shape[0]), np.unique(k))

    # Other class
    ind_sort = inter[neig_y.astype(int) != y_ref.astype(int), 2].argsort()
    inter2 = inter.copy()
    inter2 = inter2[neig_y.astype(int) != y_ref.astype(int), :][ind_sort]
    inter2 = inter2[other_index, :]
    variables = inter2.copy()
    # greater than 1
    long_ind1 = (variables[inter2[:, 2] > 1, 1] - variables[inter2[:, 2] > 1, 0]).argsort()
    # smaller
    long_ind2 = (variables[inter2[:, 2] < 1, 1] - variables[inter2[:, 2] < 1, 0]).argsort()



    intervals = np.zeros((variables.shape[0],len(ref)))
    for i in range(0,intervals.shape[0]):
        intervals[i,range(variables[i,0].astype(int),variables[i,1].astype(int))] = 1

    return np.sum(intervals, axis=0)

plot_colormap(ref, np.sum(intervals, axis=0))

def explanation2(ref, y_ref, warp_level, train_x, train_y, classifier):

    a = 8
    p = 0.3
    b = a * (1 - p) / p

    num_neig = 500

    start_end = np.zeros((num_neig, 2))
    for i in range(0, num_neig):
        start_end[i, :] = unwrapcircle(betaprime.rvs(a, b, size=1))

    start_end[start_end <= 0] = 0
    start_end = start_end * len(ref)
    start_end = start_end.astype(int)

    neig = []
    inter = np.zeros((num_neig * 6, 3))
    count = 0
    for i in range(0, num_neig):
        start = start_end[i, 0]
        end = start_end[i, 1]
        if end == len(ref):
            end = 0
        if start == 0:
            start = 1
        for k in [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]:
            inter[count, :] = np.array([start, end, k])
            neig.append(warp(ref, start, end, k))
            count = count + 1

    inter[inter[:, 1] == 0, 1] = len(ref)

    distance_matrix = np.zeros((len(neig), train_x.shape[0]))

    for i in range(0, len(neig)):
        for j in range(0, train_x.shape[0]):
            distance_matrix[i, j] = dtw_distance(neig[i], np.asarray(train_x.values[j, :][0]))

    neig_y = train_y[np.argmin(distance_matrix, axis=1)]

    inter = np.loadtxt("sinteticas_inter_boss2_c105.txt")
    neig_y = np.loadtxt("sinteticas_y_boss2_c105.txt", )
    neig_y = neig_y[inter[:, 2] ==0.3]
    inter = inter[inter[:, 2] == 0.3, :]

    steps = [
        (
            "extract",
            RandomIntervalFeatureExtractor(
                n_intervals="sqrt", features=[np.mean, np.std, _slope]
            ),
        ),
        ("clf", DecisionTreeClassifier()),
    ]


    time_series_tree = Pipeline(steps)

    tsf = TimeSeriesForestClassifier(
        estimator=time_series_tree,
        n_estimators=100,
        criterion="entropy",
        bootstrap=True,
        oob_score=True,
        random_state=1,
        n_jobs=-1,
    )


    intervals = np.zeros((inter.shape[0], len(ref)))
    for i in range(0, intervals.shape[0]):
        intervals[i, range(inter[i, 0].astype(int), inter[i, 1].astype(int))] = inter[i, 2]

    inter_less = inter[inter[:, 2] == warp_level, :]
    y = neig_y[inter[:, 2] == warp_level]

    inter_less = inter[inter[:, 2] == 0.3, :]
    y = neig_y[inter[:, 2] == 0.3]


    intervals = np.zeros((inter_less.shape[0], len(ref)))
    for i in range(0, intervals.shape[0]):
        intervals[i, range(inter_less[i, 0].astype(int), inter_less[i, 1].astype(int))] = inter_less[i, 2]

    np.unique(y, return_counts=True)

    # tsf = TimeSeriesForestClassifier(n_estimators=1)
    X_data_intervals = pd.DataFrame(intervals)
    X = pd.DataFrame()
    X["dim_0"] = [pd.Series(X_data_intervals.iloc[x, :]) for x in range(len(X_data_intervals))]
    accu = []
    kf = StratifiedKFold(n_splits=5)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        tsf.fit(X_train, y_train)
        y_pred = tsf.predict(X_test)
        accu.append(f1_score(y_test.astype(int), y_pred.astype(int), average="weighted"))

    print(np.mean(accu))

    fi = tsf.feature_importances_
    f1 = fi[0]
    f3 = fi[2]

    return  f1['mean']/f3['mean']

plot_colormap(ref,  f1['mean']/f3['mean'])
