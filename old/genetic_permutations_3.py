from sktime.utils.load_data import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import f1_score
from sktime.classifiers.interval_based import TimeSeriesForest
from itertools import permutations
import itertools as it

from sktime.classifiers.shapelet_based import ShapeletTransformClassifier


train_x, train_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/GunPoint/GunPoint_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../../datasets/Univariate_ts/GunPoint/GunPoint_TEST.ts")


# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/Coffee/Coffee_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/Coffee/Coffee_TEST.ts")


# train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
# test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")




np.unique(train_y)
tsf = ShapeletTransformClassifier(time_contract_in_mins=5)
tsf.fit(train_x, train_y)


indx_test =0
ts = pd.DataFrame(test_x.iloc[indx_test,:])
test_y[indx_test]
print(tsf.predict(ts))
#plt.plot(ts.values[0][0])





# a = np.random.rand(150)
# test_x.iloc[0,:][0] = pd.Series(a)
# tsf.predict(pd.DataFrame(test_x.iloc[0,:]))




# tsf = TimeSeriesForest(n_trees=10)
# tsf.fit(train_x, train_y)
# tsf.predict(np.asarray(np.asarray(test_x.values[0,:][0]).reshape(1,-1)))

# indx_test = 1
# ts = np.asarray(np.asarray(test_x.values[indx_test,:][0]).reshape(1,-1))[0]
# test_y[indx_test]
# ts1 = pd.DataFrame(ts)
# print(tsf.predict(ts.reshape(1,-1)))
# print(tsf.predict(ts1))


per = np.array(list(range(0,len(ts.values[0][0]))))
#per[[61,62,63,64,65,68,70]]= per[[70,68,65,64,63,62,61]]
# per[[43,64,76,91,94,96]]= per[[91,96,94,43,76,64]]


#per = np.random.permutation(len(ts))
# tsf.predict(ts[per].reshape(1,-1))

# plt.plot(ts)
# plt.plot(ts[per])

# plt.plot(ts.values[0][0])


# t = np.linspace(0, 2 * np.pi, 20)
# x = np.sin(t)
# y = np.cos(t)
#
# t = range(0, len(ts.values[0][0]))
# x = ts.values[0][0]
# plt.scatter(t,x, c= a)
# plt.show()


# plt.plot(ts.values[0][0][per])
#tsf.predict(pd.DataFrame(pd.DataFrame(ts.values[0][0][per])))


a = ts.values[0][0][per]
test_x.iloc[0,:][0] = pd.Series(a)
tsf.predict(pd.DataFrame(test_x.iloc[0,:]))



# preds = np.asarray(tsf.predict(test_x))
# preds = preds.astype(int)
# tsf_ac = sum(preds == test_y)/len(test_y)


#
#
# a = train_x.values[0,:][0]
# b = train_x.values[1,:][0]

# a = np.array([1,1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,1,1])
# b = np.array([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,1,1,1])
# len(ts)
# plt.plot(ts)


# A= list(range(0,len(ts)))
# B = chromosome



def kendallTau(A, B=None):
    # if any partial is B
    if B is None : B = list(range(len(A)))
    n = len(A)
    pairs = it.combinations(range(n), 2)
    distance = 0
    # print("IIIIMNNMNNN",list(pairs),len(A))
    for x, y in pairs:
        #if not A[x]!=A[x] and not A[y]!=A[y]:#OJO no se check B
        a = A[x] - A[y]
        try:
            b = B[x] - B[y]# if discordant (different signs)
        except:
            print("ERROR kendallTau, check b",A, B, x, y)
        # print(b,a,b,A, B, x, y,a * b < 0)
        if (a * b < 0):
            distance += 1
    return distance

#kendallTau(list(range(0,len(ts))), chromosome)/((len(ts)*(len(ts)-1))/2)




# ==============================================================================
# Class performing feature selection with genetic algorithm
# ==============================================================================
class GeneticSelector():
    def __init__(self, n_gen, size, n_best, n_rand,
                 n_children, mutation_rate):

        # Number of generations
        self.n_gen = n_gen
        # Number of chromosomes in population
        self.size = size
        # Number of best chromosomes to select
        self.n_best = n_best
        # Number of random chromosomes to select
        self.n_rand = n_rand
        # Number of children created during crossover
        self.n_children = n_children
        # Probablity of chromosome mutation
        self.mutation_rate = mutation_rate

        if int((self.n_best + self.n_rand) / 2) * self.n_children != self.size:
            raise ValueError("The population size is not stable.")

    def initilize(self):
        population = []
        for i in range(self.size):
            #chromosome =  np.random.permutation(len(ts))
            chromosome =  list(range(0,len(ts.values[0][0])))
            # mask = np.random.rand(len(chromosome)) < 0.3
            # chromosome[mask] = 1
            population.append(chromosome)
        return population

    def fitness(self, population):

        scores = []
        for chromosome in population:
            #print(chromosome)
            chromosome = list(map(int, chromosome))
            class_penalization = 0
            dist_penalization = 0

            a = ts.values[0][0][chromosome]
            test_x.iloc[0, :][0] = pd.Series(a)
            #tsf.predict(pd.DataFrame(test_x.iloc[0, :]))


            if np.asarray(tsf.predict(ts)).astype(int)[0] != np.asarray(tsf.predict(pd.DataFrame(test_x.iloc[0, :]))).astype(int)[0]:
                class_penalization = 15

            if kendallTau(list(range(0, len(ts.values[0][0]))), chromosome) / ((len(ts.values[0][0]) * (len(ts.values[0][0]) - 1)) / 2) > 0.2:
                dist_penalization = 15

            if kendallTau(list(range(0, len(ts.values[0][0]))), chromosome) / ((len(ts.values[0][0]) * (len(ts.values[0][0]) - 1)) / 2) ==0:
                score = 15 + dist_penalization + class_penalization
            else:
            #print("class pena")
            # print(class_penalization)
            # print(chromosome)
            # print(kendallTau(list(range(0, len(ts.values[0][0]))), chromosome))
            # print(((len(ts.values[0][0]) * (len(ts.values[0][0]) - 1)) / 2))
                score = 1/(kendallTau(list(range(0, len(ts.values[0][0]))), chromosome) / ((len(ts.values[0][0]) * (len(ts.values[0][0]) - 1)) / 2)) + dist_penalization+ class_penalization

            # if kendallTau(list(range(0,len(ts))), chromosome) ==0:
            #     score = kendallTau(list(range(0, len(ts))), chromosome) / ((len(ts) * (len(ts) - 1)) / 2) - \
            #             np.abs(np.asarray(tsf.predict(ts.reshape(1, -1))).astype(int)[0]
            #                    - np.asarray(tsf.predict(ts[chromosome].reshape(1, -1))).astype(int)[0]) - 1
            # else:
            #     score = 1/(kendallTau(list(range(0,len(ts))), chromosome)/((len(ts)*(len(ts)-1))/2))  -\
            #         np.abs(  np.asarray(tsf.predict(ts.reshape(1,-1))).astype(int)[0]
            #                  - np.asarray(tsf.predict(ts[chromosome].reshape(1,-1))).astype(int)[0]) -1
            scores.append(score)

        scores, population = np.array(scores), np.array(population)
        inds = np.argsort(scores)
        print(list(scores[inds])[0])
        return list(scores[inds]), list(population[inds, :])

    def select(self, population_sorted):
        population_next = []
        for i in range(self.n_best):
            population_next.append(population_sorted[i])
        for i in range(self.n_rand):
            population_next.append(random.choice(population_sorted))
        #random.shuffle(population_next)
        return population_next

    def crossover(self, population):
        population_next = []
        population_next.append(population[0]) # Manterner el mejor tal cual
        random.shuffle(population)
        for i in range(int(len(population) / 2)):
            for j in range(self.n_children):
                chromosome1, chromosome2 = population[i], population[len(population) - 1 - i]
                co_len= np.round(len(chromosome1)*0.4).astype(int)
                start = random.randint(0,len(chromosome1)- co_len)
                end = start + co_len
                child1 = np.zeros(len(ts.values[0][0]))
                child1[:] = np.nan

                child1[range(start,end)] = chromosome1[range(start,end)]
                dif = np.setdiff1d(range(0,len(ts.values[0][0])), child1[range(start,end)])
                #random.shuffle(dif)
                child1[np.isnan(child1)] = dif

                # child2 = np.zeros(len(ts))
                # child2[:] = np.nan
                # child2[range(start, end)] = chromosome2[range(start, end)]
                # dif = np.setdiff1d(range(0, len(ts)), child2[range(start, end)])
                # #random.shuffle(dif)
                # child2[np.isnan(child2)] = dif

                population_next.append(child1)
                # population_next.append(child2)
        return population_next

    def mutate(self, population):
        population_next = []
        for i in range(len(population)-1):
            chromosome = population[i]
            if i == 0:
                population_next.append(chromosome)
            else:
                if random.random() < self.mutation_rate:
                    proportion_permus_to_mutate = 0.1
                    num = np.round(len(chromosome)*proportion_permus_to_mutate).astype(int)
                    indx = random.sample(range(len(ts.values[0][0])), num)
                    indx2 = indx.copy()
                    random.shuffle(indx)
                    chromosome[indx2]= chromosome[indx]
                population_next.append(chromosome)
        return population_next

    def generate(self, population):
        # Selection, crossover and mutation
        scores_sorted, population_sorted = self.fitness(population)
        population = self.select(population_sorted)
        population = self.crossover(population)
        population = self.mutate(population)
        # History
        self.chromosomes_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))

        return population

    def fit(self):

        self.chromosomes_best = []
        self.scores_best, self.scores_avg = [], []

        population = self.initilize()
        for i in range(self.n_gen):
            print(i)
            population = self.generate(population)

            if self.scores_best[-1]< 0.02:
                return self

        return self

    @property
    def support_(self):
        return self.chromosomes_best[-1]

    def plot_scores(self):
        plt.plot(self.scores_best, label='Best')
        plt.plot(self.scores_avg, label='Average')
        plt.legend()
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        plt.show()



sel = GeneticSelector(n_gen=100, size=200, n_best=20, n_rand=60,
                              n_children=5, mutation_rate=0.6)



neig = []

# for i in range(0,50):


sel.fit()

sel.scores_best[-1]
sel.scores_avg[-1]
sel.plot_scores()
sel.support_
winner = np.asarray(list(map(int, sel.support_)))
kendallTau(range(0,len(ts.values[0][0])),winner)/((len(ts.values[0][0])*(len(ts.values[0][0])-1))/2)
plt.plot(ts.values[0][0])
plt.plot(np.asarray(ts.values[0][0][winner]))
# type(ts)
# type(winner)
# list(range(0,len(ts.values[0][0])))-winner
# plt.hist(list(range(0,len(ts)))-winner)

#neig = [neig, winner]






print(tsf.predict(ts))

np.asarray(ts.values[0][0][winner])

a = ts.values[0][0][winner]
plt.plot(ts.values[0][0])
plt.plot(np.asarray(a))
test_x.iloc[0,:][0] = pd.Series(a)
print(tsf.predict(pd.DataFrame(test_x.iloc[0,:])))

ts2 = ts.values[0][0].copy()
plt.plot(np.asarray(ts2))

ts2[17] = np.asarray(a)[17]
ts2[53] = np.asarray(a)[53]
ts2[58] = np.asarray(a)[58]
test_x.iloc[0,:][0] = pd.Series(ts2)
print(tsf.predict(pd.DataFrame(test_x.iloc[0,:])))


plt.plot(np.asarray(ts.values[0][0]))
plt.plot(ts2)

print(tsf.predict(ts.values[0][0][winner].reshape(1, -1)))

#
#
# e = np.vstack((np.array(list(range(len(ts)))), winner,np.array(list(range(len(ts))))-winner ))
# np.savetxt("fname", e, delimiter=' ', header='')
# list(range(0,len(ts)))-winner
#
#
# a1 = np.asarray([1,2,3,4,5,6,5,4,3,2,1])
# per1 = np.asarray([0,1,2,5,4,6,5,7,8,9,10])
#
#
# a1 = np.asarray([0,2,0,1,0])
# per1 = np.asarray([0,1,3,2,4])
# per1+1
#
# type(a1)
# type(per1)
# a1[per1]
# per1[a1]
#
# plt.plot(a1)
# plt.plot(a1[per1])
#
#
# list(range(0,len(a1)))-per1

# M=sel.support_
# M = M.astype(int)
#
# M1 = M.reshape(len(a),len(a))
#
# plt.plot(a.dot(M1))