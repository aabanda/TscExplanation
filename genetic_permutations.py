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

train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")






tsf = TimeSeriesForest(n_trees=10)
tsf.fit(train_x, train_y)
tsf.predict(np.asarray(np.asarray(test_x.values[0,:][0]).reshape(1,-1)))

indx_test = 2
ts = np.asarray(np.asarray(test_x.values[indx_test,:][0]).reshape(1,-1))[0]
test_y[indx_test]
print(tsf.predict(ts.reshape(1,-1)))


per = np.array(list(range(0,len(ts))))
#per[[61,62,63,64,65,68,70]]= per[[70,68,65,64,63,62,61]]
# per[[43,64,76,91,94,96]]= per[[91,96,94,43,76,64]]


#per = np.random.permutation(len(ts))
# tsf.predict(ts[per].reshape(1,-1))

# plt.plot(ts)
# plt.plot(ts[per])



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
            chromosome =  list(range(0,len(ts)))
            # mask = np.random.rand(len(chromosome)) < 0.3
            # chromosome[mask] = 1
            population.append(chromosome)
        return population

    def fitness(self, population):

        scores = []
        for chromosome in population:
            chromosome = list(map(int, chromosome))
            class_penalization = 0
            if np.asarray(tsf.predict(ts.reshape(1, -1))).astype(int)[0] == np.asarray(tsf.predict(ts[chromosome].reshape(1, -1))).astype(int)[0]:
                class_penalization = 1

            score = kendallTau(list(range(0, len(ts))), chromosome) / ((len(ts) * (len(ts) - 1)) / 2)+ class_penalization

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
                child1 = np.zeros(len(ts))
                child1[:] = np.nan

                child1[range(start,end)] = chromosome1[range(start,end)]
                dif = np.setdiff1d(range(0,len(ts)), child1[range(start,end)])
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
                    indx = random.sample(range(len(ts)), num)
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


sel.fit()


sel.scores_best[-1]
sel.scores_avg[-1]
sel.plot_scores()
sel.support_
winner = np.asarray(list(map(int, sel.support_)))
kendallTau(range(0,len(ts)),winner)/((len(ts)*(len(ts)-1))/2)
plt.plot(ts)
plt.plot(ts[winner])
#
# list(range(0,len(ts)))-winner
# plt.hist(list(range(0,len(ts)))-winner)

print(tsf.predict(ts[winner].reshape(1, -1)))

list(range(0,len(ts)))-winner

# M=sel.support_
# M = M.astype(int)
#
# M1 = M.reshape(len(a),len(a))
#
# plt.plot(a.dot(M1))