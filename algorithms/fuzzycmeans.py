
from __future__ import division, print_function
from .algorithm import Algorithm
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from .pca import PCA_algorithm
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import random

# Fuzzy C Means


class FuzzyCMeans(Algorithm):
    def __init__(self, n, data, datanum, calc_score):
        super().__init__(n, data, datanum, calc_score)

    def cluster(self):
        print("start clustering using fuzzy")

        row, col = self.data.shape
        if row > 18000:
            k = 18000
            data = np.array(pd.DataFrame(self.data).sample(
                random_state=random.randint(2, 100), n=18000))
        else:
            data = self.data

        dataframe = pd.DataFrame(self.data)
        alldata = np.array(dataframe.T)

        x, y = PCA_algorithm(np.array(dataframe))
        df = np.array([[xi, yi] for xi, yi in zip(x, y)])
        alldata = df.T
        df = pd.DataFrame(df)

        tmp = alldata

        ncenters = self.n
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, ncenters, 2, error=0.005, maxiter=1000, init=None, seed=random.randint(2, 1000))

        labels = np.argmax(u, axis=0)
        if self.calc_score:
            score1 = silhouette_score(alldata.T, labels, metric='euclidean')
        else:
            score1 = 0
        df['cluster'] = labels
        return df, labels, score1
