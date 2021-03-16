from .algorithm import Algorithm
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from .pca import PCA_algorithm
import numpy as np
import sklearn
from scipy.spatial.distance import cdist
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import pandas as pd

import random


class Kmeans(Algorithm):
    def __init__(self, n, data, datanum, calc_score):
        super().__init__(n, data, datanum, calc_score)

    def cluster(self):
        print("start clustering using kmeans")

        row, col = self.data.shape
        if row > 18000:
            k = 18000
            data = np.array(pd.DataFrame(self.data).sample(
                random_state=random.randint(2, 100), n=18000))
        else:
            data = self.data

        x, y = df = PCA_algorithm(self.data)
        d = np.array([[xi, yi] for xi, yi in zip(x, y)])

        X = np.array(list(zip(x, y))).reshape(len(x), 2)

        kmeans = KMeans(n_clusters=self.n,
                        random_state=random.randint(2, 1000)).fit(d)
        centroids = kmeans.cluster_centers_

        df = {'x': x, 'y': y}

        if self.calc_score:
            s = silhouette_score(d, kmeans.labels_, metric='euclidean')
        else:
            s = 0
        return df, kmeans.labels_.astype(int), s
