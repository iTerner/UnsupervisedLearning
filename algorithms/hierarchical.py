from .algorithm import Algorithm
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from .pca import PCA_algorithm
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score

from kneed import KneeLocator
import random


class Heirarchical(Algorithm):
    def __init__(self, n, data, datanum, calc_score):
        super().__init__(n, data, datanum, calc_score)

    def cluster(self):
        print("start clustering using Heirarchical")

        row, col = self.data.shape
        if row > 18000:
            data = np.array(pd.DataFrame(self.data).sample(
                random_state=random.randint(2, 100), n=18000))
        else:
            data = self.data

        x, y = PCA_algorithm(data)
        d = np.array([[xi, yi] for xi, yi in zip(x, y)])

        cluster = AgglomerativeClustering(
            n_clusters=self.n, affinity='euclidean', linkage='ward')
        cluster.fit_predict(d)

        labels = cluster.labels_

        if self.calc_score:
            score = silhouette_score(d, labels, metric='euclidean')
        else:
            score = 0

        return d, cluster.labels_, score
