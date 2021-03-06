from .algorithm import Algorithm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from .pca import PCA_algorithm
from sklearn.mixture import GaussianMixture
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score

from kneed import KneeLocator
import random


class GMM(Algorithm):
    def __init__(self, n, data, datanum, calc_score):
        super().__init__(n, data, datanum, calc_score)

    def cluster(self):
        print("start clustering using gmm")

        row, col = self.data.shape
        #print("data length:", self.data.shape)
        if row > 18000:
            k = 18000
            data = np.array(pd.DataFrame(self.data).sample(
                random_state=random.randint(2, 100), n=18000))
        else:
            data = self.data

        # PCA
        x, y = PCA_algorithm(self.data)
        d = np.array([[xi, yi] for xi, yi in zip(x, y)])

        # GMM
        gmm = GaussianMixture(
            n_components=self.n, random_state=random.randint(2, 50))
        gmm.fit(d)
        labels = np.array(gmm.predict(d))

        frame = pd.DataFrame(d)

        frame = pd.DataFrame(data=d, columns=['dim1', 'dim2'])

        frame['cluster'] = labels

        if self.calc_score:
            score = silhouette_score(d, labels, metric='euclidean')
        else:
            score = 0

        return frame, labels, score
