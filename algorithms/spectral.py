from .algorithm import Algorithm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from .pca import PCA_algorithm
import numpy as np
from sklearn.metrics import silhouette_score
import random

# Spectral Clustering


class Spectral(Algorithm):
    def __init__(self, n, data, datanum, calc_score):
        super().__init__(n, data, datanum, calc_score)

    def cluster(self):
        print("start clustering using spectral")
        row, col = self.data.shape
        if row > 18000:
            k = 18000
            data = np.array(pd.DataFrame(self.data).sample(
                random_state=random.randint(2, 100), n=18000))
        else:
            data = self.data

        # Scaling the Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Normalizing the Data
        X_normalized = normalize(X_scaled)

        # Converting the numpy array into a pandas DataFrame
        X_normalized = pd.DataFrame(X_normalized)

        x, y = PCA_algorithm(data)
        d = np.array([[xi, yi] for xi, yi in zip(x, y)])
        X_principal = pd.DataFrame(d)
        X_principal.columns = ['P1', 'P2']

        X_principal.head(2)

        # Building the clustering model
        spectral_model_rbf = SpectralClustering(
            n_clusters=self.n, affinity='rbf', random_state=random.randint(2, 1000))

        # Training the model and Storing the predicted cluster labels
        labels_rbf = spectral_model_rbf.fit_predict(X_principal)

        if self.calc_score:
            score = silhouette_score(
                np.array(X_principal), labels_rbf, metric='euclidean')
        else:
            score = 0
        return X_principal, labels_rbf, score
