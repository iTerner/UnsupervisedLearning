import random
import pandas as pd
import numpy as np
import csv
from algorithms.kmeans import Kmeans
from algorithms.gmm import GMM
from algorithms.fuzzycmeans import FuzzyCMeans
from algorithms.spectral import Spectral
from algorithms.hierarchical import Heirarchical
import matplotlib.pyplot as plt


class ClusterPlots:
    def __init__(self, n, data, datanum):
        self.n = n  # The number of centers we used in the algorithm
        self.data = data  # the data
        self.datanum = datanum  # the dataset number

    def plot(self):
        km = Kmeans(self.n, self.data, self.datanum, True)
        gm = GMM(self.n, self.data, self.datanum, True)
        fcm = FuzzyCMeans(self.n, self.data, self.datanum, True)
        spec = Spectral(self.n, self.data, self.datanum, True)
        heir = Heirarchical(self.n, self.data, self.datanum, True)

        # run the algorithms
        kmeans_data, kmeans_labels, kmeans_score = km.cluster()
        gmm_data, gmm_labels, gmm_score = gm.cluster()
        fuzzy_data, fuzzy_labels, fuzzy_score = fcm.cluster()
        spec_data, spec_labels, spec_score = spec.cluster()
        hier_data, hier_labels, hier_score = heir.cluster()

        avg_score = [kmeans_score, gmm_score,
                     fuzzy_score, spec_score, hier_score]

        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle("Data Set " + str(self.datanum) +
                     " with Optimal Clusters Number After Removing The Anomalous Points")
        # kmeans
        axs[0, 0].scatter(kmeans_data['x'], kmeans_data['y'], c=kmeans_labels)
        axs[0, 0].set_title("Kmeans with " + str(self.n) +
                            " clusters", fontsize=10)
        # gmm
        for k in range(0, 4):
            tmp = gmm_data[gmm_data["cluster"] == k]
            axs[0, 1].scatter(tmp["dim1"],
                              tmp["dim2"], cmap="rainbow")
        axs[0, 1].set_title("GMM with " + str(self.n) +
                            " clusters", fontsize=10)
        # fuzzy
        for k in range(4):
            f = fuzzy_data[fuzzy_data['cluster'] == k]
            axs[1, 0].scatter(f[0], f[1], cmap="rainbow")
        axs[1, 0].set_title("Fuzzy C Means with " +
                            str(self.n) + " clusters", fontsize=10)
        # Agglomerative
        axs[1, 1].scatter(hier_data[:, 0], hier_data[:, 1], c=hier_labels)
        axs[1, 1].set_title(
            "Agglomerative Clustering with " + str(self.n) +
            " Clusters", fontsize=10)
        # Spectral
        axs[2, 0].scatter(spec_data["P1"], spec_data["P2"], c=spec_labels)
        axs[2, 0].set_title("Spectral Clustering with " + str(self.n) +
                            " Clusters", fontsize=10)
        # scores for clusters
        # the average silhouette score of each algorithm
        algo = ["K-Means", "GMM", "Fuzzy", "Spectral", "Agglomerative"]
        axs[2, 1].bar(algo, avg_score, color=[
            'red', 'blue', 'purple', 'green', 'yellow'])
        axs[2, 1].set_title("Silhouette Score", fontsize=10)
        axs[2, 1].set_ylabel("Score")
        plt.show()
