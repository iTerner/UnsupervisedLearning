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
from function import get_list

"""
This class create the plots of the cluster data with the anomaly points
"""


class ClusterPlots:
    def __init__(self, n, data, datanum):
        self.n = n  # The number of centers we used in the algorithm
        self.data = data  # the data
        self.datanum = datanum  # the dataset number

    def plot(self, anomaly_index, points):
        km = Kmeans(self.n, self.data, self.datanum, False)
        gm = GMM(self.n, self.data, self.datanum, False)
        fcm = FuzzyCMeans(self.n, self.data, self.datanum, False)
        spec = Spectral(self.n, self.data, self.datanum, False)
        heir = Heirarchical(self.n, self.data, self.datanum, False)

        # run the algorithms
        kmeans_data, kmeans_labels, kmeans_score = km.cluster()
        gmm_data, gmm_labels, gmm_score = gm.cluster()
        fuzzy_data, fuzzy_labels, fuzzy_score = fcm.cluster()
        spec_data, spec_labels, spec_score = spec.cluster()
        hier_data, hier_labels, hier_score = heir.cluster()

        # fix the data
        kmeans_data = pd.DataFrame.from_dict(kmeans_data)

        t = {"x": [], "y": []}
        for n1, n2 in hier_data:
            t["x"].append(n1)
            t["y"].append(n2)

        hier_data = pd.DataFrame.from_dict(t)

        gmm_data = gmm_data.drop("cluster", axis=1)
        gmm_data = gmm_data.rename(columns={"dim1": "x", "dim2": "y"})
        fuzzy_data = fuzzy_data.drop("cluster", axis=1)
        fuzzy_data = fuzzy_data.rename(columns={0: "x", 1: "y"})
        spec_data = spec_data.rename(columns={"P1": "x", "P2": "y"})

        # remove the anomaly points from the data
        new_data = [kmeans_data, gmm_data, fuzzy_data, spec_data, hier_data]

        anomaly_points_data = []
        p = []
        for d in new_data:
            t = d
            t1 = d
            anomaly_points_data.append(
                t.drop(labels=anomaly_index, axis=0, inplace=False))
            p.append(t.drop(labels=points, axis=0, inplace=False))

        old_data = [kmeans_data.drop(anomaly_index), gmm_data.drop(anomaly_index), fuzzy_data.drop(
            anomaly_index), spec_data.drop(anomaly_index), hier_data.drop(anomaly_index)]
        #old_data = [kmeans_data, gmm_data, fuzzy_data, spec_data, hier_data]
        labels = [kmeans_labels, gmm_labels,
                  fuzzy_labels, spec_labels, hier_labels]

        new_labels = []
        for i in labels:
            new_labels.append(np.delete(i, anomaly_index))

        # get the scores
        score = []
        path = ["", "_outlier", "_density"]
        for i in range(3):
            score.append([])
            for j in range(5):
                l = get_list(j, self.datanum, path[i])
                score[i].append(sum(l) / len(l))

        fig, axs = plt.subplots(2, 3, figsize=(15, 15))
        fig.suptitle("Data Set " + str(self.datanum) +
                     " with opetimal number of clustering and anomalous points")

        # creating the figure
        title = ["Kmeans", "GMM", "Fuzzy C Means",
                 "Spectral Clustering", "Agglomerative Clustering"]
        count = 0
        for i in range(2):
            for j in range(3):
                if i == 1 and j == 2:
                    algo = ["K-Means", "GMM", "Fuzzy",
                            "Spectral", "Agglomerative"]
                    c = ['b', 'g', 'r']
                    l = ["avg score", "avg cluster anomaly",
                         "avg density anomaly"]
                    for k in range(len(score)):
                        axs[i, j].bar(algo, score[k], color=c[k], label=l[k])
                    axs[i, j].set_title("Silhouette Score", fontsize=10)
                    axs[i, j].set_ylabel("Score")
                    axs[i, j].legend(loc="lower right")
                else:
                    axs[i, j].scatter(p[count]["x"],
                                      p[count]["y"], c="black")
                    axs[i, j].scatter(anomaly_points_data[count]["x"],
                                      anomaly_points_data[count]["y"], c=new_labels[count])
                    axs[i, j].set_title(
                        str(title[count]) + " with " + str(self.n) + " clusters", fontsize=10)

                count += 1

        plt.show()

    def score_plot(self):
        # create a bar figure with the average score of each algorithm with different anomaly methods
        fig, ax = plt.subplots()
        width = 0.3

        labels = ["K-Means", "GMM", "Fuzzy", "Spectral", "Agglomerative"]
        x = np.arange(len(labels))

        # get the scores
        score = []
        path = ["", "_outlier", "_density"]
        for i in range(3):
            score.append([])
            for j in range(5):
                l = get_list(j, self.datanum, path[i])
                score[i].append(sum(l) / len(l))

        for s in score:
            print(s)

        rec1 = ax.bar(x - 2 * width / 3,
                      score[0], 2 * width / 3, label="avg score")
        rec2 = ax.bar(x, score[1], 2 * width / 3, label="avg clsuter anomaly")
        rec3 = ax.bar(x + 2 * width / 3, score[2],
                      2 * width / 3, label="avg density anomaly")

        ax.set_ylabel("Score")
        ax.set_title("Average Silhouette Score of the Algorithms")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        def autolabel(rects):
            # Attach a text label above each bar in *rects*, displaying its height.
            for rect in rects:
                height = rect.get_height()
                val = round(height, 3)
                ax.annotate('{}'.format(val),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rec1)
        autolabel(rec2)
        autolabel(rec3)

        fig.tight_layout()

        plt.show()
