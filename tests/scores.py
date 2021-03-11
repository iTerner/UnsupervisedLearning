import random
import pandas as pd
import numpy as np
import csv
from algorithms.kmeans import Kmeans
from algorithms.gmm import GMM
from algorithms.fuzzycmeans import FuzzyCMeans
from algorithms.spectral import Spectral
from algorithms.hierarchical import Heirarchical
from sklearn.metrics import adjusted_mutual_info_score


class ScoreInformation:
    def __init__(self, data, datanumber):
        self.data = data  # the data we want to get his scores
        self.datanumber = datanumber  # the data number

    """
    n represent number of times.
    centers repersents the number of cluster.
    """

    def get_all_scores(self, k, centers):
        data = self.data
        scores = [[], [], [], [], []]
        row, col = self.data.shape
        if row > 18000:
            data = pd.DataFrame(data)
            random_data = np.array(data.sample(
                random_state=random.randint(2, 100), n=18000))
        else:
            random_data = data

        algorithms = [
            Kmeans(centers, self.data, self.datanumber, True),
            GMM(centers, self.data, self.datanumber, True),
            FuzzyCMeans(centers, self.data, self.datanumber, True),
            Spectral(centers, self.data, self.datanumber, True),
            Heirarchical(centers, self.data, self.datanumber, True)
        ]

        for i in range(k):
            for j in range(len(algorithms)):
                d, l, s = algorithms[j].cluster()
                scores[j].append(s)
            print("finished " + str(i + 1))

        filename = "scores_for_dataset" + str(self.datanumber) + "_outlier.csv"
        labels = ["kmeans", "gmm", "fuzzy", "spectral", "hierarchical"]
        with open(filename, 'a+', newline='') as f:
            writer = csv.writer(f)
            for i in range(5):
                writer.writerow([labels[i]])
                for j in scores[i]:
                    writer.writerow([j])

    def mutal_information(self, colnumber, centers):
        tmp = pd.DataFrame(self.data)
        print(tmp.shape)
        l = tmp.shape
        c = list(np.array(tmp[colnumber]).T)
        random_data = []
        country = []
        tmp_data = np.array(self.data)
        for i in range(min(l[0], 18000)):
            country.append(c[i])
            random_data.append(tmp_data[i])

        algorithms = [
            Kmeans(centers, self.data, self.datanumber, False),
            GMM(centers, self.data, self.datanumber, False),
            FuzzyCMeans(centers, self.data, self.datanumber, False),
            Spectral(centers, self.data, self.datanumber, False),
            Heirarchical(centers, self.data, self.datanumber, False)
        ]

        labels = []

        for i in range(len(algorithms)):
            d, l, s = algorithms[i].cluster()
            labels.append(l)

        with open("information.csv", "a+", newline='') as f:
            writer = csv.writer(f)
            t = "dataset " + str(self.datanumber) + \
                " col number " + str(colnumber)
            writer.writerow([t])
            writer.writerow(["mutal information KMeans: ", str(
                adjusted_mutual_info_score(country, labels[0]))])
            writer.writerow(["mutal information GMM: ", str(
                adjusted_mutual_info_score(country, labels[1]))])
            writer.writerow(["mutal information Fuzzy: ", str(
                adjusted_mutual_info_score(country, labels[2]))])
            writer.writerow(["mutal information Hierarchical: ",
                             str(adjusted_mutual_info_score(country, labels[4]))])
            writer.writerow(["mutal information Spectral: ", str(
                adjusted_mutual_info_score(country, labels[3]))])

        print("mutal information KMeans:",
              adjusted_mutual_info_score(country, labels[0]))
        print("mutal information GMM:",
              adjusted_mutual_info_score(country, labels[1]))
        print("mutal information Fuzzy:",
              adjusted_mutual_info_score(country, labels[2]))
        print("mutal information Hierarchical:",
              adjusted_mutual_info_score(country, labels[4]))
        print("mutal information Spectral:",
              adjusted_mutual_info_score(country, labels[3]))
