import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import LocalOutlierFactor


class Algorithm:
    def __init__(self, n, data, datanum, calc_score):
        self.n = n  # number of clusters
        self.data = data  # the data set
        self.datanum = datanum  # the number of the dataset
        self.calc_score = calc_score  # True if we need to calculate the algorithm score

    """
    The function sets a new center number value
    """

    def set_center(self, center):
        self.n = center

    def cluster(self):
        pass

    def toString(self):
        s = "number of centers: " + \
            str(self.n) + " data set number " + str(self.datanum)
        print(s)

    def Cluster_based_outlier_detection(self):
        anomalous_points = pd.DataFrame()
        dataset = pd.DataFrame(self.data).copy()
        tmp = pd.DataFrame(self.data).copy()

        while True:
            f, labels, s = self.cluster()
            print(labels)
            print(len(labels))

            silhouette_list = silhouette_samples(dataset, labels)
            # save all the point with negative score
            neg_silhouette_list = silhouette_list < 0
            print(neg_silhouette_list)
            # remove and save anomalous rows
            temp_anomalous_points = dataset.iloc[neg_silhouette_list]
            dataset = pd.concat(
                [dataset, temp_anomalous_points]).drop_duplicates(keep=False)

            if temp_anomalous_points.empty:
                break

            if anomalous_points.empty:
                anomalous_points = temp_anomalous_points
            else:
                anomalous_points = pd.concat(
                    [anomalous_points, temp_anomalous_points])

            self.data = dataset

        #self.data = tmp
        #indexs = anomalous_points.index
        # print(indexs)

        print("anomalous points \n", anomalous_points)

    def density_based_outlier_detection(self):
        tmp = pd.DataFrame()

        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        y_pred = clf.fit_predict(self.data)
        print("y_pred", y_pred)
        X_scores = clf.negative_outlier_factor_
        print("X_scores", X_scores)

        count = X_scores < sum(X_scores) / len(X_scores)

        temp_anomalous_points = self.data.iloc[count]
        dataset = pd.concat([self.data, temp_anomalous_points]
                            ).drop_duplicates(keep=False)

        self.data = dataset
        print(self.data)
        print(self.data.shape)

        points = 0
        for i in range(len(count)):
            if count[i]:
                points += 1
        print(points)
