from algorithms.kmeans import Kmeans
from tests.noctest import NumberOfClustersTest
from loaddata import LoadData
from plot import ClusterPlots
from tests.scores import ScoreInformation
from algorithms.hierarchical import Heirarchical
from algorithms.gmm import GMM
from tests.stest import StatisticalTest
import csv
from function import get_list
import numpy as np
from loadDataMid import LoadData1

from scipy.stats import kruskal, f_oneway


"""
The function finds out what is the best number of clusters for the given dataset
"""


def best_number_of_clusters_test(data, datanum):
    a = Kmeans(2, data, datanum, True)
    test = NumberOfClustersTest(a)
    test.test()


"""
create plot of the data after clustering
"""


def plot_data(data, datanum, centers, anomaly_index, points):
    p = ClusterPlots(centers, data, datanum)
    p.plot(anomaly_index, points)


"""
The function calculate the score for each algorithm and each dataset
"""


def get_scores(data, datanum, n):
    test = ScoreInformation(data, datanum)
    test.get_all_scores(10, n)


"""
The function preform statistical tests
"""


def statistical_tests(datanum, smallindex, bigindex, add=""):
    big = get_list(bigindex, datanum, add)
    small = get_list(smallindex, datanum, add)
    vectors = [small, big]
    test = StatisticalTest(vectors)
    p = test.evaluate()
    print("p value:", p)


"""
The function compute the external quality of clustering
"""


def external_quality_of_clustering(data, datanum, colnum, centers):
    s = ScoreInformation(data, datanum)
    s.mutal_information(colnum, centers)


"""
The function compute the silhouette score for every algorithm 30 times after
we remove the anomalous points using the Cluster based outlier detection
"""


def anomaly_detection(data, datanum, n):
    a = Kmeans(n, data, datanum, False)
    anom_point_index = a.Cluster_based_outlier_detection()
    data = a.data
    # get_scores(data, datanum, n)


"""
The function compute the silhouette score for every algorithm 30 times after
we remove the anomalous points using the density based outlier detection
"""


def density_estimate(data, datanum, n):
    a = Kmeans(n, data, datanum, False)
    a.density_based_outlier_detection()
    data = a.data
    # print("data without anomaly", a.data)
    get_scores(data, datanum, n)


"""
plot of the average score of each algorithm W/O the anomaly detection methods
"""


def score_compare_plot(data, datanum, n):
    p = ClusterPlots(n, data, datanum)
    p.score_plot()


"""
The function check what is the optimal number of clustering for each data set
"""


def optimal_clustering_test(data, datanum, n):
    a = Kmeans(2, data, datanum, True)
    test = NumberOfClustersTest(a)
    test.optimal_number_of_clustering_test(n)


"""
The function preform ANOVA test for the optimal number of clusters
"""


def anova_for_noc(d):
    vectors = []
    for key in d.keys():
        vectors.append(d[key])

    test = StatisticalTest(vectors)
    test.anova_test()


def main():
    #data = LoadData().get_data1()
    datanum = 1

    # density_estimate(data, datanum, 4)
    # optimal_clustering_test(data, datanum, 30)
    # score_compare_plot(data, datanum, 4)
    # anomaly_detection(data, datanum, 4)
    # external_quality_of_clustering(data, datanum, "Class", 3)
    # statistical_tests(datanum, 3, 0)
    # get_scores(data, datanum, 4)
    # plot_data(data, datanum, 3)
    # best_number_of_clusters_test(data, datanum)


if __name__ == "__main__":
    main()
