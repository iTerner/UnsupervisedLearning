from algorithms.kmeans import Kmeans
from tests.noctest import NumberOfClustersTest
from loaddata import LoadData
from plot import ClusterPlots
from tests.scores import ScoreInformation
from algorithms.hierarchical import Heirarchical
from algorithms.gmm import GMM
from tests.stest import StatisticalTest
import csv


def get_list(n, datanumber):
    """
        param: n - a number between 0 - 4.
        0 returns the scores for KMeans
        1 returns the scores for GMM
        2 returns the scores for Fuzzy C Means
        3 returns the scores for spectral
        4 returns the scores for Heirarchical
    """
    l = []
    filename = "scores_for_dataset" + str(datanumber) + ".csv"
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # count = 1
        for row in reader:
            l.append(row)
    k = []
    for i in range(1 + n + n * 30, 30 * (n + 1) + 1 + n):
        k.append(l[i])
    l = []
    for item in k:
        l.append(float(item[0]))
    return l


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


def plot_data(data, datanum, centers):
    p = ClusterPlots(centers, data, datanum)
    p.plot()


"""
The function calculate the score for each algorithm and each dataset
"""


def get_scores(data, datanum):
    test = ScoreInformation(data, datanum)
    test.get_all_scores(30, 3)


"""
The function preform statistical tests
"""


def statistical_tests(datanum):
    big = get_list(0, datanum)
    small = get_list(1, datanum)
    test = StatisticalTest(small, big)
    p = test.evaluate()
    print("p value:", p)


"""
The function compute the external quality of clustering
"""


def external_quality_of_clustering(data, datanum, colnum, centers):
    s = ScoreInformation(data, datanum)
    s.mutal_information(colnum, centers)


def anomaly_detection(data, datanum, n):
    a = Kmeans(n, data, datanum, False)
    a.Cluster_based_outlier_detection()
    data = a.data
    plot_data(data, datanum, n)


def density_estimate(data, datanum, n):
    a = Kmeans(n, data, datanum, False)
    a.density_based_outlier_detection()


def main():
    data = LoadData().get_data()
    datanum = 2

    #density_estimate(data, datanum, 3)
    #anomaly_detection(data, datanum, 3)

    #external_quality_of_clustering(data, datanum, "Class", 2)
    # statistical_tests(datanum)
    # get_scores(data, datanum)
    #plot_data(data, datanum, 3)
    #best_number_of_clusters_test(data, datanum)


if __name__ == "__main__":
    main()
