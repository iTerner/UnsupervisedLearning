import csv


def get_list(n, datanumber, add):
    """
        param: n - a number between 0 - 4.
        0 returns the scores for KMeans
        1 returns the scores for GMM
        2 returns the scores for Fuzzy C Means
        3 returns the scores for spectral
        4 returns the scores for Heirarchical
    """
    l = []
    filename = "scores_for_dataset" + str(datanumber) + str(add) + ".csv"
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
