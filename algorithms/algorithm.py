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
