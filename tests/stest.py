from scipy.stats import kruskal, entropy, ttest_ind, f_oneway
import csv


class StatisticalTest:
    def __init__(self, small, big):
        self.small = small  # what we think is the worst
        self.big = big  # what we think is the best

    def evaluate(self):
        stat, pval = ttest_ind(self.small, self.big)
        if stat < 0:
            realPVal = 1 - pval / 2
        else:
            realPVal = pval / 2

        if realPVal > 0.05:
            print("big is better")
        else:
            print("small is better")
        print(realPVal)
        return realPVal
