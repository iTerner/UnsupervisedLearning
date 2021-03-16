from scipy.stats import ttest_ind
import csv


class StatisticalTest:
    def __init__(self, lists):
        self.vectors = lists  # list of list that represent the vector

    def evaluate(self):
        stat, pval = ttest_ind(self.vectors[0], self.vectors[1])
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

    def anova_test(self):
        pass
