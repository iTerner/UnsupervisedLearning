from scipy.stats import ttest_ind, f_oneway
import csv


class StatisticalTest:
    def __init__(self, lists):
        self.vectors = lists  # list of list that represent the vector

    """
    The function preform Two-Sample One-Tailed T-Test
    """

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

    """
    The function preform ANOVA test
    """

    def anova_test(self):
        stat, p_value = f_oneway(*self.vectors)
        if p_value < 0.05:
            print("doesnt have same population")
        else:
            print("have same population")

        print("stat =", stat)
        print("p =", p_value)
