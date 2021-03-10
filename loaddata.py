import pandas as pd
import numpy as np
import random
import csv


class LoadData():

    def __init__(self):
        pass

    def get_data(self):
        data = pd.read_csv("HTRU_2.csv")
        x, y = np.array(data).shape
        # adding a 'head' row to the data
        with open("HTRU_2.csv", "r", encoding="utf-8", newline='') as f:
            with open("HTRU.csv", "w", encoding="utf-8", newline='') as g:
                writer = csv.writer(g)
                l = []
                for i in range(y):
                    s = "factor " + str(i + 1)
                    l.append(s)
                writer.writerow(l)
                for row in f:
                    row = list(map(float, row.split(",")))
                    writer.writerow(row)

        return pd.read_csv("HTRU.csv")

    def get_data1(self):
        data = pd.read_csv("allUsers.lcl.csv", sep=",", na_values="?")
        # removing the first row, the row full with zeros
        data.drop(index=0, axis=0, inplace=True)
        # removing all the rows with more than 25% NaN values
        num_col = len(data.columns)
        p = 0.90
        num_row = int(num_col * p)
        data.dropna(axis=0, thresh=num_row, inplace=True)
        # fill the NaN values with row size median
        data.fillna(value=data.median(axis=0), axis=0, inplace=True)

        """
        l = []
        for i in np.array(data["Class"]):
            if not i in l:
                l.append(i)
        print(l)
        print(len(l))
        """
        return data


# print(LoadData().get_data())

"""
print(LoadData().get_data1())
print(data)
print(len(data))
row, col = data.shape
print("row: ", row)
print("col: ", col)
x, y = PCA_algorithm(data)
print(np.array(x))
print(np.array(y))
"""
