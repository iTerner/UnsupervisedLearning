import pandas
import numpy as np
import statistics
import random

"""
load the data for the MidTerm work
"""


class LoadData1():

    def __init__(self):
        pass

    def get_data3(self):
        df = pandas.read_csv("data3\\e-shop clothing 2008.csv", sep=";")
        data = np.array(df)
        d = {}
        l = []
        for row in data:
            if not row[7] in l:
                l.append((row[7]))
        for x in l:
            d[x] = self.get_values(x)
        for row in data:
            row[7] = d[row[7]]

        return pandas.DataFrame(data)

    def get_values(self, v):
        l1, l2 = v[0], v[1]
        if len(v) >= 3:
            l3 = v[2]
        else:
            l3 = l2
            l2 = 0
        value = 10 * int(l2) + int(l3)
        if l1 == 'B':
            value += 43
        elif l1 == 'C':
            value += 76
        elif l1 == 'P':
            value += 135
        return value

    def calculate_median(self, d):
        l = []
        for key in d.keys():
            l.append(d[key])
        return statistics.median(l)

    def get_data2(self):
        df = pandas.read_csv("dataset_diabetes\\diabetic_data.csv")
        alldata = np.array(df)
        l, k, k1, s = [], [], [], []  # K:2 - 5, K1: 10-11 s: 18-20
        counter, counterk, counterk1, counters = [], [], [], []
        for i in range(22, 50):
            l.append([])
        for i in range(2, 6):
            k.append([])
        for i in range(10, 12):
            k1.append([])
        for i in range(18, 21):
            s.append([])
        for row in alldata:
            for i in range(22, 50):
                if not row[i] in l[i - 22]:
                    l[i - 22].append(row[i])
            for i in range(2, 6):
                if not row[i] in k[i - 2] and row[i] != "?":
                    k[i - 2].append(row[i])
            for i in range(10, 12):
                if not row[i] in k1[i - 10] and row[i] != "?":
                    k1[i - 10].append(row[i])
            for i in range(18, 21):
                if not row[i] in s[i - 18] and not row[i].isdigit() and row[i] != "?":
                    s[i - 18].append(row[i])
        count = 0
        for i in range(len(l)):
            counter.append({})
            for item in l[i]:
                counter[i][item] = count
                count += 1
            count = 0
        count = 0
        for i in range(len(k)):
            counterk.append({})
            for item in k[i]:
                counterk[i][item] = count
                count += 1
            if i == 0 or i == (len(k) - 1):
                m = self.calculate_median(counterk[i])
                counterk[i]["?"] = m
            count = 0
        count = 0
        for i in range(len(k1)):
            counterk1.append({})
            for item in k1[i]:
                counterk1[i][item] = count
                count += 1
            m = self.calculate_median(counterk1[i])
            counterk1[i]["?"] = m
            count = 0
        count = 0
        for i in range(len(s)):
            counters.append({})
            for item in s[i]:
                counters[i][item] = count
                count += 1
            m = self.calculate_median(counters[i])
            counters[i]["?"] = m
            count = 0

        for row in alldata:
            for i in range(22, 50):
                row[i] = counter[i - 22][row[i]]
            for i in range(2, 6):
                row[i] = counterk[i - 2][row[i]]
            for i in range(10, 12):
                row[i] = counterk1[i - 10][row[i]]
            for i in range(18, 21):
                if not row[i].isdigit():
                    row[i] = counters[i - 18][row[i]]
        alldata = pandas.DataFrame(alldata)
        del(alldata[2])
        del(alldata[3])
        return alldata

    def get_data1(self):
        df = pandas.read_csv("online_shoppers_intention.csv")
        t = {}
        h = {}
        count1, count2 = 1, 1
        data = np.array(df)
        for row in data:
            if not row[10] in t:
                t[row[10]] = count1
                count1 += 1
            if not row[15] in h:
                h[row[15]] = count2
                count2 += 1
            if row[16]:
                row[16] = 1
            else:
                row[16] = 0
            if row[17]:
                row[17] = 1
            else:
                row[17] = 0
        for row in data:
            row[10] = t[row[10]]
            row[15] = h[row[15]]
        return pandas.DataFrame(data)
