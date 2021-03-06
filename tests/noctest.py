import matplotlib.pyplot as plt
import pandas as pd
import csv


class NumberOfClustersTest:
    def __init__(self, algorithm):
        self.algorithm = algorithm  # variable of the algorithm we test

    def test(self):
        data = self.algorithm.data
        tmp = pd.DataFrame(data)
        data = tmp.sample(random_state=42, n=min(18000, len(data)))
        scores = []
        for i in range(2, 40):
            self.algorithm.set_center(i)
            self.algorithm.toString()
            d, l, s = self.algorithm.cluster()
            scores.append([self.algorithm.n, s])
            print("cluster number " + str(i) + " score " + str(s))

        # print the max
        max_val = float("-inf")
        max_index = 0
        for i in range(len(scores)):
            if scores[i][1] >= max_val:
                max_val = scores[i][1]
                max_index = scores[i][0]
        print(scores)
        print("max value:", max_val)
        print("max index:", max_index)

        # ploting
        xpts = []
        ypts = []
        for i in range(len(scores)):
            xpts.append(scores[i][0])
            ypts.append(scores[i][1])
        plt.scatter(xpts, ypts)
        plt.plot(xpts, ypts)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Optimal number of Cluster Data Set " +
                  str(self.algorithm.datanum))
        plt.show()

        # save results
        filename = "results_best_cluster_for_dataset_" + \
            str(self.algorithm.datanum) + ".csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["number fo clusters", "score"])
            for score in scores:
                writer.writerow(score)
