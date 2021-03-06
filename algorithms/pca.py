from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

from sklearn.preprocessing import StandardScaler

"""
The function gets a data set
The function returns the data after PCA in 2 dimensions
"""


def PCA_algorithm(data):
    scalar = StandardScaler()

    # fitting - normalization
    scalar.fit(data)
    scaled_data = scalar.transform(data)
    scaled_data = normalize(scaled_data)
    # Importing PCA
    pca = PCA(n_components=2)

    x_pca = pca.fit_transform(scaled_data)

    x, y = [], []
    for row in x_pca:
        x.append(row[0])
        y.append(row[1])
    return x, y
