# -*- coding: utf-8 -*-
"""KMeans Clustering (Mall Cust).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MTOCbkhKno73hqBSxhbJpakj00Fp4pb8
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file = "iris.csv"

iris = pd.read_csv(file)

iris.head()

features = ['sepal_length', 'sepal_width']
iris_sepal = iris[features]

iris_sepal

plt.scatter(iris_sepal['sepal_length'], iris_sepal['sepal_width'], c = "brown");

features = ['petal_length', 'petal_width']
iris_petal = iris[features]

iris_petal

plt.scatter(iris_petal['petal_length'], iris_petal['petal_width'], c = "pink");

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
sepal = kmeans.fit(iris_sepal)
y1_kmeans = kmeans.predict(iris_sepal)
petal = kmeans.fit(iris_petal)
y2_kmeans = kmeans.predict(iris_petal)

plt.scatter(iris_sepal['sepal_length'], iris_sepal['sepal_width'], c=y1_kmeans, s=50, cmap='PRGn')

centers1 = kmeans.cluster_centers_
centers1

plt.scatter(iris_petal['petal_length'], iris_petal['petal_width'], c=y2_kmeans, s=50, cmap='inferno')

centers2 = kmeans.cluster_centers_
centers2