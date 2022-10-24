import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

file = "iris.csv"

iris = pd.read_csv(file)

st.header("KMeans Clustering for Iris Dataset")
iris.head()

features = ['sepal_length', 'sepal_width']
iris_sepal = iris[features]

# SEPAL
st.write("Feature: Sepal")
iris_sepal
fig = plt.figure(figsize=(8, 4))
plt.scatter(iris_sepal['sepal_length'], iris_sepal['sepal_width'], c = "brown");
st.pyplot(fig)

features = ['petal_length', 'petal_width']
iris_petal = iris[features]

kmeans = KMeans(n_clusters=4)
sepal = kmeans.fit(iris_sepal)
y1_kmeans = kmeans.predict(iris_sepal)

fig = plt.figure(figsize=(8, 4))
plt.scatter(iris_sepal['sepal_length'], iris_sepal['sepal_width'], c=y1_kmeans, s=50, cmap='PRGn')
st.pyplot(fig)

centers1 = kmeans.cluster_centers_
st.write("Cluster Center (Sepal)")
centers1

# PETAL
st.write("Feature: Petal")
iris_petal
fig = plt.figure(figsize=(8, 4))
plt.scatter(iris_petal['petal_length'], iris_petal['petal_width'], c = "salmon");
st.pyplot(fig)

petal = kmeans.fit(iris_petal)
y2_kmeans = kmeans.predict(iris_petal)

fig = plt.figure(figsize=(8, 4))
plt.scatter(iris_petal['petal_length'], iris_petal['petal_width'], c=y2_kmeans, s=50, cmap='inferno')
st.pyplot(fig)

centers2 = kmeans.cluster_centers_
st.write("Cluster Center (Petal)")
centers2
