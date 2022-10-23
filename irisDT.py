import streamlit as st
from sklearn import tree
from sklearn.tree import plot_tree
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz
import graphviz as graphviz

iris = sns.load_dataset('iris') 
X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
classf = tree.DecisionTreeClassifier()
clsf = classf.fit(xtrain, ytrain)
viz= dtreeviz(clsf, X_iris, y_iris, target_name="Classes", feature_names=["f0", "f1"], class_names=["c0", "c1"])
st.graphviz_chart(viz)

# st.write(clsf)

# fig = plt.figure(figsize=(10, 4))
# tree.plot_tree(classf.fit(xtrain, ytrain))
# st.pyplot(fig)

st.write(classf.score(xtest, ytest))


# ------

