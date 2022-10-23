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

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=0)
classf = tree.DecisionTreeClassifier()
clsf = classf.fit(xtrain.data, ytrain.data)
viz = dtreeviz(classf, xtrain.data, ytrain.target, target_name='species', feature_names = species.feature_names, class_names=["setosa", "versiolor","virginica"])
st.graphviz_chart(viz)
# viz.view() 
