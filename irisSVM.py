import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

iris = sns.load_dataset('iris') 
X_iris = iris.drop('species', axis=1)  
y_iris = iris['species']

xtrain, xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state = 0)

clf = SVC(kernel='rbf', C=1).fit(xtrain, ytrain)
st.write('Iris dataset')
st.write('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(xtrain, ytrain)))
st.write('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(xtest, ytest)))

X_iris.shape

y_iris.shape

model = SVC()                       
model.fit(xtrain, ytrain)                  
y_model = model.predict(xtest)

from sklearn.metrics import accuracy_score
a = accuracy_score(ytest, y_model) 
st.write("Accuracy score:", a)

cr = classification_report(ytest, y_model)
st.write(cr)

from sklearn.metrics import confusion_matrix 
confusion_matrix(ytest, y_model)

svm = SVC(random_state=42, kernel='linear')

# Fit the data to the SVM classifier
svm = svm.fit(xtrain, ytrain)

# Evaluate by means of a confusion matrix
# matrix = plot_confusion_matrix(svm, xtest, ytest, cmap=plt.cm.Blues, normalize='true')
confusion_matrix = metrics.confusion_matrix(ytest, y_model)
cm = confusion_matrix
st.write("Confusion matrix:",cm)
fig = plt.figure(figsize=(10, 4))
sns.heatmap(cm, annot=True)
st.pyplot(fig)


