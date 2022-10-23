import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = pd.read_csv('iris.csv')
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

from sklearn.naive_bayes import GaussianNB 
model = GaussianNB()                     
model.fit(xtrain, ytrain)               
ymodel = model.predict(xtest) 

from sklearn.metrics import classification_report
st.write(classification_report(ytest, ymodel)) 
