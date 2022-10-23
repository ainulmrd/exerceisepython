import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = load_iris()
(X_iris, y_iris) = load_iris(return_X_y = True)
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