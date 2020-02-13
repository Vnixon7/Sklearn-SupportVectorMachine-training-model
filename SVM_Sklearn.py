import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics


cancer = datasets.load_breast_cancer()
###print(cancer.feature_names)
###print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
###print(x_train)
###print(y_train)

classes = ['malignant','benign']



'''best = 1.0
for i in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    model = svm.SVC(kernel='linear', C=2)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print('Iteration:',i,'acc:',accuracy)
    if accuracy > best:
        best = accuracy
        with open('SVM_model.pickle', 'wb') as f:
            pickle.dump(model, f)
            print(best)
            break'''


load_in = open('SVM_model.pickle', 'rb')
model = pickle.load(load_in)

predicted = y_predict = model.predict(x_test)
for i in range(len(predicted)):
    print(classes[predicted[i]],x_test[i],classes[y_test[i]])
