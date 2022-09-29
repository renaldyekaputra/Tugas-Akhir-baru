
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
import pickle
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report

DKI1 = pd.read_excel("./Data/Data ISPU - Normalization.xlsx", sheet_name="DKI1")
DKI2 = pd.read_excel("./Data/Data ISPU - Normalization.xlsx", sheet_name="DKI2")
DKI3 = pd.read_excel("./Data/Data ISPU - Normalization.xlsx", sheet_name="DKI3")
DKI4 = pd.read_excel("./Data/Data ISPU - Normalization.xlsx", sheet_name="DKI4")
DKI5 = pd.read_excel("./Data/Data ISPU - Normalization.xlsx", sheet_name="DKI5")

dki = pd.concat([DKI1,DKI2,DKI3,DKI4,DKI5])

dki = dki.reset_index(drop=True)

dki = dki.applymap(lambda s: s.lower() if type(s) == str else s)

dki

y = dki["Kategori"]
x = dki.drop(columns=["Kategori"])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

#Create a svm Classifier
clf = svm.SVC(kernel='rbf') 
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

iris=datasets.load_iris()
#print(iris)
X=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(X,y)
lin_reg=LinearRegression()
log_reg=LogisticRegression()
svc_model=SVC()
lin_reg=lin_reg.fit(x_train,y_train)
log_reg=log_reg.fit(x_train,y_train)
svc_model=svc_model.fit(x_train,y_train)


pickle.dump(lin_reg,open('lin_model.pkl','wb'))
pickle.dump(log_reg,open('log_model.pkl','wb'))
pickle.dump(svc_model,open('svc_model.pkl','wb'))