# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:04:34 2019

@author: Sriharsha Komera
"""
##Iporting required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
path='F:\\Krish\\Logistic Regression\\Social_Network_Ads.csv'
dataset=pd.read_csv(path)

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

#splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Creating the Logistic model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#predicting
y_pred=classifier.predict(X_test)

#confusion matrix, accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)
cr=classification_report(y_test,y_pred)

#Visualizing the output
from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1,step=0.01))
plt.contour(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75, cmap=ListedColormap(('red','green'))) 
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
