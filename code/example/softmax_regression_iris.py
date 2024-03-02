import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter3 import softmax_regression as sfmax

if __name__ ==  '__main__':
    iris=pandas.read_csv("./chapter3/iris_num.csv")
    shuffuled_row=np.random.permutation(iris.index)
    iris=iris.loc[shuffuled_row,:]
    X=iris[['sepal.length','sepal.width','petal.length','petal.width']].values
    X=np.c_[(np.ones((X.shape[0],1),dtype=X.dtype)),X]
    y=iris[['variety']].values
    X_train=X[0:80]
    y_train=y[0:80]
    y_train=sfmax.one_hot_represent(y_train)
    X_valid=X[80:-1]
    y_valid=sfmax.one_hot_represent(y[80:-1])
    w=np.zeros((X.shape[1],len(np.unique(y))))
    history,loss_train=sfmax.gradient_descent_softmax_one_hot(w,X_train,y_train,0.,0.05,1000)
    loss_valid=sfmax.valid_loss(history,X_valid,y_valid,True)
    w=history[-1]
    print(loss_train[-1])
    plt.plot(loss_train)
    plt.plot(loss_valid)
    plt.show()
    acc=sfmax.getAccuracy(w,X,y)
    print(acc)