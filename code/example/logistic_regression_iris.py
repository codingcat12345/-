import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter3 import logistic_regression as logrg

if __name__ ==  '__main__':
    iris=pandas.read_csv("./chapter3/iris_2.csv")
    shuffuled_row=np.random.permutation(iris.index)
    iris=iris.loc[shuffuled_row,:]
    X=iris[['sepal.length','sepal.width','petal.length','petal.width']].values
    y=(iris.variety=='Versicolor').values.astype(int)
    # print(X)
    # print(y)
    y=np.expand_dims(y.transpose(),-1)
    print(y.shape)
    X_train=X[0:80]
    y_train=y[0:80]
    X_valid=X[80:-1]
    y_valid=y[80:-1]
    reg=0
    alpha=0.01
    iter=10000
    w_hisory=logrg.gradient_descent_logistic_reg(X_train,y_train,reg,alpha,iter)
    w=w_hisory[-1]
    print(w)
    loss_history_train=logrg.loss_logistic_history(w_hisory,X_train,y_train,reg)
    loss_history_valid=logrg.loss_logistic_history(w_hisory,X_valid,y_valid,reg)
    plt.plot(loss_history_train)
    plt.plot(loss_history_valid)
    plt.show()


"""
    reference:
        https://gist.github.com/netj/8836201
"""