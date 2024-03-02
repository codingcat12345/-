import sys
import os
import numpy as np
import matplotlib.pyplot as plt
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter3 import linear_regression as linerg

def get_data(filename):
    x,y=[],[]
    with open(filename) as file:
        for eachrow in file:
            s=eachrow.split(',')
            x.append(float(s[0]))
            y.append(float(s[1]))
    return x,y

if __name__ ==  '__main__':
    filename= './chapter3/data.txt'
    xtrain,ytrain=get_data(filename)
    xtrain=np.asarray(xtrain)
    ytrain=np.asarray(ytrain)
    w=linerg.normal_equation_method(xtrain,ytrain)
    print(w)
    w,b=0.,0.
    history=linerg.gradient_descent_method(xtrain,ytrain,w,b,0.02,10000)
    print(history[-1])
    costs=[linerg.loss(xtrain,ytrain,w,b) for w,b in history]
    plt.plot(costs)
    plt.show()
        



'''
    reference
    https://ithelp.ithome.com.tw/articles/10202725
    https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes
'''