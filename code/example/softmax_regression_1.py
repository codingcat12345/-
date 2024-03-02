import sys
import os
import numpy as np
import matplotlib.pyplot as plt
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter3 import softmax_regression as sfmax

if __name__ ==  '__main__':
    x=np.array([[2,3],[4,5]])
    y=np.array([[2],[1]])
    w=np.array([[0.1,0.2,0.3],[0.4,0.2,0.8]])
    reg=0.2
    # y=one_hot_represent(y,3)
    print(sfmax.gradient_softmax(w,x,y,reg))
    print(sfmax.softmax(x@w))
    print(sfmax.softmax_crossentropy(x@w,y)+reg*np.sum(w*w))
    history,loss=sfmax.gradient_descent_softmax(w,x,y)
    w=history[-1]
    print(sfmax.softmax(x@w))
    print(sfmax.softmax_crossentropy(x@w,y))
    plt.plot(loss)
    plt.show()

    