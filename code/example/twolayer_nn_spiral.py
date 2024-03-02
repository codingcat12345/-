import sys
import os
import numpy as np
import matplotlib.pyplot as plt
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter4 import two_layer_ANN as ann_file
from example import softmax_regression_spiral as spiral

def plot_spiral_ANN(X,y,parameters,ANN):
    h=0.02
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),(np.arange(y_min,y_max,h)))
    XX=np.c_[xx.ravel(),yy.ravel()]
    Z=ANN.forward_prop(XX,parameters)
    Z=np.argmax(Z,axis=1)
    Z=Z.reshape(xx.shape)
    fig=plt.figure()
    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral,alpha=1)
    plt.scatter(X[:,0],X[:,1],c=y,s=20,cmap=plt.cm.spring)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())    

if __name__ == '__main__':
    X,y=spiral.gen_spiral_dataset(150,2,5)
    ANN=ann_file.class_two_layer_ANN(X,y)
    parameters=ANN.initialize_parameters(2,3,5)
    parameters,losses=ANN.gradient_descent_ANN(0.5,1000)
    plt.plot(losses)
    print(losses[-1])
    plot_spiral_ANN(X,y,parameters,ANN)
    acc=ANN.get_accuracy(X,y,parameters)
    print(acc)
    plt.show()