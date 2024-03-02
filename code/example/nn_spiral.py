import sys
import os
import numpy as np
import matplotlib.pyplot as plt
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter4 import neural_network as nnfile
from chapter4 import train_nn
from example import softmax_regression_spiral as spiral

def plot_spiral_NN(X,y,nn):
    h=0.02
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),(np.arange(y_min,y_max,h)))
    XX=np.c_[xx.ravel(),yy.ravel()]
    Z=nn.predict(XX)
    Z=Z.reshape(xx.shape)
    fig=plt.figure()
    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral,alpha=1)
    plt.scatter(X[:,0],X[:,1],c=y,s=20,cmap=plt.cm.spring)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())  

if __name__ == '__main__':
    num_class=3
    X,y=spiral.gen_spiral_dataset(150,2,num_class)
    print(X.shape)

    nn=nnfile.neuralnetwork()
    nn.add_layer(nnfile.Dense(2,20,'tanh'))
    nn.add_layer(nnfile.Dense(20,20,'sigmoid'))
    nn.add_layer(nnfile.Dense(20,num_class,'sigmoid'))

    train_nn.train_batch(nn,X,y,train_nn.cross_entropy_loss_fun,1000,50,10,True,0,100)
    plot_spiral_NN(X,y,nn)
    print(np.mean(nn.predict(X)==y))

    plt.show()
