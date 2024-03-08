import sys
import os
import numpy as np
import matplotlib.pyplot as plt
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter4 import neural_network_2 as nnfile
from example import softmax_regression_spiral as spiral
from chapter3 import softmax_regression as sfmx
from chapter4 import optimizer as opt

def cross_entropy_loss_fun(f,y):
    loss=sfmx.softmax_crossentropy(f,y)
    loss_grad=sfmx.grad_softmax_crossentropy(f,y)
    return loss,loss_grad

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

    nn=nnfile.neuralnetwork2()
    nn.add_layer(nnfile.Dense(2,100,('random',0.01)))
    nn.add_layer(nnfile.Sigmoid())
    nn.add_layer(nnfile.Dense(100,num_class,('no',0.01)))
    nn.add_layer(nnfile.Sigmoid())
    # nn.add_layer(nnfile.Dense(200,num_class,('no',0.01)))

    optimizer=opt.SGD(nn.parameters(),10)

    losses=nnfile.tran_nn(nn,X,y,optimizer,cross_entropy_loss_fun,5000,450,True,500)
    print(np.mean(nn.predict(X)==y))
    plt.plot(losses)
    plot_spiral_NN(X,y,nn)
    plt.show()
