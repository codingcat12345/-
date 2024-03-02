import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter3 import softmax_regression as sfmax

np.random.seed(0)
PI=math.pi

def gen_spiral_dataset(N=100,D=2,K=5):
    '''
    input:
        N: number of point per class
        D: dimension of data
        K: number of class
    output:
        X: generated data (N*K)xD matrix
        y: data lebal (N*K)x1 matrix
    '''
    X=np.zeros((N*K,D))
    y=np.zeros(N*K,dtype='uint8')
    ang=2*PI/K
    for j in range(K):
        ix=range(N*j,N*(j+1))
        r=np.linspace(0.0,1,N)
        t=np.linspace(j*ang,(j+1)*ang,N)+np.random.randn(N)*0.5/K
        X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]
        y[ix]=j
    return X,y

def plot_spiral(X,y,w):
    h=0.02
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),(np.arange(y_min,y_max,h)))
    Z=sfmax.softmax(np.dot(np.c_[np.ones(xx.size),xx.ravel(),yy.ravel()],w))
    Z=np.argmax(Z,axis=1)
    Z=Z.reshape(xx.shape)
    fig=plt.figure()
    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral,alpha=1)
    plt.scatter(X[:,0],X[:,1],c=y,s=20,cmap=plt.cm.spring)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())    

if __name__ ==  '__main__':
    X,y=gen_spiral_dataset()
    X_train=np.c_[(np.ones((X.shape[0],1),dtype=X.dtype)),X]
    y=np.expand_dims(y,-1)
    y_onehot=sfmax.one_hot_represent(y)
    # print(X.shape)
    w=np.zeros((X_train.shape[1],len(np.unique(y))))
    # print(w.shape)
    history,loss_train=sfmax.gradient_descent_softmax(w,X_train,y,0.,0.2,1000)
    # plt.figure()
    plt.plot(loss_train)
    print(loss_train[-1])
    w=history[-1]
    # print(w)
    # print(w.shape)
    acc=sfmax.getAccuracy(w,X_train,y)
    print(acc)
    plot_spiral(X,y,w)
    plt.show()
