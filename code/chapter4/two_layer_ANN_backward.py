import numpy as np
import sys
import os
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter3 import softmax_regression as sfmx

def sigmoid(x):
        return 1./(1.+np.exp(-x))

def sigmoid_grad(a):
        return a*(1-a)

def mse_loss(f,y):
    m=len(f)
    loss=(1./m)*np.sum((f-y)**2)
    return loss

def mse_loss_grad(f,y):
    m=len(f)
    grad=(2./m)*(f-y)
    return grad

def abs_max(s):
    max_value=0
    for x in s:
        max_value_=np.max(np.abs(x))
        if (max_value_>max_value):
            max_value=max_value_
    return max_value


class NN:
    def __init__(self,input_unit,hidden_unit,output_unit):
        n=input_unit
        h=hidden_unit
        K=output_unit

        self.w1=0.01*np.random.randn(n,h)
        self.b1=np.zeros((1,h))
        self.w2=0.01*np.random.randn(h,K)
        self.b2=np.zeros((1,K))

    

    def train(self,X,y,reg=0,iter=10000,alpha=0.1,epsilon=1e-8):
        print("reg",reg,"iter",iter)
        w1=self.w1
        b1=self.b1
        w2=self.w2
        b2=self.b2
        loss_history=[]
        for i in range(iter):
            #forward
            z1=np.dot(X,w1)+b1
            A1=sigmoid(z1)
            z2=np.dot(A1,w2)+b2
            A2=sigmoid(z2)

            #loss
            if i%100==0:
                data_loss=sfmx.softmax_crossentropy(A2,y)
                reg_loss=reg*(np.sum(w1*w1)+np.sum(w2*w2))
                loss=data_loss+reg_loss
                loss_history.append(loss)
                # print("iter= %d loss: %f" %(i,loss))

            #backward
            dL_dA2=sfmx.grad_softmax_crossentropy(A2,y)
            dA2_dz2=sigmoid_grad(A2)
            dL_dz2=np.multiply(dL_dA2,dA2_dz2)
            dw2=np.dot(A1.T,dL_dz2)
            db2=np.sum(dL_dz2,axis=0,keepdims=True)
            dL_dA1=np.dot(dL_dz2,w2.T)
            dA1_dz1=sigmoid_grad(A1)
            dL_dz1=np.multiply(dL_dA1,dA1_dz1)
            dw1=np.dot(X.T,dL_dz1)
            db1=np.sum(dL_dz1,axis=0,keepdims=True)

            if abs_max([dw2,db2,dw1,db2])<epsilon:
                print("gradient is small enough at iter:",i)
                break
            #update w
            w1 += -alpha*dw1
            b1 += -alpha*db1
            w2 += -alpha*dw2
            b2 += -alpha*db2

        self.w1=w1
        self.b1=b1
        self.w2=w2
        self.b2=b2
        return w1,b1,w2,b2,loss_history
    
    def predict(self,X):
        z1=np.dot(X,self.w1)+self.b1
        A1=sigmoid(z1)
        z2=np.dot(A1,self.w2)+self.b2
        A2=sigmoid(z2)
        return A2
    
    def get_accuracy(self,X,y):
        A2=self.predict(X)
        predict_class=np.argmax(A2,axis=1)
        return np.mean(predict_class==y)