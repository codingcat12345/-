import numpy as np
import sys
import os
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter3 import softmax_regression as sfmx
from chapter4 import neural_network as nnfile

def data_iter(x,y,batch_size,if_shuffle=False):
    m=len(x)
    indices=list(range(m))
    if if_shuffle:
        np.random.shuffle(indices)

    for i in range(0,m-batch_size+1,batch_size):
        batch_indices=np.array(indices[i: min(i+batch_size,m)])
        yield x.take(batch_indices,axis=0),y.take(batch_indices,axis=0)

def train_batch(nn,X,Y,loss_fun,epochs=2,batch_size=10,alpha=0.1,if_shuffle=False,reg=0.,print_n=100):

    for epoch in range(epochs):
        # print("epoch",epoch)
        # for_i=1
        for x,y in data_iter(X,Y,batch_size,if_shuffle):
            # print("for_i=",for_i,"xshape=",x.shape)
            # print("y=",y)
            # for_i+=1
            f=nn.forward_prop(x)
            loss,loss_grad=loss_fun(f,y)
            loss += nn.reg_loss(reg)
            nn.backward_prop(loss_grad,reg)
            nn.update_parameter(alpha)

        if epoch%print_n==0:
            f=nn.forward_prop(X)
            loss,loss_grad=loss_fun(f,Y)
            print("epoch=",epoch,"loss=",loss)
                # print(nn.predict(x))


def cross_entropy_loss_fun(f,y):
    loss=sfmx.softmax_crossentropy(f,y)
    loss_grad=sfmx.grad_softmax_crossentropy(f,y)
    return loss,loss_grad

if __name__ == '__main__':
    np.random.seed(1)
    x=np.random.randn(4,2)
    y=np.array([1,0,1,0])

    nn=nnfile.neuralnetwork()
    nn.add_layer(nnfile.Dense(2,100,'relu'))
    nn.add_layer(nnfile.Dense(100,3,'sigmoid'))

    train_batch(nn,x,y,cross_entropy_loss_fun)