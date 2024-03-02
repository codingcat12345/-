import numpy as np
# from chapter2 import gradient_descent as gd

def softmax(x):
    """
    input:
        x mxn matrix m:num of data n:num of class
    output:
        mxn matrix with m:num of softmaxed row
    """
    e_x=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return e_x/np.sum(e_x,axis=-1,keepdims=True)

def softmax_crossentropy(z,y):
    '''
    input:
        z mxn matrix with m:num of data n:num of class
        y mx1 matrix 
    output:
        loss double
    '''
    m=len(z)
    # y=y.transpose()
    F=softmax(z)
    log_Fy=np.log(F[range(m),y.transpose()])
    return-np.mean(log_Fy)

def softmax_crossentropy_one_hot(z,y):
    '''
    input:
        z mxn matrix with m:num of data n:num of class
        y mxn matrix like [1 0 0]
    output:
        loss double
    '''
    F = softmax(z)
    loss = -np.sum(y*np.log(F),axis=1)
    return np.mean(loss)

def grad_softmax_crossentropy(z,y):
    '''
    input:
        Z mxn matrix with m:num of data n:num of class
        y mx1 matrix 
    output:
        loss grad dLdz
    '''
    m=len(z)
    y=y.transpose()
    F=softmax(z)
    F[range(m),y]=F[range(m),y]-1
    return F/m

def grad_softmax_crossentropy_one_hot(z,y):
    '''
    input:
        Z mxn matrix with m:num of data n:num of class
        y mxn matrix like [1 0 0]
    output:
        loss grad dLdz
    '''
    m=len(z)
    F=softmax(z)
    return (F-y)/m

def one_hot_represent(y,num_class=0):
    '''
    input:
        y mx1 matrix with m:num of data 
        num_class number of class
    output:
        one hot y
    '''
    num_class=len(np.unique(y))
    m=y.shape[0]
    I=np.zeros((m,num_class))
    I[range(m),y.transpose()]=1
    return I

def gradient_softmax(w,x,y,reg,onehot=False):
    '''
    input:
        w Lxn matrix with L:num of features n: num of class
        x mxL matrix with m:num of data L:num of features
        y mx1 matrix
        reg lambda
    output:
        grad Lxn matrix
    '''
    z = x @ w
    if onehot:
        grad = x.transpose() @ grad_softmax_crossentropy_one_hot(z,y) + 2*reg*w
    else:
        grad = x.transpose() @ grad_softmax_crossentropy(z,y) + 2*reg*w
    return grad

def gradient_descent_softmax(w,X,y,reg=0.,alpha=0.2,num_iter=1000,gamma=0.8,epsilon=1e-6):
    """
    input:
        w Lxn matrix with L:num of features n: num of class
        x mxL matrix with m:num of data L:num of features
        y mx1 matrix
    output:
        w_hisory list of Lxn matrix with L:num of features n: num of class
    """
    w_history=[]
    loss_history=[]
    # X=np.c_[(np.ones((X.shape[0],1),dtype=X.dtype)),X]
    #use momentum method
    v=np.zeros_like(w)
    for i in range(num_iter):
        gradient=gradient_softmax(w,X,y,reg)
        if np.max(np.abs(gradient))<epsilon:
            print("gradient is small enough")
            break
        v=gamma*v+alpha*gradient
        w=w-v
        loss=softmax_crossentropy(X@w,y)+reg*np.sum(w*w)
        w_history.append(w)
        loss_history.append(loss)
    return w_history,loss_history

def gradient_descent_softmax_one_hot(w,X,y,reg=0.,alpha=0.2,num_iter=1000,gamma=0.8,epsilon=1e-6):
    """
    input:
        w Lxn matrix with L:num of features n: num of class
        x mxL matrix with m:num of data L:num of features
        y mxn one hot matrix
    output:
        w_hisory list of Lxn matrix with L:num of features n: num of class
    """
    w_history=[]
    loss_history=[]
    # X=np.c_[(np.ones((X.shape[0],1),dtype=X.dtype)),X]
    #use momentum method
    v=np.zeros_like(w)
    for i in range(num_iter):
        gradient=gradient_softmax(w,X,y,reg,True)
        if np.max(np.abs(gradient))<epsilon:
            print("gradient is small enough")
            break
        v=gamma*v+alpha*gradient
        w=w-v
        loss=softmax_crossentropy_one_hot(X@w,y)+reg*np.sum(w*w)
        w_history.append(w)
        loss_history.append(loss)
    return w_history,loss_history

def getAccuracy(w,X,y):
    """
    input:
        w Lxn matrix with L:num of features n: num of class
        x mxL matrix with m:num of data L:num of features
        y mx1 matrix
    output:
        accuracy float
    """    
    # x=np.c_[(np.ones((X.shape[0],1),dtype=X.dtype)),X]
    probability=softmax(X @ w)
    predicts=np.expand_dims(np.argmax(probability,axis=1),-1)
    accuracy=sum(predicts==y)/(float(len(y)))
    return accuracy[0]

def valid_loss(w_hisory,X_valid,y_valid,onehot=False):
    loss_history=[]
    for w in w_hisory:
        if onehot:
            loss=softmax_crossentropy_one_hot(X_valid@w,y_valid)
        else:
            loss=softmax_crossentropy(X_valid@w,y_valid)
        loss_history.append(loss)
    return loss_history

def softmax_gradient(z,isF=False):
    if isF:
        F=z
    else:
        F=softmax(z)
    D=[]
    for i in range(F.shape[0]):
        f=F[i]
        D.append(np.diag(f.flatten()))
    grad = D - np.einsum('ij,ik->ijk',F,F) #outer product
    return grad[0]

def softmax_backward(z,dF,isF=True):
    grads=softmax_gradient(z,isF)
    grad=np.einsum("bj,bjk->bk",dF,grads)
    return grad
