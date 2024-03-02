import numpy as np

def normal_equation_method(xtrain,ytrain):
    '''
        input:
            xtrain: xdata in shape (m,)
            ytrain: ydata in shape (m,)
        output:
            w: weight (2,1) matrix (w and b for ypred = wx+b)
    '''
    x=np.ones(shape=(len(xtrain),2))
    x[:,0]=xtrain #x=[data 1]
    y=np.reshape(ytrain,(len(ytrain),1))
    xt=x.transpose()
    xty=xt@y
    w=np.linalg.inv(xt@x)@xty
    return w

def gradient_descent_method(x,y,w,b,alpha,iteration,epsilon=1e-8):
    history=[]
    for i in range(iteration):
        dL_dw=np.mean((w*x+b-y)*x)
        dL_db=np.mean(w*x+b-y)
        if max(abs(dL_dw),abs(dL_db))<epsilon:
            print("i=%d grad is small enough"%i)
            break
        w=w-alpha*dL_dw
        b=b-alpha*dL_db
        history.append([w,b])
    return history

def loss(x,y,w,b):
    m=len(x)
    cost=0
    for i in range(m):
        fx=x[i]*w+b
        cost=cost+(fx-y[i])**2
    cost=cost/2/m
    return cost
