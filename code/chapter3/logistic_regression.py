import numpy as np

def sigmoid(z):
    """
    input:
        wx array in (number of data set,1)
    output:
        calc ans after sigmoid function  (number of data set,1)
    """
    return 1/(1+np.exp(-z))

def gradient_descent_logistic_reg(X,y,lambda_,alpha,num_iter,gamma=0.8,epsilon=1e-6):
    """
    input:
        X mxn matrix
        y mx1 matrix
    output:
        w_hisory list of nx1 matrix
    """
    w_history=[]
    # y=np.expand_dims(y,-1)
    X=np.c_[(np.ones((X.shape[0],1),dtype=X.dtype)),X]
    num_features=X.shape[1]
    Xt=X.transpose()
    #use momentum
    v=np.zeros((num_features,1))
    w=np.zeros((num_features,1))
    for n in range(num_iter):
        z=X @ w
        # print("z",z.shape)
        predictions = sigmoid(z)
        errors = predictions-y
        # print("e",errors)
        gradient =  Xt @ errors / len(y) + 2*lambda_*w
        # print(gradient.shape)
        if np.max(np.abs(gradient))<epsilon:
            print("gradient is small enough")
            break

        v=gamma*v+alpha*gradient
        w=w-v

        w_history.append(w)

    return w_history

def loss_logistic(w,X,y,reg=0.):
    """
    input:
        w (n+1)*1 matrix
        X m*n matrix
        y m*1 matrix
    output:
        loss double
    """
    # y=np.expand_dims(y,1)
    f = sigmoid(X @ w[1:]+w[0])
    u=(np.log(f)*y+np.log(1-f)*(1-y))
    loss = -np.mean(u)
    loss += reg*(np.sum(np.square(w)))
    return loss

def loss_logistic_history(w_hisory,X,y,reg=0.):
    loss_history=[]
    for w in w_hisory:
        loss_history.append(loss_logistic(w,X,y,reg))
    return loss_history
