import numpy as np

class class_two_layer_ANN:
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.parameters=[]

    def initialize_parameters(self,n_x,n_h,n_o):
        '''
        input:
            n_x: number of input feature
            n_h: number of hidden layer nodes
            n_o: number of output
        '''
        np.random.seed(0)
        w1=np.random.randn(n_x,n_h)*0.01
        b1=np.zeros((1,n_h))

        w2=np.random.randn(n_h,n_o)*0.01
        b2=np.zeros((1,n_o))

        assert(w1.shape==(n_x,n_h))
        assert(b1.shape==(1,n_h))
        assert(w2.shape==(n_h,n_o))
        assert(b2.shape==(1,n_o))

        parameters=[w1,b1,w2,b2]
        self.parameters=parameters
        return parameters
    
    def sigmoid(self,z):
        """
        input:
            wx array in (number of data set,1)
        output:
            calc ans after sigmoid function  (number of data set,1)
        """
        return 1/(1+np.exp(-z))
    
    def forward_prop(self,X,parameters):
        '''
        input:
            X: mxL matrix m:number of data L:number of feature == n_x
            parameters: [w1,b1,w2,b2]
        output:
            z2: mxn matrix m:number of data n:number of class == n_o
        '''
        w1,b1,w2,b2=parameters
        z1=np.dot(X,w1)+b1
        a1=np.tanh(z1)
        z2=np.dot(a1,w2)+b2
        return z2
    
    def softmax(self,x):
        """
        input:
            x mxn matrix m:num of data n:num of class
        output:
            mxn matrix with m:num of softmaxed row
        """
        e_x=np.exp(x-np.max(x,axis=-1,keepdims=True))
        return e_x/np.sum(e_x,axis=-1,keepdims=True)
    
    def softmax_crossentropy(self,z,y):
        '''
        input:
            z mxn matrix with m:num of data n:num of class
            y mx1 matrix 
        output:
            loss double
        '''
        m=len(z)
        F=self.softmax(z)
        log_Fy=np.log(F[range(m),y.transpose()])
        return-np.mean(log_Fy)
    
    def softmax_crossentropy_one_hot(self,z,y):
        '''
        input:
            z mxn matrix with m:num of data n:num of class
            y mxn matrix like [1 0 0]
        output:
            loss double
        '''
        F = self.softmax(z)
        loss = -np.sum(y*np.log(F),axis=1)
        return np.mean(loss)
    
    def softmax_crossentropy_reg(self,z,y,parameters,onehot=False,reg=1e-3):
        w1=parameters[0]
        w2=parameters[2]
        if onehot:
            return self.softmax_crossentropy_one_hot(z,y)+reg*(np.sum(w1**2)+np.sum(w2**2))
        else:
            return self.softmax_crossentropy(z,y)+reg*(np.sum(w1**2)+np.sum(w2**2))
        
    def compute_loss_reg(self,fun,loss_fun,X,y,parameters,onehot=False,reg=1e-3):
        z2=fun(X,parameters)
        return loss_fun(z2,y,parameters,onehot,reg)
    
    def f(self):
        return self.compute_loss_reg(self.forward_prop,self.softmax_crossentropy_reg,self.X,self.y,self.parameters)
    
    def numerical_gradient(self,f,params,eps=1e-10):
        n_grads=[]
        for x in params: #params is a list
            grad=np.zeros(x.shape) #x maybe an array
            it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite']) #輸出的結果為index ex(0,0)for 2D array
            while not it.finished:
                idx=it.multi_index
                old_value=x[idx]
                x[idx]=old_value+eps
                fxp=f()
                x[idx]=old_value-eps
                fxm=f()
                grad[idx]=(fxp-fxm)/(2*eps)
                x[idx]=old_value
                it.iternext()
            n_grads.append(grad)
        return n_grads
    
    def max_abs(self,grads):
        return max([np.max(np.abs(grad)) for grad in grads])
    
    def gradient_descent_ANN(self,alpha=0.01,iter=1000,reg=0.,gamma=0.8,epsilon=1e-8):

        losses=[]
        for i in range(iter):
            loss=self.f()
            grads=self.numerical_gradient(self.f,self.parameters)
            # print(grads)
            # print(self.parameters[1])
            if self.max_abs(grads)<epsilon:
                print("grad is small enough!")
                break

            for i in range(len(self.parameters)):
                self.parameters[i]-=alpha*grads[i]


            losses.append(loss)
        return self.parameters,losses
    
    def get_accuracy(self,X,y,parameters):
        predicts=np.argmax(self.forward_prop(X,parameters),axis=-1)
        accuracy=sum(predicts==y)/(float(len(y)))
        return accuracy

    
    
    


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


if __name__ == '__main__':    
    X=np.array([[1,2],[3,4],[5,6],[7,8]])
    y=np.array([2,1,0,0])
    ann=class_two_layer_ANN(X,y)
    y_o=one_hot_represent(y)
    parameters=ann.initialize_parameters(2,4,3)
    z2=ann.forward_prop(X,parameters)
    loss=ann.softmax_crossentropy(z2,y)
    print(loss)
    loss_o=ann.softmax_crossentropy_one_hot(z2,y_o)
    print(loss_o)
    loss_c=ann.compute_loss_reg(ann.forward_prop,ann.softmax_crossentropy_reg,X,y,parameters)
    print(loss_c)
    num_grad=ann.numerical_gradient(ann.f,parameters)
    print(num_grad[0])
    print(num_grad[3])