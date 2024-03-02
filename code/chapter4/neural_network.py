import numpy as np

class Layer:
    def __init__(self):
        pass
    def forward(self,x):
        raise NotImplementedError
    def backward(self,grad):
        raise NotImplementedError
    
class Dense(Layer):
    def __init__(self,input_dim,output_dim,activation=None):
        super().__init__()
        self.w=0.01*np.random.randn(input_dim,output_dim)
        self.b=np.zeros((1,output_dim))
        self.activation=activation
        self.A=None

    def active(self,z):
        if self.activation=='relu':
            return np.maximum(0,z)
        elif self.activation=='sigmoid':
            return 1/(1+np.exp(-z))
        elif self.activation=='tanh':
            return np.tanh(z)
        else:
            return z
        
    def forward(self, x):
        self.x=x
        Z=np.matmul(x,self.w)+self.b
        self.Z=Z
        self.A=self.active(Z)
        return self.A
    
    def backward(self,dL_dA):
        A_in=self.x
        dL_dz=self.dif_z(dL_dA)

        self.dw=np.dot(A_in.T,dL_dz)
        self.db=np.sum(dL_dz,axis=0,keepdims=True)

        dL_dA_in=np.dot(dL_dz,np.transpose(self.w))
        return dL_dA_in
    
    def dif_z(self,dL_dA):
        if self.activation=='relu':
            dA_dZ=1.*(self.Z>0)
            return np.multiply(dL_dA,dA_dZ)
        elif self.activation=='sigmoid':
            dA_dZ=self.A*(1-self.A)
            return np.multiply(dL_dA,dA_dZ)
        elif self.activation=='tanh':
            dA_dZ=1-np.square(self.A)
            return np.multiply(dL_dA,dA_dZ)
        else:
            return dL_dA

class neuralnetwork:
    def __init__(self):
        self.layers=[]

    def add_layer(self,layer):
        self.layers.append(layer)

    def forward_prop(self,x):
        self.x=x
        x_f=x
        for layer in self.layers:
            x_f=layer.forward(x_f)
        return x_f
    
    def predict(self,x):
        p=self.forward_prop(x)
        if p.ndim==1:
            return np.argmax(p)
        return np.argmax(p,axis=1)
    
    def backward_prop(self,loss_grad,reg=0.):
        loss_grad_back=loss_grad
        for i in reversed(range(len(self.layers))):
            layer=self.layers[i]
            loss_grad_back=layer.backward(loss_grad_back)

            layer.dw+=2*reg*layer.w
    
    def update_parameter(self,alpha):
        for i in range(len(self.layers)):
            self.layers[i].w -= alpha*self.layers[i].dw
            self.layers[i].b -= alpha*self.layers[i].db

    def reg_loss(self, reg):
        loss = 0
        for i in range(len(self.layers)):
            loss += reg*np.sum(self.layers[i].w*self.layers[i].w)
        return loss
    
    def parameters(self):
        params = []
        for i in range(len(self.layers)):
            params.append(self.layers[i].w)
            params.append(self.layers[i].b)
        return params

    def grads(self):
        grads = []
        for i in range(len(self.layers)):
            grads.append(self.layers[i].dw)
            grads.append(self.layers[i].db)
        return grads
    
        
def numerical_grad_from_dL_dA(Act,z,dL_dA,h=1e-5):
    numerical_grad=np.zeros_like(z)
    it = np.nditer(z,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx=it.multi_index

        old_val=z[idx]
        z[idx]=old_val+h
        pos=Act()
        z[idx]=old_val-h
        neg=Act()
        z[idx]=old_val
        numerical_grad[idx]=np.sum((pos-neg)*dL_dA)/(2*h)
        it.iternext()
    return numerical_grad

if __name__ == '__main__':
    np.random.seed(1)
    x=np.random.randn(3,48)# 3 data set with 48 features
    dense=Dense(2,10,'tanh')
    o=dense.forward(x)
    dx=dense.backward(o)
    dx_num=numerical_grad_from_dL_dA(lambda :dense.forward(x),x,o)
    print(o.shape)
    # print("dx",dx,"dx_num",dx_num)
    # print(dx.shape)
    # print(dx_num.shape)
    # print(np.max(np.abs(dx-dx_num)))
    nn=neuralnetwork()
    nn.add_layer(Dense(2,100,'relu'))
    nn.add_layer(Dense(100,3,'sigmoid'))
    o_nn=nn.forward_prop(x)
    print(o_nn.shape)
    p=nn.predict(x)
    print(p)