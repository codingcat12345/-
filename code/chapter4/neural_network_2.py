import numpy as np

class Layer:
    def __init__(self):
        self.params = None

    def forward(self, x):
        raise NotImplementedError  

    def backward(self, dZ):
        raise NotImplementedError  

    def reg_grad(self, reg):
        pass  

    def reg_loss(self, reg):
        return 0.0 
    
class Dense(Layer):
    def __init__(self, input_dim, out_dim, init_method=('random', 0.01)):
        super().__init__()
        random_method_name, random_value = init_method
        if random_method_name == "random":
            # use random value as weight matrix and bias
            self.w = np.random.rand(input_dim, out_dim) * random_value
            self.b = np.random.rand(1, out_dim) * random_value
        elif random_method_name == "he":
            # use He method to set wieght matrix and set bias as 0
            self.w = np.random.randn(input_dim, out_dim) * np.sqrt(2 / input_dim)
            self.b = np.zeros((1, out_dim))
        elif random_method_name == "xavier":
            # use Xavier method to set wieght matrix and set bias as random value
            self.w = np.random.randn(input_dim, out_dim) * np.sqrt(1 / input_dim)
            self.b = np.random.rand(1, out_dim) * random_value
        elif random_method_name == "zeros":
            # set weight matrix and bias to 0
            self.w = np.zeros((input_dim, out_dim))
            self.b = np.zeros((1, out_dim))
        else:
            # by default use random method
            self.w = np.random.rand(input_dim, out_dim) * random_value
            self.b = np.zeros((1, out_dim))

        # use lists to contain parameters and grads
        self.params = [self.w, self.b]
        self.grads = [np.zeros_like(self.w), np.zeros_like(self.b)]

    def forward(self, x):
        self.x=x
        x_flattened=x.reshape(x.shape[0], np.prod(x.shape[1:]))
        Z=np.matmul(x_flattened,self.w)+self.b
        return Z
    
    def backward(self,dL_dz):
        x_in=self.x
        x_in_flattened=x_in.reshape(x_in.shape[0], np.prod(x_in.shape[1:]))
        self.grads[0]+=np.dot(x_in_flattened.T,dL_dz)
        self.grads[1]+=np.sum(dL_dz,axis=0,keepdims=True)
        dL_dx_in=np.dot(dL_dz,np.transpose(self.w))
        dL_dx_in=dL_dx_in.reshape(x_in.shape)
        return dL_dx_in
    
    def reg_grad(self, reg):    
        self.grads[0] += 2 * reg * self.w

    def reg_loss(self, reg):   
        return reg * np.sum(self.w**2)

    def reg_loss_grad(self, reg):
        self.grads[0] += 2 * reg * self.w
        return reg * np.sum(self.w**2)
    
class Relu(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        relu_grad = self.x > 0
        return grad_output * relu_grad

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, grad_output):
        sigmoid_output = 1.0 / (1.0 + np.exp(-self.x))
        return grad_output * sigmoid_output * (1 - sigmoid_output)

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        self.a = np.tanh(x)
        return self.a

    def backward(self, grad_output):
        d = 1 - np.square(self.a)
        return grad_output * d

class LeakyRelu(Layer):
    def __init__(self, leaky_slope):
        super().__init__()
        self.leaky_slope = leaky_slope

    def forward(self, x):
        self.x = x
        return np.maximum(self.leaky_slope * x, x)

    def backward(self, grad_output):
        d = np.zeros_like(self.x)
        d[self.x <= 0] = self.leaky_slope
        d[self.x > 0] = 1
        return grad_output * d
    
class neuralnetwork2:
    def __init__(self):
        self._layers = []  
        self._params = []  # contain layer params and layer grad [[w dw],[b db]]

    def add_layer(self, layer):
        self._layers.append(layer)
        if layer.params:
            for i, _ in enumerate(layer.params):
                self._params.append([layer.params[i], layer.grads[i]])

    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        # when use NN(x) it will retrun NN.forward(x)
        return self.forward(x)

    def predict(self, x):
        D = self.forward(x)
        if D.ndim == 1:
            return np.argmax(D)
        return np.argmax(D, axis=1)

    def backward(self, loss_grad, reg=0.):
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            loss_grad = layer.backward(loss_grad)
            layer.reg_grad(reg)
        return loss_grad

    def reg_loss(self, reg):
        reg_loss = 0
        for i in range(len(self._layers)):
            reg_loss += self._layers[i].reg_loss(reg)
        return reg_loss

    def parameters(self):
        return self._params

    def zero_grad(self):
        #reset all layers' grad
        for i, _ in enumerate(self._params):
            self._params[i][1][:] = 0

    def get_parameters(self):
        return self._params

    def save_parameters(self, filename):
        params = {}
        for i in range(len(self._layers)):
            if self._layers[i].params:
                params[i] = self._layers[i].params
        np.save(filename, params)

    def load_parameters(self, filename):
        params = np.load(filename, allow_pickle=True).item()
        count = 0
        for i in range(len(self._layers)):
            if self._layers[i].params:
                layer_params = params.get(i)
                self._layers[i].params = layer_params
                for j in range(len(layer_params)):
                    self._params[count][0] = layer_params[j]
                    count += 1

def data_iter(x,y,batch_size,if_shuffle=False):
    m=len(x)
    indices=list(range(m))
    if if_shuffle:
        np.random.shuffle(indices)

    for i in range(0,m-batch_size+1,batch_size):
        batch_indices=np.array(indices[i: min(i+batch_size,m)])
        yield x.take(batch_indices,axis=0),y.take(batch_indices,axis=0)

def tran_nn(nn,X,Y,optimizer,loss_fun,epochs=2,batch_size=10,if_shuffle=False,print_n=100,reg=0.):
    # iter_time=0
    losses=[]
    for epoch in range(epochs):
        for x,y in data_iter(X,Y,batch_size,if_shuffle):
            optimizer.zero_grad()
            f=nn(x)
            loss,loss_grad=loss_fun(f,y)
            nn.backward(loss_grad,reg)
            loss += nn.reg_loss(reg)

            optimizer.step()

            losses.append(loss) 
        if epoch%print_n==0:
            f=nn(X)
            loss,loss_grad=loss_fun(f,Y)
            print("epoch=",epoch,"loss=",loss)
    return losses