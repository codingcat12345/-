import numpy as np

class SGD:
    def __init__(self, model_params, learning_rate=0.01, momentum=0.9):
        """
        initialize SGD optimizer

        input:
            model_params (list of arrays):[[w,dw],[b,db]]model parameter each element in list is a np array
            learning_rate(float)
            momentum(float)
        """
        self.params = model_params
        self.lr = learning_rate
        self.momentum = momentum

        # initialize velocity
        self.vs = []
        for param,_ in self.params:
            v = np.zeros_like(param)
            self.vs.append(v)

    def zero_grad(self):
        """
        reset grad to zero for the next step
        """
        for _, grad in self.params:
            grad.fill(0)

    def step(self):
        """
        do a optimization step
        """
        for (param, grad), v in zip(self.params, self.vs):
            v = self.momentum * v + self.lr * grad
            # update model_param
            param -= v

    def scale_learning_rate(self, scale):
        self.lr *= scale

class Adam:
    def __init__(self, model_params, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        """
        initialize adam optimizer

        input:
            model_params(list of pairs)
            learning_rate(float,optional)
            beta_1(float,optional)
            beta_2(float,optional)
            epsilon(float,optional)
        """
        self.params = model_params
        self.lr = learning_rate
        self.beta_1, self.beta_2, self.epsilon = beta_1, beta_2, epsilon

        self.ms = [np.zeros_like(p) for p, _ in self.params]
        self.vs = [np.zeros_like(p) for p, _ in self.params]
        self.t = 0

    def zero_grad(self):

        for _, grad in self.params:
            grad.fill(0)

    def step(self):
        beta_1, beta_2, lr, t = self.beta_1, self.beta_2, self.lr, self.t
        t += 1
        self.t = t

        for i, (p, grad) in enumerate(self.params):
            m, v = self.ms[i], self.vs[i]
            
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * (grad**2)

            m1 = m / (1 - np.power(beta_1, t))
            v1 = v / (1 - np.power(beta_2, t))

            p -= lr * m1 / (np.sqrt(v1) + self.epsilon)

    def scale_learning_rate(self, scale):
        self.lr *= scale