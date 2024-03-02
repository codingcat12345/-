import numpy as np

class optimizator:
  def __init__(self,params):
    self.params=params
  def step(self,grads):
    pass
  def parameters(self):
    #print(type(self.params))
    return self.params
  
class SGD(optimizator):
  def __init__(self,params,learning_rate):
    super().__init__(params) #調用optimizator性質一
    self.lr=learning_rate

  def step(self,grads,t):
    for i in range(len(self.params)):
      self.params[i]=self.params[i]-self.lr*grads[i] #grads 也是用list送進來
    return self.params
  
class SGD_momentum(optimizator):
  def __init__(self,params,learning_rate,gamma):
    super().__init__(params)
    self.ga=gamma
    self.lr=learning_rate
    self.v=[]
    for param in params: # initialize v to 0
     self.v.append(np.zeros_like(param))

  def step(self,grads,t):
    for i in range(len(self.params)):
     self.v[i]=self.v[i]*self.ga+self.lr*grads[i] #grads 也是用list送進來
     self.params[i]=self.params[i]-self.v[i]
    return self.params
  
class SGD_adam(optimizator):
  def __init__(self, params,learning_rate,beta1,beta2):
    super().__init__(params)
    self.beta1=beta1
    self.beta2=beta2
    self.lr=learning_rate
    self.m=[]
    for param in params: # initialize m to 0
      self.m.append(np.zeros_like(param))
    self.v=[]
    for param in params: # initialize v to 0
      self.v.append(np.zeros_like(param))

  def step(self,grads,t):
    for i in range(len(self.params)):
      self.m[i]=self.beta1*self.m[i]+(1-self.beta1)*grads[i]
      self.v[i]=self.beta2*self.v[i]+(1-self.beta2)*np.power(grads[i],2)
      m_t=self.m[i]/(1-np.power(self.beta1,t+1))
      v_t=self.v[i]/(1-np.power(self.beta2,t+1))
      self.params[i]=self.params[i]-self.lr*m_t/(np.sqrt(v_t)+1e-8)
    return self.params

def qq():
  print("QQ")
    
