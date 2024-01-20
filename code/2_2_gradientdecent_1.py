import numpy as np
import matplotlib.pyplot as plt
def gradient_decent(df,x,alpha,iteration,epsilon): # alpha is step
  history=[x] # history is a list
  for i in range(iteration):
    if abs(df(x))<epsilon:
      print("k=%d gradient is small enough"%i)
      break
    x=x-alpha*df(x)
    history.append(x)
  return history
f=lambda x: x**3-3*x**2-9*x+2
df=lambda x: 3*x**2-6*x-9 # simple way to def a function
path=gradient_decent(df,0,0.05,200,1e-8)
print(path[-1])
x=np.arange(-3,4,0.01)
y=f(x)
plt.plot(x,y)
path_x=np.asarray(path) # change list to array
path_y=f(path_x)
plt.quiver(path_x[:-1],path_y[:-1],path_x[1:]-path_x[:-1],path_y[1:]-path_y[:-1],
           scale_units='xy',angles='xy',scale=1,color='k')
plt.scatter(path[-1],f(path[-1]))
plt.show()
