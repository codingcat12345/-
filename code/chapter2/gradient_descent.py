import numpy as np
from .Optimizator import optimizator
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def gradient_descent(f,optimizator,iterations,epsilon=1e-2):
    x=optimizator.parameters() #回傳parameters給x,
    x=x.copy()
    history=[x]
    grad=numerical_gradient(f,x,1e-8)[0]
    for i in range(iterations):
        if np.max(np.abs(grad))<epsilon:
            print("k=%d grad is small enough"%i)
            break
        grad=numerical_gradient(f,x,1e-8)[0]
        x=optimizator.step([grad],i)
        x=x.copy()
        history.append(x)
    return history
    
def numerical_gradient(f,params,eps=1e-10):
    n_grads=[]
    for x in params: #params is a list
        def fun():
            return f(x) #呼叫fun()就會返回f(x)的值，其中x是在外面的變數而f也是外面的函式，需要先定義
        grad=np.zeros(x.shape) #x maybe an array
        it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite']) #輸出的結果為index ex(0,0)for 2D array
        while not it.finished:
            idx=it.multi_index
            old_value=x[idx]
            x[idx]=old_value+eps
            fxp=fun()
            x[idx]=old_value-eps
            fxm=fun()
            grad[idx]=(fxp-fxm)/(2*eps)
            x[idx]=old_value
            it.iternext()
        n_grads.append(grad)
    return n_grads

def plot_path(path,x,y,z,minima_,xmin,xmax,ymin,ymax,c):
  fig,ax=plt.subplots(figsize=(10,6))
  ax.contour(x,y,z,levels=np.logspace(0,5,35),norm=LogNorm(),cmap=plt.cm.jet)
  ax.quiver(path[:-1,0],path[:-1,1],path[1:,0]-path[:-1,0],path[1:,1]-
            path[:-1,1],scale_units='xy',angles='xy',scale=1,color=c)
  ax.plot(*minima_,'r*',markersize=18)
  ax.set_xlim((xmin,xmax))
  ax.set_ylim((ymin,ymax))
  return ax
   