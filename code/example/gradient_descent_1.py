import sys
import os
import numpy as np
import matplotlib.pyplot as plt
pathnow=os.getcwd()
sys.path.append(pathnow)
from chapter2 import Optimizator as op
from chapter2 import gradient_descent as gd

if __name__ ==  '__main__':

    f=lambda x: x[0]**2/16+9*x[1]**2
    f2=lambda x,y:x**2/16+9*y**2
    x0=np.array([-2.4,2.4])

    minima=np.array([0.,0.]) #as float
    minima_=minima.reshape(-1,1)
    xmin,xmax,xstep=-4.,4.,0.2
    ymin,ymax,ystep=-4.,4.,0.2
    x_list=np.arange(xmin,xmax+xstep,xstep)
    y_list=np.arange(ymin,ymax+ystep,ystep)
    x,y=np.meshgrid(x_list,y_list)
    z=f2(x,y)

    optimizator=op.SGD_adam([x0],0.01,0.8,0.8)
    path=gd.gradient_descent(f,optimizator,1000)
    print(path[-1])
    path=np.asarray(path)
    path=np.reshape(path,(path.shape[0],2,))
    ax=gd.plot_path(path,x,y,z,minima_,xmin,xmax,ymin,ymax,'k')
    plt.show()