import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

f=lambda x,y:(1.5-x+x*y)**2+(2.25-x+x*y**2)**2+(2.625-x+x*y**3)**2
minima=np.array([3.0,0.5]) #as float
minima_=minima.reshape(-1,1) #the -1 here means the row number is created automatically

xmin,xmax,xstep=-4.5,4.5,0.2
ymin,ymax,ystep=-4.5,4.5,0.2

x_list=np.arange(xmin,xmax+xstep,xstep)
y_list=np.arange(ymin,ymax+ystep,ystep)

x,y=np.meshgrid(x_list,y_list)
z=f(x,y)

#dfx=lambda x,y: 2*(1.5-x+x*y)*(y-1)+2*(2.5-x+x*y**2)*(y**2-1)+2*(2.625-x+x*y**3)*(y**3-1)
#dfy=lambda x,y: 2*(1.5-x+x*y)*x+2*(2.5-x+x*y**2)*(2*x*y)+2*(2.625-x+x*y**3)*(3*x*y**2)

xv=np.vstack((x.reshape(1,-1),y.reshape(1,-1)))
df=lambda x:np.array([2*(1.5-x[0]+x[0]*x[1])*(x[1]-1)+2*(2.5-x[0]+x[0]*x[1]**2)*(x[1]**2-1)\
+2*(2.625-x[0]+x[0]*x[1]**3)*(x[1]**3-1),2*(1.5-x[0]+x[0]*x[1])*x[0]+2*(2.5-x[0]+x[0]\
*x[1]**2)*(2*x[0]*x[1])+2*(2.625-x[0]+x[0]*x[1]**3)*(3*x[0]*x[1]**2)])

def gradient_decent(df,x,alpha,iteration,epsilon): # alpha is step
  history=[x] # history is a list
  for i in range(iteration):
    if np.max(abs(df(x)))<epsilon:
      print("iter={i}","gradient is small enough")
      break
    x=x-alpha*df(x)
    history.append(x)
  return history

def gradient_decent_mom(df,x,alpha,gamma,iteration,epsilon):
  history=[x]
  v0=np.zeros_like(x)
  for i in range(iteration):
   if np.max(abs(df(x)))<epsilon:
    print("iter=%d"%i,"gradient is small enough")
    break
   v=gamma*v0+alpha*df(x)
   x=x-v
   v0=v
   history.append(x)
  return history

def gradient_decent_adadelta(df,x,alpha,rho,iteration,epsilon):
  history=[x]
  Eg=np.ones_like(x)
  Edelta=np.ones_like(x)
  for i in range(iteration):
   if np.max(abs(df(x)))<epsilon:
    print("iter={i}","gradient is small enough")
    break
   grad=df(x)
   Eg=rho*Eg+(1-rho)*(grad**2)
   delta=np.sqrt((Edelta+epsilon)/(Eg+epsilon))*grad
   x=x-alpha*delta
   Edelta=rho*Edelta+(1-rho)*(delta**2)
   history.append(x)
  return history

def gradient_decent_adam(df,x,alpha,beta1,beta2,iteration,epsilon):
  history=[x]
  m=np.zeros_like(x)
  v=np.zeros_like(x)
  for t in range(iteration):
   if np.max(abs(df(x)))<epsilon:
    print("iter={i}","gradient is small enough")
    break
   grad=df(x)
   m=beta1*m+(1-beta1)*(grad)
   v=beta2*v+(1-beta2)*(grad**2)
   if t != 0:
    m_h=m/(1-np.power(beta1,t))
    v_h=(1-np.power(beta2,t))
   else:
    m_h=m
    v_h=v
   x=x-alpha*m_h/(np.sqrt(v_h)+epsilon)
   history.append(x)
  return history

def plot_path(path,x,y,z,minima_,xmin,xmax,ymin,ymax,c):
  fig,ax=plt.subplots(figsize=(10,6))
  ax.contour(x,y,z,levels=np.logspace(0,5,35),norm=LogNorm(),cmap=plt.cm.jet)
  ax.quiver(path[:-1,0],path[:-1,1],path[1:,0]-path[:-1,0],path[1:,1]-
            path[:-1,1],scale_units='xy',angles='xy',scale=1,color=c)
  ax.plot(*minima_,'r*',markersize=18)
  ax.set_xlim((xmin,xmax))
  ax.set_ylim((ymin,ymax))
  return ax

def add_plot_path(path,ax,xmin,xmax,ymin,ymax,c):
  ax.quiver(path[:-1,0],path[:-1,1],path[1:,0]-path[:-1,0],path[1:,1]-
            path[:-1,1],scale_units='xy',angles='xy',scale=1,color=c)
  ax.set_xlim((xmin,xmax))
  ax.set_ylim((ymin,ymax))

x0=np.array([3.0,4.0])
path=gradient_decent(df,x0,0.000005,500000,1e-8)
path2=gradient_decent_mom(df,x0,0.000005,0.8,500000,1e-8)
path3=gradient_decent_adadelta(df,x0,1.0,0.9,500000,1e-8)
path4=gradient_decent_adam(df,x0,0.000005,0.9,0.8,500000,1e-8)

path=np.asarray(path)
ax=plot_path(path,x,y,z,minima_,xmin,xmax,ymin,ymax,'k')
path2=np.asarray(path2)
add_plot_path(path2,ax,xmin,xmax,ymin,ymax,'m')
path3=np.asarray(path3)
add_plot_path(path3,ax,xmin,xmax,ymin,ymax,'b')
path4=np.asarray(path4)
add_plot_path(path4,ax,xmin,xmax,ymin,ymax,'r')
