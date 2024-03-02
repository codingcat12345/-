import numpy as np
import chapter2
from chapter2 import Optimizator as op


a=np.arange(1,10)
b=np.arange(11,20)
a=np.expand_dims(a,1)
print(a.shape)
print(1-a)