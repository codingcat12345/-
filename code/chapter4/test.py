
import sys
import os
pathnow=os.getcwd()
pathup=os.path.abspath(os.path.dirname(os.getcwd()))
print("123")
print(pathnow)
sys.path.append(pathnow)

from chapter2 import Optimizator as op


if __name__ ==  '__main__':
    op.qq()
