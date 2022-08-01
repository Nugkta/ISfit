# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:04:36 2022

@author: pokey

this file is for learning the routines

"""

import numpy as np
arr2D = np.array([[11, 12, 13, 22], [21, 7, 23, 14], [31, 10, 33, 7]])

sortedArr = arr2D[arr2D[:,1].argsort()]



#%%
a = np.array([[0.46162934, 0.67833399, 0.87730068]])
b = np.array([[0.10153951, 0.70881156, 0.38736128]])


c = np.stack((a,b), axis = -1)



#%%
from sympy import symbols, Eq, solve

x = symbols('x')
eq1 = Eq(2*x**2 + x + 1, 0)


sol = solve(eq1)




















































