# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 23:26:48 2022

@author: pokey
"""

import read_func as rf
import pero_ig_fit_func  as pif

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import pandas as pd
import pandas as pd
from scipy.signal import argrelextrema




#%%
dfs = rf.read_imp_folder('MAPIdev2p5DRIEDLiCl25C/')


#%%
init_guess = pif.get_init_guess(dfs)



#%%
dft= dfs[0]
maxi = argrelextrema(np.imag(dft['impedance'].values),np.less)
x = np.real(dft['impedance'].values)
y = np.imag(dft['impedance'].values)

plt.plot(x,-y,'.')
plt.plot(x[maxi],-y[maxi],'r.')



































