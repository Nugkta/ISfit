# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:10:37 2022

@author: pokey


In this file, I will try to let the user to be guide by bottons on the plots instead of running cell by cell.


"""
import pero_model_fit4 as pmf
import init_guess_plot4 as igp 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import glob 
from matplotlib.widgets import TextBox,Slider, Button,CheckButtons
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit

dfs = []
for file in glob.glob('paperdata/**.xlsx'): 
    df = pd.read_excel(file)
    df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]
    df['z_imag'] = -df['z_imag'].values
    dfs.append(df)

for df in dfs:
    df['impedance'] = df['z_real'].values + df['z_imag'].values * 1j

dfs.sort(key = lambda x: x['bias voltage'][0])  # making the dfs list sorted by the magnitude of the bias voltaege of each data set.

wlist = np.logspace(-6,6,1000)
a = 2 #change this to change the set of data to fit
v = [0,.795,.846,.894]

dfs=dfs[0:4]
df = dfs[a]
v = v[a]

#%%
df_1 = dfs[2]






#%%

igp.__main__(dfs)





#%%
df = dfs[0]
init = igp.init_guess_find_0V(df)

















