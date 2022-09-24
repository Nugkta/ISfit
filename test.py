# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 00:37:58 2022

@author: pokey 

In this file, I will test the tool with different simulated data
"""

import Main8 as main
import function_tree8 as ft
import pero_model_fit8 as pmf
import init_guess_plot8 as igp 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
def get_clean_data_dfs(dfs): # preprocessing the data, but not from excel. From df directly
    for df in dfs:
        df['impedance'] = df['Z_real'].values + df['Z_imag'].values * 1j
        
    dfs.sort(key = lambda x: x['bias voltage'][0]) # sort the lis tof dataframe by its bias voltage
    
    return dfs


#%%
C_A_0 = 2e-6
C_ion_0 = 1e-6
R_ion = 6e4
C_g = 1e-7
J_s = 1e-11
nA = 1.4
V_bi = 1.1
R_s = 10
R_shnt = 1e5 





#%% 
Vb = [0.1, 0.3, 0.5]
wlist = np.logspace(-3, 5, 40)
dfs = []


for i in Vb:
    vlist = np.ones(len(wlist)) * i
    wvlist = np.stack((wlist, vlist) , axis = -1)
    Z_simu, J_simu = pmf.pero_model_glob(wvlist, C_A_0, C_ion_0, R_ion, C_g, J_s, nA, V_bi, R_s, R_shnt)
    df = pd.DataFrame()
    df['frequency'] = wlist
    df['Z_real'] = np.real(Z_simu)
    df['Z_imag'] =  np.imag(Z_simu)
    df['bias voltage'] = vlist 
    df['recomb current'] = J_simu
    dfs.append(df) 

dfs_gn = get_clean_data_dfs(dfs)

ft.global_no0V(dfs_gn)






















