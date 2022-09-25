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
#C_A_0 = 2e-6
C_A_0 = 3e-6
C_ion_0 = 1e-6
R_ion = 6e4
C_g = 1e-7
J_s = 1e-10
nA = 1.4
#V_bi = 1.1
V_bi = 1.4
R_s = 10
R_shnt = 1e6


C_A = 2e-6
C_ion = 1e-6



#%% global non 0


# Setting some initial parameters
Vb = [0.3, 0.6, 0.9,1.1] # List of bias voltage
Vb = [0.3, 0.6, 0.9] # List of bias voltage
# Vb = [0.1, 0.3, 0.5]# This s
num_of_data = 40
wlist = np.logspace(-3,6, num_of_data)
wlist=wlist[::-1]
dfs = []
mu, sigma = 0, .0001

#adding the noise



for i in Vb:
    vlist = np.ones(len(wlist)) * i
    wvlist = np.stack((wlist, vlist) , axis = -1)
    Z_simu, J_simu = pmf.pero_model_glob(wvlist, C_A_0, C_ion_0, R_ion, C_g, J_s, nA, V_bi, R_s, R_shnt)
    df = pd.DataFrame()
    df['frequency'] = wlist
    # adding noise to the Z
    noise_real =  np.random.normal(mu, sigma, num_of_data)
    noise_imag =  np.random.normal(mu, sigma, num_of_data)
    
    df['Z_real'] = np.real(Z_simu)*(1 + noise_real)
    df['Z_imag'] =  np.imag(Z_simu)*(1 + noise_imag)
    df['bias voltage'] = vlist 
    df['recomb current'] = J_simu
    dfs.append(df) 

dfs_gn = get_clean_data_dfs(dfs)

ft.global_no0V(dfs_gn)





#%%in

Vb = [0.9] # List of bias voltage
# Vb = [0.1, 0.3, 0.5]# This s
num_of_data = 40
wlist = np.logspace(-3, 6, num_of_data)
wlist=wlist[::-1]
dfs = []
mu, sigma = 0, .005

#adding the noise



for i in Vb:
    vlist = np.ones(len(wlist)) * i
    wvlist = np.stack((wlist, vlist) , axis = -1)
    Z_simu, J_simu = pmf.pero_model_glob(wvlist, C_A_0, C_ion_0, R_ion, C_g, J_s, nA, V_bi, R_s, R_shnt)
    df = pd.DataFrame()
    df['frequency'] = wlist
    # adding noise to the Z
    noise_real =  np.random.normal(mu, sigma, num_of_data)
    noise_imag =  np.random.normal(mu, sigma, num_of_data)
    
    df['Z_real'] = np.real(Z_simu)*(1 + noise_real)
    df['Z_imag'] =  np.imag(Z_simu)*(1 + noise_imag)
    df['bias voltage'] = vlist 
    df['recomb current'] = J_simu
    dfs.append(df) 

dfs_in = get_clean_data_dfs(dfs)

ft.individual_no0V(dfs_in)


#%% ind_0
# R_ion = 6e4
# C_g = 3e-7
# J_s = 1e-7
# nA = 1.4
# V_bi = 1.1
# R_s = 10
# R_shnt = 1e7

# C_A = 2e-6
# C_ion = 2e-5
# J_nA = 4e-8


Vb = [0] # List of bias voltage
# Vb = [0.1, 0.3, 0.5]# This s
num_of_data = 40
wlist = np.logspace(-5, 4, num_of_data)
wlist = wlist[::-1]
dfs = []
mu, sigma = 0, .03

#adding the noise



for i in Vb:
    vlist = np.ones(len(wlist)) * i
    wvlist = np.stack((wlist, vlist) , axis = -1)
    Z_simu, J_simu = pmf.pero_model_glob(wvlist, C_A_0, C_ion_0, R_ion, C_g, J_s, nA, V_bi, R_s, R_shnt)
    #Z_simu1, J_simu = pmf.pero_model_ind_no0V(wlist, C_A, C_ion, R_ion, C_g, J_s, nA, R_s, R_shnt,0.9)
    # Z_simu = pmf.pero_model_ind_0V(wlist, C_ion, C_g, R_ion, J_nA, R_s, R_shnt)
    df = pd.DataFrame()
    df['frequency'] = wlist
    # adding noise to the Z
    noise_real =  np.random.normal(mu, sigma, num_of_data)
    noise_imag =  np.random.normal(mu, sigma, num_of_data)
    
    df['Z_real'] = np.real(Z_simu)*(1 + noise_real)
    df['Z_imag'] =  np.imag(Z_simu)*(1 + noise_imag)
    df['bias voltage'] = vlist 
    df['recomb current'] = J_s *np.ones(len(wlist))
   # df['recomb current'] = J_simu
    dfs.append(df) 




dfs_i0 = get_clean_data_dfs(dfs)

ft.individual_0V(dfs_i0)


#%% global 0


# Setting some initial parameters
Vb = [0, 0.3, 0.6, 0.9] # List of bias voltage
# Vb = [0.1, 0.3, 0.5]# This s
num_of_data = 40
wlist = np.logspace(-3, 5, num_of_data)
wlist=wlist[::-1]
dfs = []
mu, sigma = 0, .03

#adding the noise



for i in Vb:
    vlist = np.ones(len(wlist)) * i
    wvlist = np.stack((wlist, vlist) , axis = -1)
    Z_simu, J_simu = pmf.pero_model_glob(wvlist, C_A_0, C_ion_0, R_ion, C_g, J_s, nA, V_bi, R_s, R_shnt)
    df = pd.DataFrame()
    df['frequency'] = wlist
    # adding noise to the Z
    noise_real =  np.random.normal(mu, sigma, num_of_data)
    noise_imag =  np.random.normal(mu, sigma, num_of_data)
    
    df['Z_real'] = np.real(Z_simu)*(1 + noise_real)
    df['Z_imag'] =  np.imag(Z_simu)*(1 + noise_imag)
    df['bias voltage'] = vlist 
    df['recomb current'] = J_simu
    dfs.append(df) 

dfs_gn = get_clean_data_dfs(dfs)

ft.global_0V(dfs_gn)






