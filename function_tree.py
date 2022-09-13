# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:45:24 2022

@author: pokey 

In this file, the functions to be used for different scenerios are written.(With/without 0V, individual/global fitting.) 


"""
import pero_model_fit4 as pmf
import init_guess_plot4 as igp 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import glob 
from matplotlib.widgets import TextBox,Slider, Button,CheckButtons
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit


def individual_no0(df):
    '''
    This is for the case of individual fit and V is not 0.
    this function is the same as the original only version of fitting scenerio, so
    here I directly used the previously written function
    '''
    dfs = [df] #making the individual dataframe a list to use the global_fit function
    igp.__main__(dfs) 
    


def global_no0(dfs):
    '''
    This is for the case of global fit and no V = 0 data
    Could also be adapted directly from the previous main function
    '''
    igp.__main__(dfs)
    
    


def individual_0(df):
    return 2
 


def global_0(dfs):
    
    
    
#%% test

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

dfs=dfs[1:4]
df = dfs[a]
v = v[a]

#%%
individual_no0(df)
#%%
global_no0(dfs)



































































