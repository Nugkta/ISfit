# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:07:31 2022

@author: pokey

The main loop including the tree of functions is written in this file.
"""

import pero_model_fit4 as pmf
import init_guess_plot4 as igp 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import glob 
from matplotlib.widgets import TextBox,Slider, Button,CheckButtons
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import function_tree as ft






def get_clean_data():
    dfs = []
    fold_name = input('Input data folder name: ')
    for file in glob.glob(fold_name + '/**.xlsx'):  #data must be store in .xlsx files
        df = pd.read_excel(file)
        df = df[['frequency','z_real','z_imag','applied voltage','J', 'J_ph']]
        # calculate the values needed for the fitting
        Vb = df['applied voltage'].values - (df['J'].values[0] * min(df['z_real'].values)) * np.ones(len(df['applied voltage'].values))
        J_n = df['J'].values + df['J_ph'].values
        
        df['z_imag'] = -df['z_imag'].values
        df['recomb current'] = J_n
        df['bias voltage'] = Vb
        #rearrage and only keep the useful columns
        df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]
        
        dfs.append(df)
        
    for df in dfs:
        df['impedance'] = df['z_real'].values + df['z_imag'].values * 1j
        
    dfs.sort(key = lambda x: x['bias voltage'][0]) # sort the lis tof dataframe by its bias voltage
    
    return dfs







def __main__(): #the input must be a list of dataframe, even when there is only one dataframe
    dfs = get_clean_data()
    V0 = input('Do the data contain a set with 0 V bias? y/n: ')
    ind_glo = input('More than one bias voltage? y/n: ')
    if V0 == 'y' and ind_glo == 'n':
        ft.individual_0V(dfs)
    elif V0 == 'y' and ind_glo == 'y':
        ft.global_0V(dfs)
    elif V0 == 'n' and ind_glo == 'n':
        ft.individual_no0V(dfs)
    elif V0 == 'n' and ind_glo == 'y':
        ft.global_no0V(dfs)


















