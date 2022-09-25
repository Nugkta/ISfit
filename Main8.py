# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:07:31 2022

@author: pokey

The main loop including the tree of functions is written in this file.
As well as the initial data processing for an Excel file.
"""


import numpy as np 
import pandas as pd
import glob 
import function_tree8 as ft






def get_clean_data():
    dfs = []
    fold_name = input('Input data folder name: ')
    for file in glob.glob(fold_name + '/**.xlsx'):  #data must be store in .xlsx files
        df = pd.read_excel(file)
        df = df[['angular frequency','Z_real','Z_imag','applied voltage','J', 'J_ph']]
        df['frequency'] = df['angular frequency'].values
        # calculate the values needed for the fitting
        Vb = df['applied voltage'].values - (df['J'].values[0] * min(df['Z_real'].values)) * np.ones(len(df['applied voltage'].values))
        J_n = df['J'].values + df['J_ph'].values
        
        df['Z_imag'] = -df['Z_imag'].values
        df['recomb current'] = J_n
        df['bias voltage'] = Vb
        #rearrage and only keep the useful columns
        df = df[['frequency','Z_real','Z_imag','bias voltage','recomb current']]
        
        dfs.append(df)
        
    for df in dfs:
        df['impedance'] = df['Z_real'].values + df['Z_imag'].values * 1j
        
    dfs.sort(key = lambda x: x['bias voltage'][0]) # sort the lis tof dataframe by its bias voltage
    
    return dfs







def main(): #the input must be a list of dataframe, even when there is only one dataframe
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


if __name__ == '__main__':
    main()















