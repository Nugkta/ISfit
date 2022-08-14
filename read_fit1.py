# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:09:06 2022

@author: pokey

In this file, I will try to read and fit the data stored from the paper Fig 2a by using the functions 
written previously

"""

import pero_ig_fit_func  as pif
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import pandas as pd
from scipy.signal import argrelextrema
import glob 




dfs = []
for file in glob.glob('paperdata/**.xlsx'):
    df = pd.read_excel(file)
    df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]
    df['z_imag'] = -df['z_imag'].values
    dfs.append(df)
    #plt.plot(df['z_real'],-df['z_imag'],'.')



# because the experimental data do not contain a 0V data set, I will generate the data first in order to test the automaticality of the function. using the fitted parameters obtained by the first three data sets

w = dfs[0]['frequency'].values
zlist, J1 = pif.pero_model(w,2.03534548e-04, 8.53497697e-05, 2.59998353e+04, 4.04292147e-07, 3.36778932e-12, 1.39011012e+00, 0)
df0 = pd.DataFrame(columns = ['frequency','z_real','z_imag','bias voltage' , 'recomb current'])
df0['frequency'] = w
df0['z_real'] = zlist.real
df0['z_imag'] = zlist.imag #Note there is a minus sign here to keep it consistent with the experiment data.
df0['bias voltage'] = np.zeros(len(w))
df0['recomb current'] = np.ones(len(w)) * 6.1e-13
dfs.append(df0)

for df in dfs:
    df['impedance'] = df['z_real'].values + df['z_imag'].values * 1j
    plt.plot(df['z_real'],-df['z_imag'],'.')

dfs.sort(key = lambda x: x['bias voltage'][0])  # making the dfs list sorted by the magnitude of the bias voltaege of each data set.
dfs

#%% trying to include the simulated 0V data
for i in range(0,4):
    df = dfs[i]
    plt.plot(np.real(df['impedance'].values), np.imag(df['impedance']),'.')

#%% testing
df = pd.read_excel('paperdata/0.01sun.xlsx')
df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]

#%% Doing the global fit now, first using paper values as initial guess




#popt,pcov = pif.global_fit(dfs,[7.2e-2, 7.2e-2, 6.7, 4.4e-4, 6.1e-9, 1.79])
v = [.795,.876,.894]
popt,pcov = pif.global_fit(dfs,[7.2e-6, 3e-6, 6.7e5, 4.4e-7, 6.1e-13, 1.79])
wlist = np.logspace(-2,5,1000)
for i in v:
    z , j = pif.pero_model(wlist,*popt,i)
    plt.plot(np.real(z),-np.imag(z),'-')

df2=[]
for file in glob.glob('paperdata/**.xlsx'):        #plotting the original points as comparison
    df = pd.read_excel(file)
    df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]
    df2.append(df)
    plt.plot(df['z_real'],df['z_imag'],'.')




#%% plotting out the parameter
w = np.logspace(-10,15,1000)
zlist, J1 = pif.pero_model(w, 7.2e-6, 3e-6, 6.7e5, 4.4e-7, 6.1e-12, 1.8,.9)
plt.plot(np.real(zlist), -np.imag(zlist),'.')


#%% After all the testing,put the simulated and actual data into the fitting funtion as a whole

init_guess = pif.get_init_guess(dfs)
init_guess2 = [2.03534548e-04, 8.53497697e-05, 2.59998353e+04, 4.04292147e-07, 3.36778932e-12, 1.39011012e+00]
print('the initial guesses are', init_guess)
popt, pcov = pif.global_fit(dfs , init_guess)
print('the fitted parameters are',popt)

v = [0,.795,.876,.894]
wlist = np.logspace(-2,5,1000)
for i in v:
    z , j = pif.pero_model(wlist,*popt,i)
    plt.plot(np.real(z),-np.imag(z),'--')

for i in range(0,4):
    df = dfs[i]
    plt.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'.')

plt.xlim([-300,3000])
plt.ylim([-200,1000])













































