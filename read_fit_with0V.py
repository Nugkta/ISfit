# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:09:06 2022

@author: pokey

In this file, I will try to read and fit the data stored from the paper Fig 2a by using the functions 
written previously. 
Using older inital_guess algorithm

"""

import pero_ig_fit_func_3_old_init_model as pif
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import pandas as pd
from scipy.signal import argrelextrema
import glob 
 

#%%

dfs = []
for file in glob.glob('paperdata/**.xlsx'):
    plt.figure()
    df = pd.read_excel(file)
    df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]
    df['z_imag'] = -df['z_imag'].values
    dfs.append(df)
    plt.plot(df['z_real'],-df['z_imag'],'.')



# no longer effective # (because the experimental data do not contain a 0V data set, I will generate the data first in order to test the automaticality of the function. using the fitted parameters obtained by the first three data sets)

# w = dfs[0]['frequency'].values
# zlist, J1 = pif.pero_model(w,2.03534548e-04, 8.53497697e-05, 2.59998353e+04, 4.04292147e-07, 3.36778932e-12, 1.39011012e+00, 0)
# df0 = pd.DataFrame(columns = ['frequency','z_real','z_imag','bias voltage' , 'recomb current'])
# df0['frequency'] = w
# df0['z_real'] = zlist.real
# df0['z_imag'] = zlist.imag #Note there is a minus sign here to keep it consistent with the experiment data.
# df0['bias voltage'] = np.zeros(len(w))
# df0['recomb current'] = np.ones(len(w)) * 6.1e-13
# dfs.append(df0)


#change the extracted dataframe to be the format used in the previous study (minus z_imag and complex impedance)
for df in dfs:
    df['impedance'] = df['z_real'].values + df['z_imag'].values * 1j
    #plt.plot(df['z_real'],-df['z_imag'],'.')

dfs.sort(key = lambda x: x['bias voltage'][0])  # making the dfs list sorted by the magnitude of the bias voltaege of each data set.
dfs









#%%
init_guess = pif.get_init_guess(dfs)


#%% testing global fit

popt, pcov = pif.global_fit(dfs , init_guess)


v = [0,.795,.864,.894]
wlist = np.logspace(-6,6,1000)

for i in v:
    z , j = pif.pero_model(wlist,*popt,i)
    plt.plot(np.real(z),-np.imag(z),'--')

for i in range(0,4):
    df = dfs[i]
    plt.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'.')

plt.xlim([-300,3000])
plt.ylim([-200,1000])


#%% testing individual fit
#AND adding the slider function for changing the initial guess
from matplotlib.widgets import Slider, Button
wlist = np.logspace(-6,6,1000)

a = 3 #change this to change the set of data to fit
v = [0,.795,.864,.894]

popt, pcov = pif.global_fit([dfs[a]] , init_guess)
# popt, pcov = pif.global_fit([dfs[1]] , init_guess)
# popt, pcov = pif.global_fit([dfs[1]] , init_guess)
# popt, pcov = pif.global_fit([dfs[1]] , init_guess)
df = dfs[a]
v1 = v[a]
# v = [.795,]
# v = [.864,]
# v = [.894,]
z , j = pif.pero_model(wlist,*popt,v1)
fig, axs = plt.subplots(figsize=(5, 5),ncols = 1 , nrows = 1)
ax = axs
line1 = plt.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'x', ms=4)
line2, = plt.plot(np.real(z),-np.imag(z),'r--')
ax.set_xlabel('Z\'')
ax.set_ylabel('Z\'\'')
plt.subplots_adjust(left=0.25, bottom=.5)
#change only the C_a in popt now as a test
# axC_a = plt.axes([0.25, 0.5, 0.65, 0.03])
ax_list = {} 
sliders = {}
param_name = ['C_a', 'C_b', 'R_i', 'C_g', 'J_s', 'nA' ]




for i in range(0,6):
    ax_list[i] = plt.axes([0.25, 0.05 * (i+2)-0.02, 0.55, 0.03])
    sliders[i] = Slider(
        ax = ax_list[i], 
        label = 'the value of ' + param_name[i],
        valmin = 1/3 * popt[i],
        valmax = 3 * popt[i],
        valinit = popt[i],
        )

sl_val_list =[]
for key in sliders:
    sl_val_list.append(sliders[key])

def update(val,  ):
    # popt = np.delete(popt , i)
    # popt = np.insert(popt,i,sliders[i].val)
    vals = [i.val for i in sl_val_list]
    z , j = pif.pero_model(wlist,*vals,v1)
    #z , j = pif.pero_model(wlist,sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val,sliders[4].val,sliders[5].val,v1)
    #z , j = pif.pero_model(wlist,*popt,v1)
    line2.set_ydata(-np.imag(z))
    fig.canvas.draw_idle()

for key in sliders:
    #print(type(sliders[key]))
    sliders[key].on_changed(lambda val: update(val))
    
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')
def reset(event):
    for key in sliders:
        sliders[key].reset()
button.on_clicked(reset)
# sliders[0].on_changed(lambda val: update(val,))
# sliders[1].on_changed(lambda val: update(val,))
# sliders[2].on_changed(lambda val: update(val,))
# sliders[3].on_changed(lambda val: update(val,))
# sliders[4].on_changed(lambda val: update(val,))
# sliders[5].on_changed(lambda val: update(val,))
#%%some other plots to examine the effectiveness of the fit

z , j = pif.pero_model(df['frequency'].values,*popt,v1)
plt.plot(df['frequency'].values,np.abs(z),'--')


df = dfs[a]
plt.plot(np.real(df['frequency'].values), np.abs(df['impedance']),'.')
plt.title('freq vs. abs(z)')
plt.show()
#####################################################
#%%
z , j = pif.pero_model(df['frequency'].values,*popt,v1)
plt.plot(df['frequency'].values,np.angle(z),'r--')


df = dfs[a]
plt.plot(np.real(df['frequency'].values), np.angle(df['impedance']),'g.')
plt.title(r'freq vs. $\theta$(z) ')

# plt.xlim([-300,3000])
# plt.ylim([-200,1000])






































#%%
r = pif.find_Ri(dfs,3.6e-6)








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













































