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

dfs=dfs[1:4]
df = dfs[a]
v = v[a]








# crit_points = igp.find_point(df)
# ig = igp.init_guess_find(df,crit_points) 
# init_guess = igp.init_guess_class()
# init_guess.update_all(ig)

#%%
# def __main__(dfs,v):
#     df = dfs[-1]#only uses the last plot to find the initial guess (becasue it has the stable shape)
#     crit_points = find_point(dfs[-1]) 
#     ig = init_guess_find(df,crit_points) 
#     init_guess = init_guess_class()
#     init_guess.update_all(ig)
#     R_ion_Slider(init_guess, df, v,crit_points)

df = dfs[-1]
crit_points = igp.find_point(dfs[-1]) 
ig = igp.init_guess_find(df,crit_points)
init_guess = igp.init_guess_class()
init_guess.update_all(ig)
#%%
#igp.fit_plot_comp_plots(dfs,init_guess)


#%%
# zlist_big = np.array([])      #because we are fitting the w,v to z, need to stack up all the w v and z list from different Vb each to be one big list. 
# wlist_big = np.array([])
# vlist_big = np.array([])
# for df in dfs:
#     zlist_big = np.concatenate((zlist_big , df['impedance'].values))
#     wlist_big = np.concatenate((wlist_big , df['frequency'].values.real))
#     vlist_big = np.concatenate((vlist_big , df['bias voltage'].values.real))
# wvlist_big = np.stack((wlist_big,vlist_big),axis = 1)
# z = pmf.pero_sep(wvlist_big,*init_guess.values(), 1, 1)

# plt.plot(z[0:144],-z[144:288],'.')


igp.__main__(dfs)























