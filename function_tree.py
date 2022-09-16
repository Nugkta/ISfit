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
VT = 0.026

def individual_no0V(dfs):
    '''
    This is for the case of individual fit and V is not 0.
    this function is the same as the original only version of fitting scenerio, so
    here I directly used the previously written function
    ''' 
    igp.__main__(dfs,mode = 2) 
    


def global_no0V(dfs):
    '''
    This is for the case of global fit and no V = 0 data
    Could also be adapted directly from the previous main function
    '''
    igp.__main__(dfs)
    
    


def individual_0V(dfs):
    df = dfs[0]
    ig = igp.init_guess_find_0V(df)

    init_guess = igp.init_guess_class()
    init_guess.update_all_0V(ig)
    
    # print('----------',init_guess.J_nA)
    result = pmf.global_fit([df], init_guess, mode = 1)
    report_fit(result)
    result_dict = result.params.valuesdict()
    #putting the resultant parameters into the popt list
    popt = []
    for key in result_dict:
        popt.append( result_dict[key])
    dfs= [df]
    igp.plot_comp(popt,init_guess, dfs, mode = 1 )
 


def global_0V(dfs):
   
    df = dfs[-1]#only uses the last plot to find the initial guess (becasue it has the stable shape)
    crit_points = igp.find_point(dfs[-1]) 
    
    k = crit_points[1] / crit_points[0]
    nA_e , J_s_e = igp.find_nA_Js(dfs, k, mode = 0)
    print('A different method(different from the built-in method in the following steps) gives estimation of nA and J_s to be', nA_e , J_s_e)
    
    # df = dfs[-1]#only uses the last plot to find the initial guess (becasue it has the stable shape)
    ig = igp.init_guess_find(df,crit_points,V0 = True, df_0V = dfs[0]) 
    init_guess = igp.init_guess_class()
    init_guess.update_all(ig)
    igp.R_ion_Slider(init_guess, dfs,crit_points)
    return 


    









# #%% test

# dfs = []
# for file in glob.glob('paperdata/**.xlsx'): 
#     df = pd.read_excel(file)
#     df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]
#     df['z_imag'] = -df['z_imag'].values
#     dfs.append(df)

# for df in dfs:
#     df['impedance'] = df['z_real'].values + df['z_imag'].values * 1j

# dfs.sort(key = lambda x: x['bias voltage'][0])  # making the dfs list sorted by the magnitude of the bias voltaege of each data set.

# wlist = np.logspace(-6,6,1000)
# a = 2 #change this to change the set of data to fit
# v = [0,.795,.846,.894]

# dfs=dfs[0:4]
# df = dfs[a]
# v = v[a]

# #%%
# global_0V(dfs)




# #%%
# df= dfs[0]
# individual_0V(df)


# #%%
# df= dfs[2]
# individual_no0V(df)


# #%%
# dfs1=dfs[1:4]
# global_no0V(dfs1)



    #%%
# wlist = np.logspace(-1,4,100)
# wlist2 = np.logspace(-5,4,100)
# z = pmf.pero_model_0V(wlist, 1.5e-5, 7.2e-7, 7.4e4, 3.1e-8, 107, 644288)
# z2 = pmf.pero_model_0V(wlist2, 1.5e-5, 7.2e-7, 7.4e4, 3.1e-8, 107, 644288)
# plt.plot(z.real,-z.imag,'.',ms = 5)
# plt.plot(z2.real,-z2.imag,'-',linewidth = .2)
# #plt.plot(max(z2.real),0,'rx')






# #%%
# individual_no0(df)
# #%%
# global_no0(dfs)


# #%%

# init = igp.init_guess_find_0V(df)
# #%%
# df= dfs[0]
# z = df['impedance'].values
# plt.plot(np.real(z),-np.imag(z),'.') 


# z_ig = pmf.pero_model_0V(wlist,*init)
# plt.plot(np.real(z_ig),-np.imag(z_ig),'.')

























































