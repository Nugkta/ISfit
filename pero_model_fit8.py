# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 01:09:17 2022

@author: Stan

In this file, the model of the equivalent circuit and the function for the global fit is written. 

"""

import numpy as np
from lmfit import Model

#constants used in the model
VT = 0.026 # = kT/q
q = 1.6e-19 #elementary charge
T = 300 #room temperture
kb = 1.38e-23 #boltzmann constant

def Zcap(c, w): #returns the impedance of a capacitor with a input of capacitance and frequency
    return 1 / (1j * w * c)

def pero_model_glob(w_Vb, C_A_0, C_ion_0, R_ion, C_g, J_s, nA, V_bi, R_s, R_shnt): #w is the list of frequency, the independent variable
    '''
    The is the model of the perovskite.
    Used for the golbal fit for both with/without 0V case.
    The meaning of the variable are explained the document.
    The corresponding impedance Z(dependent variable) can be obtained, together with the current in electronic branch J1.
    '''
    w = w_Vb[:,0]
    Vb = w_Vb[:,1]
    C_ion = C_ion_0 * np.sqrt(V_bi/(V_bi - Vb))
    C_A = C_A_0 * np.sqrt(V_bi/(V_bi - Vb))
    C_B = 1 / (1/C_ion - 1/C_A)
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_ion)
    Z_ion = (Zcap(C_A , w) + Zcap(C_B , w) + Z_d) #the impedance of the ionic branch
    v1 = Vb * (C_A / (C_A + C_B))
    J1 = J_s*(np.e**((v1 - 0) / (nA * VT)) - np.e**((v1 - Vb) / (nA * VT))) #the current density of the electronic branch
    Jrec = J_s * np.e**(v1 / (nA * VT))        #the recombination current
    Jgen = J_s * np.e**((v1 - Vb) / (nA * VT)) #the generation current
    A = Zcap(C_A , w)/ Z_ion
    djdv = (1 - A) * Jrec / (nA * VT) + A * Jgen / (nA * VT)
    Z_elct = 1 / djdv #the impedance of the electronic branch
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct) #the total impedance
    Z_tot2 = 1 / (1/Z_tot + 1/R_shnt) # the Z_tot3 with the R_s and the R_shnt 
    Z_tot3 = Z_tot2 + R_s
    return Z_tot3, J1      #returning the total impedance and the current in electronic branch

def pero_model_ind_no0V(w, C_A, C_ion, R_ion, C_g, J_s, nA, R_s, R_shnt,Vb): # for individual set
    '''
    The is the model of the perovskite.
    Used for individual fit of non-0 bias voltage.
    The meaning of the variable are explained the document.
    The corresponding impedance Z(dependent variable) can be obtained, together with the current in electronic branch J1.
    '''
    C_B = 1 / (1/C_ion - 1/C_A)
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_ion)
    Z_ion = (Zcap(C_A , w) + Zcap(C_B , w) + Z_d) #the impedance of the ionic branch
    v1 = Vb * (C_A / (C_A + C_B))
    J1 = J_s*(np.e**((v1 - 0) / (nA * VT)) - np.e**((v1 - Vb) / (nA * VT))) #the current density of the electronic branch
    Jrec = J_s * np.e**(v1 / (nA * VT))        #the recombination current
    Jgen = J_s * np.e**((v1 - Vb) / (nA * VT)) #the generation current
    A = Zcap(C_A , w)/ Z_ion
    djdv = (1 - A) * Jrec / (nA * VT) + A * Jgen / (nA * VT)
    Z_elct = 1 / djdv #the impedance of the electronic branch
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct) #the total impedance
    Z_tot2 = 1 / (1/Z_tot + 1/R_shnt) # the Z_tot3 with the R_s and the R_shnt 
    Z_tot3 = Z_tot2 + R_s
    return Z_tot3, J1      #returning the total impedance and the current in electronic branch

def pero_model_ind_0V(w, C_ion, C_g, R_ion, J_nA, R_s, R_shnt):
    '''
    The is the model of the perovskite.
    Used for individual fit of 0 bias voltage.
    '''
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_ion)
    Z_ion = Zcap(C_ion,w) + Z_d
    djdv = J_nA/VT
    Z_elct = 1 / djdv #the impedance of the electronic branch
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct) #the total impedance
    Z_tot2 = 1 / (1/Z_tot + 1/R_shnt) # the Z_tot3 with the R_s and the R_shnt 
    Z_tot3 = Z_tot2 + R_s
    return Z_tot3 


 
#%% The functions below are used to separate the complex Z to real and imaginary part to use for the fitting package

def pero_sep_glob(w_Vb,C_A_0, C_ion_0, R_ion, C_g, J_s, nA,  V_bi, R_s, R_shnt): 
    z = pero_model_glob(w_Vb,C_A_0, C_ion_0, R_ion, C_g, J_s, nA, V_bi, R_s, R_shnt)[0] # only keep the Z_tot, not J1
    return np.hstack([z.real, z.imag])     # this will return a list of the form [z1real, z2real,...,z1imag, z2imag, ...]


def pero_sep_ind_no0V(w, C_A, C_ion, R_ion, C_g, J_s, nA, R_s, R_shnt,Vb):
    z,j = pero_model_ind_no0V(w, C_A, C_ion, R_ion, C_g, J_s, nA, R_s, R_shnt,Vb)
    return np.hstack([z.real, z.imag])


def pero_sep_ind_0V(w, C_ion, C_g, R_ion, J_nA, R_s, R_shnt):
    z = pero_model_ind_0V(w, C_ion, C_g, R_ion, J_nA, R_s, R_shnt)
    return np.hstack([z.real, z.imag])  



#%% The function below is used for the global fitting for different scenerios.
def global_fit(dfs, init_guess, fix_index=[], mode = 'glob_0V', V_bi_guess = 1):
    '''
    This is the function fot the global fit, so the data need to be stack together to form big arrays.
    For individual fit, just input a list of dataframes with only one dataframe as element.
    There are three modes for the global_fit
    
    0. without 0V global and individual
    1. without 0V individual
    2. without 0V global
    '''
    #The following lines are for building up global arrays that stores different groups of the data for the later global fit.
    zlist_big = np.array([])      #because we are fitting the w,v to z, need to stack up all the w v and z list from different Vb each to be one big list. 
    wlist_big = np.array([])
    vlist_big = np.array([])
    for df in dfs:
        zlist_big = np.concatenate((zlist_big , df['impedance'].values))
        wlist_big = np.concatenate((wlist_big , df['frequency'].values.real))
        vlist_big = np.concatenate((vlist_big , df['bias voltage'].values.real))
    wvlist_big = np.stack((wlist_big,vlist_big),axis = 1)          # the v is changing for as the condition change, so in order to do the global fit, I regard is as a variable here, and stacked it with w.
    zrlist_big = zlist_big.real 
    zilist_big = zlist_big.imag 
    Zlist_big = np.hstack([zrlist_big, zilist_big]) 
   
    
    #print(V_bi_guess,1111111111111111111111)
    #The following lines are for establishing the models for the fitting
    #Different parameters and models corresponds to different scenerios.

    if mode == 'glob_0V' or mode == 'glob_no0V': #this mode is for global fit
        params_list = ['C_A_0', 'C_ion_0', 'R_ion','C_g','J_s', 'nA', 'V_bi','R_s', 'R_shnt'] #the list of parameters names
        # params_list2 = ['C_A_0', 'C_ion_0', 'R_ion','C_g','J_s', 'nA', 'V_bi','R_s', 'R_shnt']
        mod = Model(pero_sep_glob)
        pars = mod.make_params(C_A_0=init_guess.C_A,C_ion_0=init_guess.C_ion,R_ion=init_guess.R_ion,C_g=init_guess.C_g,
                               J_s=init_guess.J_s,nA=init_guess.nA, V_bi = V_bi_guess,R_s = init_guess.R_s , R_shnt = init_guess.R_shnt) #define the parameters for the fitting
        
    if mode == 'ind_0V': #this mode is for 0 V individually
        params_list = ['C_ion', 'C_g','R_ion','J_nA','R_s', 'R_shnt']
        # params_list2 = ['C_ion', ' C_g','R_ion','J_nA','R_s', 'R_shnt']
        mod = Model(pero_sep_ind_0V)
        pars = mod.make_params(C_ion=init_guess.C_ion,R_ion=init_guess.R_ion,C_g=init_guess.C_g,
                               J_nA=init_guess.J_nA, V_bi= V_bi_guess,R_s = init_guess.R_s , R_shnt = init_guess.R_shnt)
        
    if mode == 'ind_no0V': #this mode is for no 0V individually
        params_list = ['C_A', 'C_ion', 'R_ion','C_g','J_s', 'nA','R_s', 'R_shnt']
        # params_list2 = ['C_A', 'C_ion', 'R_i','C_g','J_s', 'nA','R_s', 'R_shnt']
        mod = Model(lambda w, C_A, C_ion, R_ion, C_g, J_s, nA, R_s, R_shnt:pero_sep_ind_no0V(w, C_A, C_ion, R_ion, C_g, J_s, nA, R_s, R_shnt,vlist_big[0]) )
        pars = mod.make_params(C_A = init_guess.C_A,C_ion=init_guess.C_ion,R_ion=init_guess.R_ion,C_g=init_guess.C_g,
                               J_s=init_guess.J_s,nA=init_guess.nA,R_s = init_guess.R_s , R_shnt = init_guess.R_shnt)

        
    # for i in params_list:
    #     pars[i].min = -.01

    # make the user-selected fixed parameters to have a very narrow fitting range, virtually fixed.
    for i in fix_index:    
        pars[params_list[i]].vary  = False
        
    # setting a boundary for the bias voltage. 
    if mode == 'glob_0V' or mode == 'glob_no0V': 
        pars['V_bi'].min = V_bi_guess - .18
        print(V_bi_guess - .18  ,  'xxxxxxxxxxxxxxxxxxxxxxxxxx')
        pars[ 'V_bi'].max = V_bi_guess + .4
    
    # setting a boundary for the the ideality factor for global fit
    if mode != 'ind_0V':
        pars['nA'].min = 1

    
    # the following lines implement the fit on the data using different models for different cases.
    if mode == 'glob_0V' or mode == 'glob_no0V':
        pars.pretty_print()
        result = mod.fit(Zlist_big,pars, w_Vb = wvlist_big, weights = 1 / Zlist_big)
        return result#This result is a class defined from the package lmfit, containing informations such as the fitted value, uncertainties, initial guess ...
    
    elif mode == 'ind_0V': 
        pars.pretty_print()
        result = mod.fit(Zlist_big,pars, w = wlist_big, weights = 1 ) 
        return result 
    
    elif mode == 'ind_no0V': 
        pars.pretty_print()
        result = mod.fit(Zlist_big,pars, w = wlist_big, weights = 1 / Zlist_big) 
        return result 
  
    
































