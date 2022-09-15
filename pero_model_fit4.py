# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:56:03 2022

@author: pokey

In this file:
-The model of the perovskite equivalent circuit is defined 
-The (global) fit with the function of fixing selected parameters is written
"""

import numpy as np
from lmfit import Model

#constants used in the model
VT = 0.026 # = kT/q
q = 1.6e-19
T = 300
kb = 1.38e-23

def Zcap(c, w): #returns the impedance of a capacitor
    return 1 / (1j * w * c)

def pero_model(w_Vb, C_A_0, C_ion_0, R_i, C_g, J_s, nA, V_bi_A, V_bi_ion, R_srs, R_shnt): #w is the list of frequency, the independent variable
    '''
    The is the model of the perovskite, by inputting C_A, C_B, R_i, C_g, J_s, nA, Vb as parameters, w(frequency) as the independent variable,
    the corresponding impedance Z(dependent variable) can be obtainded, together with the current in electronic branch J1.
    '''
    w = w_Vb[:,0]
    Vb = w_Vb[:,1]
    C_ion = C_ion_0 * np.sqrt(V_bi_ion/(V_bi_ion - Vb))
    C_A = C_A_0 * np.sqrt(V_bi_A/(V_bi_A - Vb))
    C_B = 1 / (1/C_ion - 1/C_A)
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_i)
    Z_ion = (Zcap(C_A , w) + Zcap(C_B , w) + Z_d) #the impedance of the ionic branch
    v1 = Vb * (C_A / (C_A + C_B))
    J1 = J_s*(np.e**((v1 - 0) / (nA * VT)) - np.e**((v1 - Vb) / (nA * VT))) #the current density of the electronic branch
    Jrec = J_s * np.e**(v1 / (nA * VT))        #the recombination current
    Jgen = J_s * np.e**((v1 - Vb) / (nA * VT)) #the generation current
    A = Zcap(C_A , w)/ Z_ion
    djdv = (1 - A) * Jrec / (nA * VT) + A * Jgen / (nA * VT)
    Z_elct = 1 / djdv #the impedance of the electronic branch
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct) #the total impedance
    Z_tot2 = Z_tot + R_srs
    Z_tot3 = 1 / (1/Z_tot + 1/R_shnt) # the Z_tot3 with the R_srs and the R_shnt 
    
    return Z_tot3, J1      #returning the total impedance and the current in electronic branch


def pero_model_ind(w, C_A, C_ion, R_i, C_g, J_s, nA, R_srs, R_shnt,Vb,): # for individual set
    '''
    The is the model of the perovskite, by inputting C_A, C_B, R_i, C_g, J_s, nA, Vb as parameters, w(frequency) as the independent variable,
    the corresponding impedance Z(dependent variable) can be obtainded, together with the current in electronic branch J1.
    '''
    C_B = 1 / (1/C_ion - 1/C_A)
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_i)
    Z_ion = (Zcap(C_A , w) + Zcap(C_B , w) + Z_d) #the impedance of the ionic branch
    v1 = Vb * (C_A / (C_A + C_B))
    J1 = J_s*(np.e**((v1 - 0) / (nA * VT)) - np.e**((v1 - Vb) / (nA * VT))) #the current density of the electronic branch
    Jrec = J_s * np.e**(v1 / (nA * VT))        #the recombination current
    Jgen = J_s * np.e**((v1 - Vb) / (nA * VT)) #the generation current
    A = Zcap(C_A , w)/ Z_ion
    djdv = (1 - A) * Jrec / (nA * VT) + A * Jgen / (nA * VT)
    Z_elct = 1 / djdv #the impedance of the electronic branch
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct) #the total impedance
    Z_tot2 = Z_tot + R_srs
    Z_tot3 = 1 / (1/Z_tot + 1/R_shnt) # the Z_tot3 with the R_srs and the R_shnt 
    return Z_tot3, J1      #returning the total impedance and the current in electronic branch


def pero_model_0V(w, C_ion, C_g, R_ion, J_nA, R_srs, R_shnt):
    '''
    This is the model for only the 0V individual fit.
    in this case the value of C_A is not distinguishable
    
    '''
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_ion)
    Z_ion = Zcap(C_ion,w) + Z_d
    djdv = J_nA/VT
    Z_elct = 1 / djdv #the impedance of the electronic branch
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct) #the total impedance
    Z_tot2 = Z_tot + R_srs
    Z_tot3 = 1 / (1/Z_tot + 1/R_shnt) # the Z_tot3 with the R_srs and the R_shnt 
    return Z_tot3 


def pero_sep(w_Vb,C_A_0, C_ion_0, R_ion, C_g, J_s, nA, V_bi_A, V_bi_ion, R_srs, R_shnt): 
    '''
    This is the function used as the model for the curve fit.
    
    Because the pero_model will return a complex Z, while the curve fitting function does not support complex fitting,
    the function here will separate the real and imaginary part of the Z and put them in an array for the fitting function.
    '''
    z = pero_model(w_Vb,C_A_0, C_ion_0, R_ion, C_g, J_s, nA, V_bi_A, V_bi_ion, R_srs, R_shnt)[0] # only keep the Z_tot, not J1
    return np.hstack([z.real, z.imag])     # this will return a list of the form [z1real, z2real,...,z1imag, z2imag, ...]


def pero_sep_0V(w, C_ion, C_g, R_ion, J_nA, R_srs, R_shnt):
    z = pero_model_0V(w, C_ion, C_g, R_ion, J_nA, R_srs, R_shnt)
    return np.hstack([z.real, z.imag])  


    


def global_fit(dfs, init_guess, fix_index=[], mode = 0):
    '''
    This is the function fot the global fit, so the data need to be stack together to form big arrays.
    For individual fit, just input a list of dataframes with only one dataframe as element.
    There are three modes for the global_fit
    0. without 0V global and individual
    1. without 0V individual
    2. without 0V global
    '''

    zlist_big = np.array([])      #because we are fitting the w,v to z, need to stack up all the w v and z list from different Vb each to be one big list. 
    wlist_big = np.array([])
    vlist_big = np.array([])
    for df in dfs:
        print(type(df))
        zlist_big = np.concatenate((zlist_big , df['impedance'].values))
        wlist_big = np.concatenate((wlist_big , df['frequency'].values.real))
        vlist_big = np.concatenate((vlist_big , df['bias voltage'].values.real))
    wvlist_big = np.stack((wlist_big,vlist_big),axis = 1)          # the v is changing for as the condition change, so in order to do the global fit, I regard is as a variable here, and stacked it with w.
    zrlist_big = zlist_big.real 
    zilist_big = zlist_big.imag 
    Zlist_big = np.hstack([zrlist_big, zilist_big]) 
    
    
    if mode == 0:#this mod is for without 0V global and individual
        params_list = ['C_A_0', 'C_ion_0', 'R_i',' C_g',' J_s', 'nA',' V_bi_A', 'V_bi_ion','R_srs', 'R_shnt'] #the list of parameters names
        #mod = Model(lambda wvb,C_A,C_ion,R_ion,C_g,J_s,nA: pero_sep(wvb,C_A,C_ion,R_ion,C_g,J_s,nA,vb)) # using the pero_sep function to define a model for the fitting
        mod = Model(pero_sep)
        pars = mod.make_params(C_A_0=init_guess.C_A,C_ion_0=init_guess.C_ion,R_ion=init_guess.R_ion,C_g=init_guess.C_g,
                               J_s=init_guess.J_s,nA=init_guess.nA,V_bi_A = 1, V_bi_ion = 1,R_srs = init_guess.R_srs , R_shnt = init_guess.R_shnt) #define the parameters for the fitting
    if mode == 1: #this mod is for 0 V individually
        params_list = ['C_ion', ' C_g','R_i',' J_nA','R_srs', 'R_shnt']
        mod = Model(pero_sep_0V)
        pars = mod.make_params(C_ion=init_guess.C_ion,R_ion=init_guess.R_ion,C_g=init_guess.C_g,
                               J_nA=init_guess.J_nA,V_bi_A = 1,R_srs = init_guess.R_srs , R_shnt = init_guess.R_shnt)
    
    
    print(mod.param_names, mod.independent_vars)
    #print(init_guess[2])
    for i in fix_index:    #make the user-selected fixed parameters to have a very narrow fitting range, virtually fixed.
        pars[params_list[i]].max = init_guess[i] *1.001
        pars[params_list[i]].min = init_guess[i] *0.999
    if mode == 0:
        pars['V_bi_A'].min = 0.9
        pars[ 'V_bi_ion'].min = 0.9
        pars['V_bi_A'].max = 1.5
        pars[ 'V_bi_ion'].max = 1.5
        

    
    if mode == 0:
        result = mod.fit(Zlist_big,pars, w_Vb =wvlist_big , weights = 1 / Zlist_big)
        return result
    elif mode == 1: 
        result = mod.fit(Zlist_big,pars, w =wlist_big) 
                         #,weights = Zlist_big)
        return result #This result is a class defined from the package lmfit, containing informations such as the fitted value, uncertainties, initial guess ...
   
    pars.pretty_print()



















































































































