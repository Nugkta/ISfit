# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:56:03 2022

@author: pokey

In this file:
-The model of the perovskite equivalent circuit is defined 
-The (global) fit with the function of fixing selected parameters is written
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, Model

#constants used in the model
VT = 0.026 # = kT/q
q = 1.6e-19
T = 300
kb = 1.38e-23

def Zcap(c, w): #returns the impedance of a capacitor
    return 1 / (1j * w * c)

def pero_model(w, C_A, C_B, R_i, C_g, J_s, nA, Vb): #w is the list of frequency, the independent variable
    '''
    The is the model of the perovskite, by inputting C_A, C_B, R_i, C_g, J_s, nA, Vb as parameters, w(frequency) as the independent variable,
    the corresponding impedance Z(dependent variable) can be obtainded, together with the current in electronic branch J1.
    '''
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
    return Z_tot, J1      #returning the total impedance and the current in electronic branch

def pero_sep(w,C_A,C_B,R_ion,C_g,J_s,nA,vb): 
    '''
    This is the function used as the model for the curve fit.
    
    Because the pero_model will return a complex Z, while the curve fitting function does not support complex fitting,
    the function here will separate the real and imaginary part of the Z and put them in an array for the fitting function.
    '''
    z = pero_model(w, C_A,C_B,R_ion,C_g,J_s,nA, vb)[0] # only keep the Z_tot, not J1
    return np.hstack([z.real, z.imag])     # this will return a list of the form [z1real, z2real,...,z1imag, z2imag, ...]

def global_fit(dfs, init_guess, fix_index):
    '''
    This is the function fot the global fit, so the data need to be stack together to form big arrays.
    For individual fit, just input a list of dataframes with only one dataframe as element.
    '''
    
    params_list = ['C_A' , 'C_B', "R_ion", 'C_g', 'J_s' , 'nA'] #the list of parameters names
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
    vb = vlist_big[0]  
    mod = Model(lambda wvb,C_A,C_B,R_ion,C_g,J_s,nA: pero_sep(wvb,C_A,C_B,R_ion,C_g,J_s,nA,vb)) # using the pero_sep function to define a model for the fitting
    pars = mod.make_params(C_A=init_guess[0],C_B=init_guess[1],R_ion=init_guess[2],C_g=init_guess[3],J_s=init_guess[4],nA=init_guess[5]) #define the parameters for the fitting
    for i in fix_index:    #make the user-selected fixed parameters to have a very narrow fitting range, virtually fixed.
        pars[params_list[i]].max = init_guess[i] *1.001
        pars[params_list[i]].min = init_guess[i] *0.999
    pars.pretty_print()
    result = mod.fit(Zlist_big,pars, wvb =wvlist_big[:,0] )
    return result #This result is a class defined from the package lmfit, containing informations such as the fitted value, uncertainties, initial guess ...




















































































































