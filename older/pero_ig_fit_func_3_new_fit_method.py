# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 00:05:02 2022

@author: pokey

In this file, I use the lm minimizer to try do the curve fit




"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import pandas as pd
import pandas as pd
from scipy.signal import argrelextrema
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit






def Zcap(c, w): #returns the impedance of a capacitor
    return 1 / (1j * w * c)

VT = 0.026 # = kT/q
C_a = 2e-7
C_b = 2e-7
R_i = 5e8
Vb = .2
C_g = 2.8e-8  
J_s = 7.1e-11
nA = 1.93
q = 1.6e-19
T = 300
kb = 1.38e-23


#For model of the perovskite
# def pero_model(w, C_a, C_b, R_i, C_g, J_s, n, Vb): #w is the list of frequency range
#     Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_i)
#     Z_ion = (Zcap(C_a , w) + Zcap(C_b , w) + Z_d) #the impedance of the ionic branch
#     v1 = Vb * (C_a / (C_a + C_b))
#     J1 = J_s*(np.e**((v1 - 0) / (n * VT)) - np.e**((v1 - Vb) / (n * VT))) #the current densitf of the electronic branch
#     #print('the power in J1 is', (v1 - Vb) / (n * VT))
#     #J1 = J_s*(np.exp((v1 - 0) / (n * VT)) - 0)
#     Jrec = J_s * np.e**(v1 / (n * VT))        #the recombination current and the generation current
#     Jgen = J_s * np.e**((v1 - Vb) / (n * VT))
#     A = Zcap(C_a , w)/ Z_ion
#     djdv = (1 - A) * Jrec / (n * VT) + A * Jgen / (n * VT)
#     Z_elct = 1 / djdv #the impedance of the electronic branch
#     Z_tot = 1 / (1/Z_ion + 1/ Z_elct)
#     return Z_tot, J1





def pero_model(w, params, Vb): #w is the list of frequency range
    C_A = params['C_A']
    C_B  = params['C_B']
    R_ion  = params['R_ion']
    C_g  = params['C_g']
    J_s = params['J_s']
    nA  = params['nA']
    
    
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_i)
    Z_ion = (Zcap(C_A , w) + Zcap(C_b , w) + Z_d) #the impedance of the ionic branch
    v1 = Vb * (C_A / (C_A + C_b))
    J1 = J_s*(np.e**((v1 - 0) / (nA * VT)) - np.e**((v1 - Vb) / (nA * VT))) #the current densitf of the electronic branch
    #print('the power in J1 is', (v1 - Vb) / (n * VT))
    #J1 = J_s*(np.exp((v1 - 0) / (n * VT)) - 0)
    Jrec = J_s * np.e**(v1 / (nA * VT))        #the recombination current and the generation current
    Jgen = J_s * np.e**((v1 - Vb) / (nA * VT))
    A = Zcap(C_A , w)/ Z_ion
    djdv = (1 - A) * Jrec / (nA * VT) + A * Jgen / (nA * VT)
    Z_elct = 1 / djdv #the impedance of the electronic branch
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct)
    return Z_tot, J1




def pero_sep(params,wvb,data,vb):
    z = pero_model(wvb, params, vb)[0] # because w and vb are both regarded as variables here.
    
    return np.hstack([z.real, z.imag]) - data         # this will return a list of the form [z1real, z2real,...,z1imag, z2imag, ...]


#the function for global fitting, need to input the list of df and the initial guess obtained above. 
def global_fit(dfs, init_guess, fix_index):
    #define parameters:
    params = Parameters()
    # params.add('C_A',   value= init_guess[0],min = 0, max = 1e-2)
    # params.add('C_B', value= init_guess[1],min = 0, max = 1e-2)
    # params.add('R_ion', value= init_guess[2],min = 1e3, max = 1e8)
    # params.add('C_g', value= init_guess[3],min = 0, max = 1e-2)
    # params.add('J_s', value= init_guess[4],min = 0, max = 1e-2)
    # params.add('nA', value= init_guess[5],min =0.5,max = 2)
    params.add('C_A',   value= init_guess[0])
    params.add('C_B', value= init_guess[1])
    params.add('R_ion', value= init_guess[2])
    params.add('C_g', value= init_guess[3])
    params.add('J_s', value= init_guess[4])
    params.add('nA', value= init_guess[5])
    
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
    v = vlist_big[0]


    minner = Minimizer(pero_sep , params , fcn_args = (wvlist_big[:,0],Zlist_big,v))
    #minner = Minimizer(pero_sep , params , fcn_args = (a,b))
    result = minner.minimize(method = 'leastsq')

    

    
    
    
    return result







#%%  TEST
result = global_fit([df] , init_guess.values() , fix_index)
report_fit(result)
result_dict = result.params.valuesdict()
popt = []
for key in result_dict:
    popt.append( result_dict[key])


#%%
import pero_ig_fit_func_3_old_init_model as pif
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import pandas as pd
from scipy.signal import argrelextrema
import glob 
import init_guess3 as ig3
from matplotlib.widgets import TextBox,Slider, Button,CheckButtons


z , j = pif.pero_model(wlist,*popt,v)
z_ig, j_ig = pif.pero_model(wlist,*init_guess.values(),v)

fig, ((ax1 ,ax2),(ax3,ax4)) = plt.subplots(figsize=(18, 12),ncols = 2 , nrows = 2)
fig.suptitle('Comparison between the initial guess and the fitted parameters', fontsize = 16)

#The Nyquist plot
ax_nyq = plt.subplot(212)
line1, = ax_nyq.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'x', ms=4 , label = 'actual data')   #actual data line
line2, = ax_nyq.plot(np.real(z),-np.imag(z),'m-', label = 'fitted') #fitted parameter line
line3, =ax_nyq.plot(np.real(z_ig),-np.imag(z_ig),'b--', label = 'initial guess') #initial guess line
ax_nyq.legend()
ax_nyq.set_xlabel('Z\'')
ax_nyq.set_ylabel('Z\'\'')

#Z_real plot
line_zr, = ax1.plot(df['frequency'],np.real(df['impedance'].values),'bx',ms = 4, label = 'actual data Z\' ')
line_zr_ig, = ax1.plot(wlist,np.real(z_ig),linestyle = 'dotted',color = 'c', label = ' initial guess Z\'')
line_zr_fit, = ax1.plot(wlist,np.real(z),linestyle = 'solid',color = 'm', label = ' fitted Z\'')
ax1.set_xscale('log')
ax1.set_ylabel('Z\'')
ax1.set_xlabel(r'frequency $\omega$')
ax1.set_title('Real Z, effective capacitance vs. frequency')
ax1.legend(loc = 3, fontsize = 'small')
ax1.spines['left'].set_color('c')
ax1.tick_params(axis='y', colors='c')

#effective ccapacitance

ax_eff = ax1.twinx()
C_eff = np.imag(1 / df['impedance']) / df['frequency']
C_eff_ig = np.imag(1 / z_ig) / wlist
C_eff_fit = np.imag(1 / z) / wlist


line_Ceff, = ax_eff.plot(df['frequency'],C_eff,'x',ms = 4,color = 'peru', label = 'experiment effective capacitance')
line_Ceff_ig, = ax_eff.plot(wlist , C_eff_ig,linestyle = 'dotted',ms = 4,color = 'orange', label = 'initial guess effective capacitance')
line_Ceff_fit, = ax_eff.plot(wlist , C_eff_fit,linestyle = 'solid',ms = 4,color = 'y', label = 'fitted effective capacitance')
ax_eff.set_yscale('log')
ax_eff.set_xscale('log')
ax_eff.set_ylabel(r'Im($Z^{-1}$)$\omega^{-1}$')
ax_eff.legend(loc = 1, fontsize = 'small')
ax_eff.spines['right'].set_color('orange')
ax_eff.tick_params(axis='y', colors='orange')


#abs Z part plot
line_absz, = ax2.plot(df['frequency'],np.abs(df['impedance'].values),'bx',ms = 4, label = 'experiment |Z| ')
line_absz_ig, = ax2.plot(wlist,np.abs(z_ig),linestyle = 'dotted',color = 'c', label = ' initial guess |Z|')
line_absz_ig, = ax2.plot(wlist,np.abs(z),linestyle = 'solid',color = 'm', label = ' fitted |Z|')
ax2.set_xscale('log')
ax2.set_ylabel('|Z|')
ax2.set_xlabel(r'frequency $\omega$')
ax2.set_title(r'|Z|, $\theta$ vs. frequency')
ax2.legend(loc = 3, fontsize = 'small')
ax2.spines['left'].set_color('c')
ax2.tick_params(axis='y', colors='c')

#theta plot

ax_t = ax2.twinx()
line_t, = ax_t.plot(df['frequency'],np.angle(df['impedance'].values),'x',ms = 4,color = 'peru', label = r'experiment $\theta$')
line_t_ig, = ax_t.plot(wlist , np.angle(z_ig),linestyle = 'dotted',ms = 4,color = 'orange', label = r'initial guess $\theta$')
line_t_ig, = ax_t.plot(wlist , np.angle(z),linestyle = 'solid',ms = 4,color = 'y', label = r'fitted $\theta$')
ax_t.set_xscale('log')
ax_t.set_ylabel(r'$\theta$')
ax_t.legend(loc = 1, fontsize = 'small')
ax_t.spines['right'].set_color('orange')
ax_t.tick_params(axis='y', colors='orange')
print('the fitted parameters are: \n C_A is %.2e, \n C_B is %.2e, \n R_ion is %.2e, \n C_g is %.2e, \n J_s is %.2e, \n nA is %.2e.' %(popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]))

















