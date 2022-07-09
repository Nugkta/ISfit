# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:48:35 2022

@author: pokey

Finding the spectra of a more complex circuit. Using more segmented equations
to find the total impedance of the circuit

see Goodnotes section 2
"""

#可以添加一个C = 0时的报错

from sre_constants import BRANCH
import sec_circuit2_cleaned as circuit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func_imp(w, c1, c2, r1, r2, func):                   #the funtion used in the curve fit(only to return a stacked real/imaginary part)
    z = func(w, c1, c2, r1, r2)
    return np.hstack([z.real, z.imag])
    
def fit(wlist, zrlist, zilist):                          #returns the fitting parameters 
    Zlist = np.hstack([zrlist, -zilist])
    popt, pcov = curve_fit(lambda w, c1, c2, r1, r2: func_imp(w, c1, c2, r1, r2, circuit.find_imp) , wlist, Zlist, p0 = None, maxfev = 10000)
    return popt, pcov

def plot_fit(wlist, popt, zrlist, zilist): 
    zfit = func_imp(wlist, *popt, circuit.find_imp)
    zfit_r = zfit[0: len(wlist)]
    zfit_i = zfit[len(wlist): 2 * len(wlist)]
    fig = plt.figure()
    plt.subplot((221))                          #z_real vs. freq
    plt.xscale('log')
    plt.plot(wlist, zrlist)
    plt.plot(wlist, zfit_r,'r--')
    plt.title("Z_real vs. freq")
    plt.subplot((222))                      #z_image vs. freq
    plt.xscale('log')
    plt.plot(wlist, zilist)
    plt.plot(wlist, -zfit_i,'r--')      #negative the fitted z_imag for Nyquist plot
    plt.title("Z_imag vs. freq")
    fig.add_subplot(2, 2, (3, 4))           #z_real vs. z_imag
    plt.plot(zrlist,zilist)
    plt.plot(zfit_r, -zfit_i,'r--')
    plt.title("Nyquist ")
    

def main(wlist, zrlist, zilist):
    popt, popv = fit(wlist, zrlist, zilist)
    plot_fit(wlist, popt, zrlist, zilist)
    print('the fitting parameters are', *popt)
    return popt, pcov

    
    

wlist, zrlist, zilist, fzlist = circuit.find_implist(1e-3,100,2,15,5,10)    #gernerating simulated data
popt, pcov = main(wlist, zrlist, zilist)
# #%%
# plot_fit(wlist, popt, zrlist, zilist)


























