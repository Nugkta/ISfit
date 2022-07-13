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
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def capimp(c, w):  #finding the impedance of a capacitor.   Input(capacitance, frequency)
    z = 1/(w*c*1j)
    return z


# finding the total impedance of the circuit given in section 2 of notes
def find_imp(w, c1, c2, r1, r2): 
    z_c1 = capimp(c1,w)
    z_c2 = capimp(c2,w)
    z1 = r2*z_c2/(r2+z_c2)
    z2 = r1 + z1
    z_tot = (z_c1*z2) / (z_c1 + z2)   
    return z_tot


# Findling Z for a list of input frequencies
def find_implist(w_h, w_l, c1, c2, r1, r2):
    wlist = np.arange(w_h, w_l, 0.001)        #first resistence, second resistance, capacitance, type of plot(Nyquist or freqency)
    zrlist = []                                 #reference Note section 1
    zilist = []
    fzlist = []                                 #the y axis of the frequency spectrum
    for w in wlist:
        z = find_imp(w, c1, c2, r1, r2)
        zrlist.append(z.real)
        zilist.append(-z.imag)                    # use positive zimag to keep image in first quadrant
        fzlist.append((1/z).imag / w)           #fzlist is the effective capacitance term in the Bode plot y axis
    return wlist, zrlist, zilist, fzlist

#Plotting Nyquist plot for zr and zi list
def plot_N(zrlist, zilist):         #parameters are: high end of freq, low end of freq,   
    plt.plot(zrlist, zilist,'.')
    plt.xlabel('real z')
    plt.ylabel('imag z')
    plt.title('Nyquist plot')
    plt.show()
    
#plotting Bode plot for a frequency list and effective capacitance  
def plot_B(wlist, fzlist):   
    plt.plot(wlist, fzlist,'.')
    plt.title('Bode plot')
    plt.xlabel('Freqency')
    plt.ylabel('Im($Z^{-1}$)/$\omega$')
    plt.xscale('log')


#general function that plot Nyquist or Bode plot for a given circuit and a given range of frequencies
def plot_spec(w_h, w_l, c1, c2, r1, r2, tp):
    wlist, zrlist, zilist, fzlist = find_implist(w_h, w_l, c1, c2, r1, r2)
    if tp == 'N':
         plot_N(zrlist, zilist)   
    if tp == 'B':
        plot_B(wlist, fzlist)   


#%%Plotting the Nyquist plot
plot_spec(1e-3,100,2,15,5,10,'N')      #double semicircle parameters



#%% now try to curve fit the data coming out of the simulation zrlist vs. zilist
wlist, zrlist, zilist, fzlist = find_implist(1e-3,100,2,15,5,10) 
plt.plot(zilist,wlist,'.')
plt.xscale("log")


#%%
def func_imp(w, c1, c2, r1, r2, func):                   #the funtion used in the curve fit
    z = func(w, c1, c2, r1, r2)
    return np.hstack([z.real, z.imag])
    #return np.stack((z.real, z.imag), axis = 1)

Zlist = np.hstack([zrlist, zilist])
#Zlist = np.stack((zrlist, zilist), axis = 1)

popt, pcov = curve_fit(lambda w, c1, c2, r1, r2: func_imp(w, c1, c2, r1, r2, find_imp) , wlist, Zlist, p0 = None, maxfev = 10000)

#%%
zfit = func_imp(wlist, *popt,find_imp)
zfit_r = zfit[0: len(wlist)]
zfit_i = zfit[len(wlist): 2 * len(wlist)]

plt.xscale('log')
plt.plot(wlist, zrlist)
plt.plot(wlist, zfit_r,'r--')
#%%
plt.xscale('log')
plt.plot(wlist, zilist)
plt.plot(wlist, zfit_i,'r--')
#%%
plot_spec(1e-3,100,2,15,5,10,'N')  
plt.plot(zfit_r, zfit_i,'r--')



