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

def plot_spec(w_h, w_l, c1, c2, r1, r2, tp):         #parameters are: high end of freq, low end of freq,   
    wlist = np.arange(w_h, w_l, 0.001)        #first resistence, second resistance, capacitance, type of plot(Nyquist or freqency)
    zrlist = []                                 #reference Note section 1
    zilist = []
    fzlist = []                                 #the y axis of the frequency spectrum
    for w in wlist:
        z = find_imp(w, c1, c2, r1, r2)
        zrlist.append(z.real)
        zilist.append(-z.imag)                    # use positive zimag to keep image in first quadrant
        fzlist.append((1/z).imag / w)
    if tp == 'N':
        plt.plot(zrlist, zilist,'.')
        plt.xlabel('real z')
        plt.ylabel('imag z')
        plt.title('Nyquist plot')
        plt.show()
        return zrlist, zilist
    if tp == 'F':
        plt.plot(wlist, fzlist,'.')
        plt.title('Freqency plot')
        plt.xlabel('Freqency')
        plt.ylabel('Im($Z^{-1}$)/$\omega$')
        plt.xscale('log')
        
#%%

plot_spec(1e-2,100,2,3,15,7,'N')
plot_spec(1e-2,100,1,1,10,10,'N')
# plot_spec(1,100,1,1,10,15,'N')


#%%
plot_spec(1e-2,100,1,1,10,10,'F')
    
#%% testing

a, b = find_imp(0.00001,5,5,2,2 )

print(a,b)

























