# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 21:39:01 2022

@author: hx5118

this
"""
import numpy as np
import matplotlib.pyplot as plt
import perovskite_circuit as circuit



def generate_data(w_h, w_l,  C_a, C_b, R_i, C_g, C_c, J_s, n, V, q_init):
    wlist, zrlist, zilist, fzlist = circuit.find_implist(w_h, w_l,  C_a, C_b, R_i, C_g, C_c, J_s, n, V, q_init)
    return  wlist, zrlist, zilist, fzlist 

wlist, zrlist, zilist, fzlist = generate_data(1e-4, 10., 1., 1., 1., 1., 1., 1., 1., 1., 10)

#%%

plt.plot(wlist, zrlist,'.')
plt.show()
#%%
plt.plot(wlist, fzlist)
plt.xscale('log')

#%%
wlist, zrlist, zilist, fzlist = generate_data(1e-4, 10, 1e-4,1e-4,4,4,1e-4,1e-4, 1,5,1)

#%%
plt.plot(zrlist, -zilist,'.')







































