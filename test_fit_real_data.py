# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 23:26:48 2022

@author: pokey
"""

import read_func as rf
import pero_ig_fit_func  as pif





#%%
dfs = rf.read_imp_folder('C:/Users/pokey/Documents/UROP\Humidity Dark Impedance Data\MAPI\MAPIdev2p5MgCl30C/')


#%%
init_guess = pif.get_init_guess(dfs)








































