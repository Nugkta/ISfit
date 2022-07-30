# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 23:43:33 2022

@author: pokey
In this file, I cleaned up the function for the equivalent circuit, and improved the the feature extracion function. Also, I added more documentation.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#defining the constants and inputted parameters for testing
VT = 0.026
C_a = 2e-7
C_b = 2e-7
R_i = 5e8
Vb = .2
C_g = 2.8e-8  
J_s = 7.1e-11
nA = 1.93


#%%

def Zcap(c, w): #returns the impedance of a capacitor
    return 1 / (1j * w * c)

def pero_model(w, C_a, C_b, R_i, C_g, J_s, n, Vb): #w is the list of frequency range
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_i)
    Z_ion = (Zcap(C_a , w) + Zcap(C_b , w) + Z_d) #the impedance of the ionic branch
    v1 = Vb * (C_a / (C_a + C_b)) 
    J1 = J_s*(np.exp((v1 - 0) / (n * VT)) - np.exp((v1 - Vb) / (n * VT))) #the current densitf of the electronic branch
    Jrec = J_s * np.exp(v1 / (n * VT))        #the recombination current and the generation current
    Jgen = J_s * np.exp((v1 - Vb) / (n * VT))
    A = Zcap(C_a , w)/ Z_ion
    djdv = (1 - A) * Jrec / (n * VT) + A * Jgen / (n * VT)
    Z_elct = 1 / djdv #the impedance of the electronic branch
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct)
    return Z_tot, J1



#%% generating a set of data and plotting


w = np.logspace(-4 , 5 , 1000)
zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, Vb)
zreal = zlist.real
zimag = -zlist.imag



#Nyquist plot
plt.figure()
plt.plot(zreal , zimag , '.')
plt.title('Nyquist plot')
plt.xlabel('z_real')
plt.ylabel('z_imag') 


#real, imag vs. freq plot
plt.figure()
plt.plot(w , zreal, '.')
plt.plot(w , zimag ,'.')
plt.xscale('log')
plt.yscale('log')
plt.legend(['z_real','z_imag'])
plt.xlabel('frequency')
plt.ylabel('magnitude of z')


#Bode plot 
plt.figure()
C_ap = (1 / zlist).imag / w #the list of apparent capacitance
plt.plot(w, C_ap)
plt.xscale('log')
plt.title('Bode plot')
plt.ylabel('Apparent capacitance')
plt.xlabel('Frequency $\omega$')


#%% 
# now using functions to automatically extract relations from the plotted data



#%% for relation 2:
zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, 0) # relation 2 is specifically for the case of 0 background voltage so generating new set of data.
C_ap = (1 / zlist).imag / w
C_eff = C_ap[0] #the C_eff extract from the plot

C_eff_t = 1/(1/C_a + 1/C_b + 1/C_g) #the theoretical C_eff from the supposed relation

print(C_eff , C_eff_t,'Here for some reason they are not equal. Need to be fixed!')









#%% For relation 3 and 4

def find_extremum(wzlist): #algorithm to find the extremums of the Nyquist plot
    zlist = wzlist[:,1]
    n = 0
    nhlist = [] #index of local maximum(high)
    nllist = [] #points of local minimum(low)
    seek = 'max' #will encounter a maximum first#
    while n < len(zlist) - 2:
        while seek == 'max':
            n += 1
            if -zlist[n].imag > -zlist[n+1].imag:
                nhlist.append(n)
                seek = 'min'
            if n>= len(zlist) -2:
                break
        while seek == 'min':
            n += 1
            if -zlist[n].imag < -zlist[n+1].imag:  #note that the imaginary part should be nagative in the Nyquist plot.
                nllist.append(n)
                seek = 'max'
            if n>= len(zlist) -2:
                break
    return nhlist, nllist



zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, Vb) #regenerate the data set
plt.figure()
plt.plot(zreal , zimag , '.')
plt.title('Nyquist plot')
plt.xlabel('z_real')
plt.ylabel('z_imag') 


wzlist = np.stack((w, zlist), axis = -1)
wzlist = wzlist[wzlist[: , 1].real.argsort()]#a corresponding list of w and z, then sort the list by the real part of z


nhlist, nllist= find_extremum(wzlist)
plt.plot(zreal , zimag , '.')
for i in range (0,len(nhlist)):
    plt.plot(wzlist[nhlist[i]][1].real, -wzlist[nhlist[i]][1].imag,'r.')


whlist = []  
tlist = []
for i in nhlist:
    tlist.append(1/wzlist[i][0])   #the time constant t = w^-1
    whlist.append(wzlist[i][0])       # the frequency

print('nhlist is', nhlist)
print('wlist is',whlist)
print('tlist is', tlist)
r_reci = nA * VT/ J1
t1 = r_reci *C_a/2
t2 = R_i * C_a 

print('t infinity, t0 =',t1, t2)


#%% verifying relation 5, 6
r_reci = nA * VT/ J1 #impedance for infinite frequency
r_rec0 = nA *VT /(J1) * (C_a+C_b)/C_a 
print('end of circle 1 given by relation 5 is', r_reci)
print('the actual end of circle 1 is', wzlist[nllist[0]][1].real)
print('end of circle 1 given by relation 5 is', r_rec0)
print('the actual end of circle 1 is', wzlist[-1][1].real)
             


#%%verifying relation 7
zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, 0)
plt.plot(zlist.real , -zlist.imag , '.')
wzlist = np.stack((w, zlist), axis = -1)
wzlist = wzlist[wzlist[: , 1].real.argsort()]#a corresponding list of w and z, then sort the list by the real part of z
nhlist, nllist = find_extremum(wzlist)
for i in range (0,len(nhlist)):
    plt.plot(wzlist[nhlist[i]][1].real, -wzlist[nhlist[i]][1].imag,'r.')


whlist = []  
tlist = []
for i in nhlist:
    whlist.append(wzlist[i][0])       # the frequency

r_reci = nA * VT/ J_s
C_eff = 1/(1/C_a + 1/C_b + 1/C_g)
print(whlist[0])
w_t = 1 / C_eff * (1 / R_i + 1 / r_reci)
print(w_t )























































