# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:13:26 2022

@author: pokey

the pero is after the new understanding of v1 and Z_e.
-v1 directly obtained from the ratio impedance .
-Z_elec is obtained by differentiating the J's first term .
removed C_c for simplicity.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%
q = 1.6e-19     #charge of electron
k = 1.38e-23    #boltzmann constant
T = 300     #room temperature
VT = 0.026
n = 1.93
C_a = 2e-7
C_b = 2e-7
R_i = 5e8
n =  1.93
Vb = .3
C_g = 2.8e-8  
J_s = 7.1e-11
Vb = .2
V=0
nA=1.93
def Zcap(C , w):    #the impedance of a capacitor
    return 1/(1j * w * C)


#define the circuit of an equivalent circuit of a pervskite device
#required parameters are
#C_a,b,c,g: the capacitors in the circuit. See Onenote
#J_s: the saturation current of the transistor
#n: the ideality factor the transistor
#R_i: the ionic resistence
#V: the background voltage
#w: the frequency of the perturbation



def find_imp(w, C_a, C_b, R_i, C_g, J_s, n, V, Vb):  
    Z_d = 1 / (1/Zcap(C_g,w) + 1/R_i)    #the small chunk of cg and rion
    Z_ion = (Zcap(C_a,w) + Zcap(C_b,w) + Z_d) #the impedance of the ionic branch
    #Z_ion = 1/(1j*w*C_g + 1j*w*(C_a/2 - C_g)/(1 + 1j*w*R_i*C_a/2))#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #print('Z_ion is -----', Z_ion)
    #print('Z_ion 2 is ----',Z_ion2)
    v1_w = V * (1 - Zcap(C_a , w)/ Z_ion) #the v1 contributed by the perturbation voltage
    # Q = Vb * 1/(C_a**(-1) + C_b**(-1) + C_g**(-1))
    # vb_a = Q/C_a #the potential difference across c_a
    # vb1 = Vb-vb_a #the part of v1 contributed by the background voltage
    #v1= v1_w  
    #v1= v1_w + vb1
    vb1 = Vb * (C_a / (C_a +C_b))
    v1= vb1  #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #v1 = 0.5 * Vb
    #print('v1 is ----', v1)
    # Q = V/(1/C_a + 1/C_b + 1/C_c)
    # print('Q is ----', Q)
    # v1 = V - Q/C_a
    #print('v1 is ----',v1)
    #print('v1w is ----',v1_w)
    #print('vb1 is ----',vb1)
#######################################################################################
#this part used the Z_elct from the Matlab code (with a change Cion---C_g)
    J1 = J_s*(np.exp((v1-0)/(n*VT)) - np.exp((v1 - Vb)/(n* VT)))      # - np.exp((v1 - V)/VT))
    #J1 = J_s*(np.exp((v1-0)/(n*VT)))
    Jrec = J_s*np.exp(v1/(n*VT))
    Jgen = J_s*np.exp((v1 - Vb)/(n*VT))
    A = Zcap(C_a , w)/ Z_ion 
    djdv = (1 - A)*Jrec/(n*VT) + A*Jgen/(n*VT) #note this dv only concerns the perturbation part's contribution
    Z_elct = 1/djdv    
    #Z_elct = 1./((1 - A)*J1/(n*VT) + A*Jgen/(n*VT))
    #print(djdv,1111,1/Z_elct)    
    #Z_elct = 1./(1/2*(2 - 1./(1 + 1j*w*Z_ion*C_a/2))*J1/VT)  # different from the matlab version proly because the matlab used a different circuit
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct)
    #print('Z_ i is---------',Z_ion)
    # print("z_elct is -----", Z_elct)
    #print('z_tot is ------', Z_tot)
    #print('their difference is', Z_tot - Z_elct)
    #return Z_tot
    return Z_tot


#!!!!implist 里面的J v1 和上面的find——imp要同时改掉，很不方便！

def find_implist(w_h, w_l,  C_a, C_b, R_i, C_g,  J_s, n, V ,Vb):
    #wlist = np.arange(w_h, w_l, 1e-3)        #first resistence, second resistance, capacitance, type of plot(Nyquist or freqency)
    # Q = Vb * 1/(C_a**(-1) + C_b**(-1) + C_g**(-1))
    # vb_a = Q/C_a 
    # vb1 = Vb-vb_a 
    # v1= vb1  
    #J1 = J_s*(np.exp((v1-0)/(n*VT)) - np.exp((v1 - Vb)/(n* VT)))
    v1 = Vb * (C_a / (C_a +C_b))
    J1 = J_s*(np.exp((v1-0)/(n*VT)) - np.exp((v1 - Vb)/(n* VT)))      
    print('11111111----------------------------1',J1)
    wlist = np.logspace(w_h, w_l, 1000)
    zrlist = []                                 #reference Note section 1
    zilist = []
    fzlist = []
    j=0                                 #the y axis of the Bode spectrum (the effective capacitance)
    for w in wlist:
        j+=1
        #print('the parameters are',w, C_a, C_b, R_i, C_g, C_c, J_s, n, V, q_init)
        z= find_imp(w, C_a, C_b, R_i, C_g, J_s, n, V, Vb)
        zrlist.append(z.real)
        #print(z.real)
        zilist.append(-z.imag)                    # use positive zimag to keep image in first quadrant
        fzlist.append((1/z).imag / w)           #fzlist is the effective capacitance term in the Bode plot y axis
    return np.array(wlist), np.array(zrlist), np.array(zilist), fzlist, J1

#%%

# realistic parameters
#a, b, c, d= find_implist(-3,    3,  2.6e-7,2.6e-7,  3.8e5, 2.8e-8,  7.1e-11,  1.93, 2e-2, 1)
                        #(w_h, w_l,  C_a, C_b,      R_i, C_g,       J_s,      n, V ,Vb)
#a, b, c, d= find_implist(-3,    3,  2.6e-7,2.6e-7,  2e6, 2.8e-8,  7.1e-11,  1.93, 2e-2, 0.2)
#a, b, c, d= find_implist(-3,    6,  2.6e-7,2.6e-7,  2e6, 2.8e-8,  7.1e-11,  1.93, 2e-2, .32)
#a, b, c, d, J1= find_implist(-4,    5,  C_a,C_b,  R_i, 2.8e-8,  7.1e-11,  1.93, 2e-2, Vb)
a, b, c, d ,J1= find_implist(-4,    5,  C_a, C_b, R_i, C_g, J_s, n,V , Vb)
#when no background voltage, Z_elct should be really big
#a, b, c, d= find_implist(-3,    3,  2.6e-7,2.6e-7,  3.8e5, 2.8e-8,  7.1e-11,  1.93, 2e-2, 5)



plt.plot(b,c,'.')
plt.title('Nyquist plot')
plt.xlabel('z_real')
plt.ylabel('z_imag')

#%%
plt.plot(a,b,'.')
plt.plot(a,c,'.')
plt.xscale('log')
plt.yscale('log')
plt.legend(['z_real','z_imag'])
plt.xlabel('frequency')
plt.ylabel('magnitude of z')

#%% investigating the features of effective capacitance plot
#investigaitng the relation 1 and 2.
a, b, c, d,j1= find_implist(-4,    5,  C_a,C_b, R_i, C_g,  J_s,  nA, 0, Vb)
plt.plot(a, d)
plt.xscale('log')

a2, b2, c2, d2,j1= find_implist(-4,    5,   C_a,C_b,  R_i, C_g,  J_s,  nA, 0, 0)
plt.plot(a2, d2)
plt.xscale('log')
#plt.yscale('log')
plt.title('effective capacitance plot')

print('the start of the orange curve =', d2[0])
c_eff = 1/(1/C_a + 1/C_b + 1/C_g)
print('the theoretical C_eff = ', c_eff)


#C_eff ~= start of 0V capacitance curve as expected.

#%% investigaitng the feature of the two curve Nyquist plots
#investigating vertex 
#investigating relation 3,4,5,6

a, b, c, d ,J1= find_implist(-4,    5,  C_a, C_b, R_i, C_g, J_s, n, V, Vb)
-4,    5,  2.6e-7,2.6e-7,  2e6, 2.8e-8,  7.1e-11,  1.93, 2e-2, .32

plt.plot(b,c,'.')
plt.title('Nyquist plot')
plt.xlabel('z_real')
plt.ylabel('z_imag')

zrlist = b
zilist = c


#zizr = np.array([[0,0]])
zizr = np.empty([0,2])
for n in range (0, len(b)):
    zizr = np.concatenate((zizr,np.array([[zilist[n], zrlist[n]]])))
#dict(sorted(zizr.items()))
zizr0 = zizr
#zizr = np.sort(zizr, axis = 1)
zizr1 = zizr[zizr[:,1].argsort()[::-1]]



def find_top(zizr):
    n = 999
    vlist = [] #points of local maximum
    nlist = [] #index of local maximum
    nllist = [] #points of local minimum
    while n >1:
        if zizr[n][0] > zizr[n-1][0]:
            vlist.append(zizr[n][1])
            nlist.append(n)
            # print('break')
            break
        n -= 1
        # print(n)
    while n >1:
        # print(n)
        # print(zilist[n] , zilist[n-1])
        # print(n)
        if zizr[n][0] < zizr[n-1][0]:
            #print('change')
            nllist.append(n)
            break 
        n -= 1
        
        #print(n)
    while n >1:
        if zizr[n][0] > zizr[n-1][0]:
            vlist.append(zizr[n][1])
            #print('change back')
            nlist.append(n)
            break
        n -= 1
        # print(n)
    return vlist,nlist,nllist


vlist, nlist, nllist = find_top(zizr)

for i in range (0,len(nlist)):
    plt.plot(zizr[nlist[i]][1], zizr[nlist[i]][0],'r.')

#print(find_top(zizr),'vlist,nlist,nllist')



#%% investigating the bottom length relation 5, 6
#a, b, c, d ,J1= find_implist(-4,    5,  C_a,C_b,  2e8, 2.8e-8,  7.1e-11,  1.93, 2e-2, .2)

nA = 1.93
r_reci = nA * VT/ J1 #impedance for infinite frequency
r_rec0 = nA *VT /(J1) * (C_a+C_b)/C_a 
print(n)# 这里的n是错的
print(r_reci)
print(r_rec0)



#%%
wlist = a
zrlist = b
zilist = c
zizr2 = np.empty([0,3])
for n in range (0, len(b)):
    zizr2 = np.concatenate((zizr2,np.array([[zilist[n], zrlist[n], wlist[n]]])))
#dict(sorted(zizr.items()))
zizr2 = zizr2[zizr2[:,1].argsort()][::-1]

wlist = []
tlist = []
for i in nlist:
    tlist.append( 1/zizr2[i][2])   #the time constant t = w^-1
    wlist.append(zizr2[i][2])       # the frequency

print('wlist is',wlist)
print('tlist is', tlist)
t1 = r_reci *C_a/2
t2 = R_i * C_a 

print('t0,t infinity =',t2, t1)




#%%

a, b, c, d ,J1= find_implist(-4,    5,  2.6e-7,2.6e-7,  2e6, 2.8e-8,  7.1e-11,  1.93, 2e-2, .32)
plt.plot(b,c,'.')
plt.title('Nyquist plot')
plt.xlabel('z_real')
plt.ylabel('z_imag')
















