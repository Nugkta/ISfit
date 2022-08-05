# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 23:43:33 2022

@author: pokey
In this file, I cleaned up the function for the equivalent circuit, and improved the the feature extracion function. Also, I added more documentation.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import pandas as pd
#defining the constants and inputted parameters for testing
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

#%%

def Zcap(c, w): #returns the impedance of a capacitor
    return 1 / (1j * w * c)

def pero_model(w, C_a, C_b, R_i, C_g, J_s, n, Vb): #w is the list of frequency range
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_i)
    Z_ion = (Zcap(C_a , w) + Zcap(C_b , w) + Z_d) #the impedance of the ionic branch
    v1 = Vb * (C_a / (C_a + C_b)) 
    J1 = J_s*(np.exp((v1 - 0) / (n * VT)) - np.exp((v1 - Vb) / (n * VT))) #the current densitf of the electronic branch
    #J1 = J_s*(np.exp((v1 - 0) / (n * VT)) - 0)
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

C_eff_t = 1/(1/C_a + 1/C_b ) #the theoretical C_eff from the supposed relation
C_sum = 1/(1/C_a + 1/C_b )
print(C_eff, C_eff_t)









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


whlist = wzlist[nhlist][: , 0]
tlist = 1 / whlist 

r_reci = nA * VT/ J1 #impedance for infinite frequency
r_rec0 = nA *VT /(J1) * (C_a+C_b)/C_a 

print('nhlist is', nhlist)
print('wlist is',whlist)
print('tlist is', tlist)
r_reci = nA * VT/ J1
t1 = r_reci *C_eff
t2 = R_i * C_sum

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
w_t = 1 / C_eff * (1 / R_i + 1 / r_reci)# the omega in relation 7
print(w_t ) 


#%% Now try to code a function that's gonna take in a simulated set of data and spit out a set of reasonable initial guess

# First, generating the data set of frequency vs. Impedance Z (complex)
w = np.logspace(-4 , 5 , 1000)
zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, Vb)
wzlist = np.stack((w, zlist), axis = -1) #the wzlist represents the data set
wzlist = wzlist[wzlist[: , 1].real.argsort()]
# Obtain k, a ratio of C_a and C_b, and rec0, reci by relation 5, and 6
nhlist, nllist= find_extremum(wzlist)
r_reci_e = wzlist[nllist[0]][1].real
r_rec0_e = wzlist[-1][1].real
k = r_rec0_e / r_reci_e

#Obtain w3, w4 in relation 3 and 4
whlist = wzlist[nhlist][: , 0]
w4 = whlist[0] #w4 corresponds to lower frequency ---- lower n ---- first element in whlist
w3 = whlist[1]

#obtain w7 from relation 7 (need to use another set of data with 0 background voltage)
zlist_0, J1_0 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, 0)
wzlist_0 = np.stack((w, zlist), axis = -1)
wzlist_0 = wzlist[wzlist[: , 1].real.argsort()]#a corresponding list of w and z, then sort the list by the real part of z
nhlist_0 , nllist_0 = find_extremum(wzlist)
whlist_0 = wzlist_0[nhlist_0][: , 0]
w7 = whlist_0[0]


C_ap = (1 / zlist_0).imag / w
C_eff_e = C_ap[0] #estimated C_eff  不是很对 这里应该是C_sum

R_i_e = 1 / (C_eff_e * w7 -1/ r_reci_e)



#Obtain C_sum by solving the second order equation#
#C_sum = (1/(r_reci * w7))/ (1 - w4 / w7 )
# x = symbols('x')
# eq1 = Eq(w4 * x ** 2 + (1 / r_reci) * x - w7, 0)
# sol = solve(eq1)


# for i in sol:
#     if i > 0:
#         C_sum = float(i)
#C_eff = 1/(1/C_a + 1/C_b + 1/C_g)
C_eff = 1/(1/C_a + 1/C_b )
r_reci = nA * VT/ J_s
#C_a_e = C_sum * k / (k-1) #estimated C_a
print(C_eff, C_eff_e)
print(w_t, w7) #result from previous, just for testing
print(r_reci , r_reci_e)

R_i_t = 1 / (C_eff * w_t -1/ r_reci)

print(R_i_e, R_i_t)







#%% RETRY FINDING THE FIT PARAMETERS

#First, generate a list of Impedance spectrum data of different Vb

#might need to use dataframe here



Vblist = np.linspace(0, 1, 20)
w = np.logspace(-6 , 10 , 1000)
Vb_z_list = np.empty([2 , 1000])
Vb_wzjv_list = [] #the list of dataframes, each dataframe corresponds to a specific bias voltage
for V in Vblist:
    zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, V)
    wzlist = np.stack((w , zlist) , axis = -1)
    wzlist = wzlist[wzlist[:, 1].real.argsort()]
    vlist = np.ones(len(wzlist)) * V
    jlist = np.ones(len(wzlist)) * J1
    wz_v_jlist = np.column_stack((wzlist , vlist,jlist))
    #v_wz_jlist = np.column_stack((v_))
    df = pd.DataFrame(wz_v_jlist , columns=['frequency' , 'impedance' , 'bias voltage','recomb current'])
    Vb_wzjv_list.append(df)
Vb_wzjv_list
    


#%%list of functions for finding the initial guesses
###################### the sumulated data are generated and stored in Vb_wzlist   


#STEP0
def find_k(dfs): #function for finding an appropriate k (ratio of C_a/(C_a +C_b) from a list of dataframes just like above
    klist = []
    for df in dfs:
        zlist = df['impedance'].to_numpy()
        #plt.plot(zlist.real , -zlist.imag,'.')
        #plt.show()
        wzlist = df[['frequency','impedance']].to_numpy()
        nhlist, nllist= find_extremum(wzlist)
        r_reci_e = wzlist[nllist[0]][1].real
        r_rec0_e = wzlist[-1][1].real
        k = r_rec0_e / r_reci_e
        klist.append(k)
    for i in range (0, len(klist)-1):
        if np.abs(klist[i+1]-klist[i]) < 0.001: #Here I used 0.001 as the threshold indicating the k value is stablised arbitrarily. Could be changed in the future
            return klist[i+1]
            break
    print('the k is not stablised for this range of bias voltage')

# def find_nA_Js(dfs , k): #need to input the list of dataframes and the ratio k found before.
#     jlist = []
#     vlist = []  
#     for wzjvdf in dfs:
#         vlist.append(wzjvdf['bias voltage'].values[-1])#-1 because the element at -1 corresponds to lowest frequency ~steady state
#         jlist.append(wzjvdf['recomb current'].values[-1])
#     log_jlist =np.log( np.array(jlist)[1:].real) #[1:]because the first element gives log0
#     vlist = np.array(vlist).real[1:]
#     grad,b = np.polyfit(vlist, log_jlist,1)
#     nA_e = 1/VT*grad/k
#     log_Js = -b/grad
#     Js_e = np.exp(log_Js)
#     #x = np.linspace(0,7,100)
#     # poly1d_fn = np.poly1d([grad,b]) 
#     # plt.plot(x , poly1d_fn(x),'--k')
#     # plt.plot(vlist,log_jlist,'.')
#     # plt.xlim([-1,5])
#     # plt.ylim([-25,25])
#     return Js_e , nA_e

#STEP1
def find_nA_Js(dfs , k):
    jlist = []
    vlist = []  
    for wzjvdf in dfs:
        vlist.append(wzjvdf['bias voltage'].values[-1]) #-1 because the element at -1 corresponds to lowest frequency ~steady state
        jlist.append(wzjvdf['recomb current'].values[-1])
    log_jlist =np.log( np.array(jlist)[1:].real) #[1:]because the first element gives log0
    vlist = np.array(vlist).real[1:]
    #x = np.linspace(0,7,100)
    grad,b = np.polyfit(log_jlist , vlist ,1)
    log_Js = -b/grad
    #poly1d_fn = np.poly1d([grad,b]) 
    #plt.plot(log_jlist , poly1d_fn(log_jlist),'--k')
    #plt.plot(log_jlist,vlist,'.')
    # plt.xlim([-1,5])
    # plt.ylim([-25,25])
    Js_e = np.exp(log_Js)
    nA_e = 1/VT*grad/k
    return  Js_e,nA_e

#STEP2, 3
def find_Cabg(dfs , k):
    exist = False
    for df in dfs:
        if df['bias voltage'][0] == 0: #only use the dataframe with 0 bias voltage.
            wzjv0 = df
            exist = True
    if exist == False:
        print('No 0 bias situation')
    zlist = np.array(wzjv0['impedance'].values)
    w = np.array(wzjv0['frequency'].values)
    C_ap = (1 / zlist).imag / w
    C_sum = C_ap[-1].real
    C_a_e = (1 + 1/(k-1)) * C_sum  #the estimated C_a
    C_b_e = (k - 1) * C_a     #the estimated C_b
    C_g = C_ap[1].real       #the estiamted C_g
    return C_a_e , C_b_e , C_g  

#STEP 4
def find_Ri(dfs): 
    df = dfs[-1]#using the last dataframe to gurantee that the k is stablised
    zlist = df['impedance'].to_numpy()
    wzlist = df[['frequency','impedance']].to_numpy()
    nhlist, nllist= find_extremum(wzlist)
    whlist = wzlist[nhlist][: , 0]
    plt.plot(wzlist[:,1].real,-wzlist[:,1].imag)
    w4 = min(whlist)
    R_i_e = 1/(w4 * C_a).real
    return R_i_e

def get_init_guess(dfs):
    k = find_k(dfs) #k is obtained, Vb_wzjv_list is the list of dataframes storing the experiment data.
    Js_e , nA_e = find_nA_Js(dfs , k)
    C_a_e , C_b_e , C_g_e = find_Cabg(dfs , k)
    R_i_e = find_Ri(dfs)
    return [C_a_e , C_b_e, C_g_e , R_i_e , nA_e ,Js_e]






#%% MAIN 
init_guess = get_init_guess(Vb_wzjv_list)
























#%% testing
zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, Vb)
wzlist = np.stack((w , zlist) , axis = -1)
wzlist = wzlist[wzlist[:, 1].real.argsort()]
vlist = np.ones(len(wzlist)) * Vb
jlist = np.ones(len(wzlist)) * J1
v_wzlist = np.column_stack((wzlist , vlist, jlist))
df = pd.DataFrame(v_wzlist , columns=['frequency' , 'impedance' , 'bias voltage', 'recomb current'])

#print(type(df['impedance'].to_numpy()))

#%%temporary run for finding j and n (step1)
jlist = []
vlist = []  
for wzjvdf in Vb_wzjv_list:
    vlist.append(wzjvdf['bias voltage'].values[-1]) #-1 because the element at -1 corresponds to lowest frequency ~steady state
    jlist.append(wzjvdf['recomb current'].values[-1])
log_jlist =np.log( np.array(jlist)[1:].real) #[1:]because the first element gives log0
vlist = np.array(vlist).real[1:]

x = np.linspace(0,7,100)

grad,b = np.polyfit(log_jlist , vlist ,1)
log_Js = -b/grad
poly1d_fn = np.poly1d([grad,b]) 
plt.plot(log_jlist , poly1d_fn(log_jlist),'--k')
plt.plot(log_jlist,vlist,'.')
# plt.xlim([-1,5])
# plt.ylim([-25,25])
Js_e = np.exp(log_Js)
nA_e = 1/VT*grad/k

print(nA_e, Js_e)

#plt.xscale('log')






#%% implementing step 2 , 3
Vb_wzjv_list
for df in Vb_wzjv_list:
    if df['bias voltage'][0] == 0:
        wzjv0 = df
zlist = np.array(wzjv0['impedance'].values)
w = np.array(wzjv0['frequency'].values)
C_ap = (1 / zlist).imag / w
C_sum = C_ap[-1].real
C_a_e = (1 + 1/(k-1)) * C_sum      #the estimated C_a
C_b_e = (k - 1) * C_a   ##the estimated C_b
C_g = C_ap[1].real




#%% implementing step 4

df = Vb_wzjv_list[-1]#using the last dataframe to gurantee that the k is stablised
zlist = df['impedance'].to_numpy()
wzlist = df[['frequency','impedance']].to_numpy()
nhlist, nllist= find_extremum(wzlist)
whlist = wzlist[nhlist][: , 0]
plt.plot(wzlist[:,1].real,-wzlist[:,1].imag)
# for i in range (0,len(nhlist)):
#     plt.plot(wzlist[nhlist[i]][1].real, -wzlist[nhlist[i]][1].imag,'r.')
w4 = min(whlist)
R_i_e = 1/(w4 * C_a)
print(R_i_e)













#%% TEST MAIN
    
k = find_k(Vb_wzjv_list) #k is obtained, Vb_wzjv_list is the list of dataframes storing the experiment data.
Js_e , nA_e = find_nA_Js(Vb_wzjv_list , k)
C_a_e , C_b_e , C_g_e = find_Cabg(Vb_wzjv_list , k)
R_i_e = find_Ri(Vb_wzjv_list)



print('the estimated and actual k are  %.2f %.2f' %(k , (C_a + C_b)/C_a))
print('the estimated and actual Js are %.1e %.1e' %(Js_e, J_s))
print('the estiamted and actual nA are  %.2f %.2f' %(nA_e , nA))
print('the estiamted and actual C_a are  %.1e %.1e' %(C_a_e , C_a))
print('the estiamted and actual C_b are  %.1e %.1e' %(C_b_e , C_b))
print('the estiamted and actual C_g are  %.1e %.1e' %(C_g_e , C_g))
print('the estiamted and actual R_ion are  %.2d %.2d' %(R_i_e , R_i))

# klist = []
# for df in Vb_wzjv_list:
#     zlist = df['impedance'].to_numpy()
#     plt.plot(zlist.real , -zlist.imag,'.')
#     plt.show()
#     wzlist = df[['frequency','impedance']].to_numpy()
#     nhlist, nllist= find_extremum(wzlist)
#     r_reci_e = wzlist[nllist[0]][1].real
#     r_rec0_e = wzlist[-1][1].real
#     k = r_rec0_e / r_reci_e
#     klist.append(k)
#     v_wzlist = np.stack((wzlist , vlist) , axis = -1)
#     plt.plot(zlist.real , -zlist.imag , '.')
#     plt.title('the bias voltage is' + str(i))
#     plt.show()
    
    
    
    #Vb_z_list = np.append(Vb_z_list , np.array([zlist]), axis = 0)

#Vb_z_list stores all the zlist data with different bias voltage Vb (act as the actual data sets collected)
#for z_list in Vb_z_list:

    
