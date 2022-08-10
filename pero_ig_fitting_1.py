# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:03:47 2022

@author: pokey

perovskite model & initial guess & fitting
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import pandas as pd
from scipy.signal import argrelextrema
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


#%% defining all functions 

#For calculating impedance of a capacitor
def Zcap(c, w): #returns the impedance of a capacitor
    return 1 / (1j * w * c)


#For model of the perovskite
def pero_model(w, C_a, C_b, R_i, C_g, J_s, n, Vb): #w is the list of frequency range
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_i)
    Z_ion = (Zcap(C_a , w) + Zcap(C_b , w) + Z_d) #the impedance of the ionic branch
    v1 = Vb * (C_a / (C_a + C_b))
    J1 = J_s*(np.e**((v1 - 0) / (n * VT)) - np.e**((v1 - Vb) / (n * VT))) #the current densitf of the electronic branch
    #J1 = J_s*(np.exp((v1 - 0) / (n * VT)) - 0)
    Jrec = J_s * np.e**(v1 / (n * VT))        #the recombination current and the generation current
    Jgen = J_s * np.e**((v1 - Vb) / (n * VT))
    A = Zcap(C_a , w)/ Z_ion
    djdv = (1 - A) * Jrec / (n * VT) + A * Jgen / (n * VT)
    Z_elct = 1 / djdv #the impedance of the electronic branch
    Z_tot = 1 / (1/Z_ion + 1/ Z_elct)
    return Z_tot, J1

#For finding the position of extrema of the Nyquist plot
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
            # print(n)
            # print(len(zlist))
            if -zlist[n].imag < -zlist[n+1].imag:  #note that the imaginary part should be nagative in the Nyquist plot.
                nllist.append(n)
                seek = 'max'
            if n>= len(zlist) -2:
                break
    return nhlist, nllist



#function below are all for finding the initial guess

#STEP0
def find_k(dfs): #function for finding an appropriate k (ratio of C_a/(C_a +C_b) from a list of dataframes just like above
    klist = []
    for df in dfs:
        zlist = df['impedance'].to_numpy()
        #plt.plot(zlist.real , -zlist.imag,'.')
        #plt.show()
        wzlist = df[['frequency','impedance']].to_numpy()
        nhlist = np.array(argrelextrema(wzlist[:,1].imag, np.less))[0]
        nllist = np.array(argrelextrema(wzlist[:,1].imag, np.greater))[0]
        # nhlist, nllist= find_extremum(wzlist)
        # print(nhlist, 'is nllist')
        # print(nhlist1[0], 'is nllist222222222')
        r_reci_e = wzlist[nllist[0]][1].real
        r_rec0_e = wzlist[-1][1].real
        k = r_rec0_e / r_reci_e
        klist.append(k)
    for i in range (0, len(klist)-1):
        if np.abs(klist[i+1]-klist[i]) < 0.001: #Here I used 0.001 as the threshold indicating the k value is stablised arbitrarily. Could be changed in the future
            return klist[i+1]
            break
    print('the k is not stablised for this range of bias voltage')

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
    #zlist = df['impedance'].to_numpy()
    wzlist = df[['frequency','impedance']].to_numpy()
    nhlist, nllist= find_extremum(wzlist)
    whlist = wzlist[nhlist][: , 0]
    plt.plot(wzlist[:,1].real,-wzlist[:,1].imag)
    w4 = min(whlist)
    R_i_e = 1/(w4 * C_a).real
    return R_i_e

#For finding the initial guess using the functions defined above
def get_init_guess(dfs):
    k = find_k(dfs) #k is obtained, Vb_wzjv_list is the list of dataframes storing the experiment data.
    Js_e , nA_e = find_nA_Js(dfs , k)
    C_a_e , C_b_e , C_g_e = find_Cabg(dfs , k)
    R_i_e = find_Ri(dfs)
    return [C_a_e , C_b_e,  R_i_e ,C_g_e ,Js_e, nA_e ]


#%% generating simulated sets of data
Vblist = np.linspace(0, 1, 20)      #defining the range of the bias volatage
w = np.logspace(-6 , 10 , 1000)     #defining the range of the frequency 
Vb_z_list = np.empty([2 , 1000])    #initialising Vbzlist
Vb_wzjv_list = [] #initialising the list of dataframes, each dataframe corresponds to a specific bias voltage
for V in Vblist:
    zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, V)
    wzlist = np.stack((w , zlist) , axis = -1)
    wzlist = wzlist[wzlist[:, 1].real.argsort()]     # sorting the wzlist by the real part of the impedance(preparing for the find extrema function)
    vlist = np.ones(len(wzlist)) * V
    jlist = np.ones(len(wzlist)) * J1
    wz_v_jlist = np.column_stack((wzlist , vlist,jlist))
    #v_wz_jlist = np.column_stack((v_))
    df = pd.DataFrame(wz_v_jlist , columns=['frequency' , 'impedance' , 'bias voltage','recomb current'])
    Vb_wzjv_list.append(df)

#Vb_wzjv_list, the sequence of dfs to be analysed and fit  is generated here. 

#%% Finding the initial guess
init_guess = get_init_guess(Vb_wzjv_list)
C_a_e , C_b_e, C_g_e , R_i_e , nA_e ,Js_e = init_guess
print(init_guess)







#%% the function for the global fit

#in order to fit the complex function, split it to real and imaginary part and then do the fit to the stacked list of real and imaginary part
def pero_sep(wvb, C_a, C_b, R_i, C_g, J_s, n):
    z = pero_model(wvb[:,0], C_a, C_b, R_i, C_g, J_s, n, wvb[:,1])[0] # because w and vb are both regarded as variables here.
    return np.hstack([z.real, z.imag])          # this will return a list of the form [z1real, z2real,...,z1imag, z2imag, ...]



#the function for global fitting, need to input the list of df and the initial guess obtained above. 
def global_fit(dfs, init_guess):
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
    popt, pcov = curve_fit(pero_sep , wvlist_big, Zlist_big,p0 = init_guess)   #fitting the function to the variables (wv(independent) and z(dependent)) for parameters
    return popt, pcov
#%%

init_guess = get_init_guess(Vb_wzjv_list)
print('the initial guesses are', init_guess)
popt, pcov = global_fit(Vb_wzjv_list , init_guess)
print('the fitted parameters are',popt)
print('the original parameters are',C_a, C_b, R_i, C_g, J_s, nA )




















#%% TRY FITTING BY SETTING BIAS VOLTAGE AS ANOTHER VARIABLE

# def pero_sep(w, C_a, C_b, R_i, C_g, J_s, n,Vb):
#     z = pero_model(w, C_a, C_b, R_i, C_g, J_s, n, Vb)[0]
#     return np.hstack([z.real, z.imag])

#Vb_wzjv_lists = Vb_wzjv_list[0]


from scipy.optimize import curve_fit
zlist_big = np.array([])
wlist_big = np.array([])
vlist_big = np.array([])
for df in Vb_wzjv_list:
    zlist_big = np.concatenate((zlist_big , df['impedance'].values))
    wlist_big = np.concatenate((wlist_big , df['frequency'].values.real))
    vlist_big = np.concatenate((vlist_big , df['bias voltage'].values.real))

wvlist_big = np.stack((wlist_big,vlist_big),axis = 1)

zrlist_big = zlist_big.real 
zilist_big = zlist_big.imag 
Zlist_big = np.hstack([zrlist_big, zilist_big])

popt, pcov = curve_fit(lambda w,  C_a,C_b,R_i,C_g,J_s,nA: pero_sep(w, C_a,C_b, R_i, C_g, J_s, nA) , wvlist_big, Zlist_big,p0 = init_guess)


#working!!!


#%%
#a = pero_sep(wvlist_big, C_a, C_b, R_i, C_g, J_s, nA)


# x = np.stack((wlist_big,vlist_big),axis = 1)



# #%% TRY FITTING ONLY ONE DATA SET

# def pero_sep(w, C_a, C_b, R_i, C_g, J_s, n,Vb):
#     z = pero_model(w, C_a, C_b, R_i, C_g, J_s, n, Vb)[0]
#     return np.hstack([z.real, z.imag])

# df = Vb_wzjv_list[0]
# zlist = df['impedance'].values
# wlist = df['frequency'].values
# vlist = df['bias voltage'].values
# wvlist= np.stack((wlist,vlist),axis = 1)
# zrlist = zlist.real 
# zilist = zlist.imag 
# Zlist = np.hstack([zrlist, zilist])
# popt, pcov = curve_fit(lambda w,  C_a,C_b,R_i,C_g,J_s,nA: pero_sep(w, C_a,C_b, R_i, C_g, J_s, nA,0) , wlist, Zlist,p0 = init_guess)


# #WORKING!


















#%% now trying to find the fitted parameters by using the initial guess




# from symfit import Parameter , Variable , Fit , parameters, variables, Model

# def sep_com():
    

# def pero(w, C_a, C_b, R_i, C_g, J_s, n, Vb):
#       return w+ C_a+ C_b+ R_i+ C_g+ J_s+ n+ Vb

# model = Model({
#     z1: pero_model(w1, 1,1,1,1,1,1, 0)[0]    ,                              
#     z2: pero_model(w2, 1,1,1,1,1,1,1)[0] ,
# })




#%% FIRST trying to fit 2 sets of data in the same time by using symfit
# from symfit import Parameter , Variable , Fit , parameters, variables, Model
# import sympy as sp
# # we need to fit w_z relation
# w = np.logspace(-6 , 10 , 1000)  
# zlist0 = np.hstack([np.array(Vb_wzjv_list[0]['impedance'].values.real),np.array(Vb_wzjv_list[0]['impedance'].values.imag)])
# zlist1 = np.hstack([np.array(Vb_wzjv_list[1]['impedance'].values.real), np.array(Vb_wzjv_list[1]['impedance'].values.imag)])

# C_af, C_bf, R_if, C_gf, J_sf, nf = parameters('C_af, C_bf, R_if, C_gf, J_sf, nf')
# for i in range(0,6):
#     [C_af, C_bf, R_if, C_gf, J_sf, nf][i].value = init_guess[i]

# #%%
# def pero_sep(w, C_a, C_b, R_i, C_g, J_s, n,Vb):
#     z = pero_model(w, C_a, C_b, R_i, C_g, J_s, n, Vb)[0]
#     return np.hstack([sp.re(z), sp.im(z)])

# z1 , z2 , w1, w2 = variables('z1 , z2 , w1, w2')
# model = Model({
#     z1: sp.re(pero_sep(w1, C_af, C_bf, R_if, C_gf, J_sf, nf, 0)[0])    ,                              
#     z2: sp.re(pero_sep(w2, C_af, C_bf, R_if, C_gf, J_sf, nf, Vb_wzjv_list[1]['bias voltage'][0].real)[0] ),

# })


# fit = Fit(model, w1=w, w2=w, z1=zlist0, z2=zlist1)
# fit_result = fit.execute()





# #%%
# r = pero_sep(w, C_a, C_b, R_i, C_g, J_s, nA, Vb_wzjv_list[1]['bias voltage'][0].real)[0]















# 














