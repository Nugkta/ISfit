# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 00:05:02 2022

@author: pokey
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import pandas as pd
import pandas as pd
from scipy.signal import argrelextrema

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
def pero_model(w, C_a, C_b, R_i, C_g, J_s, n, Vb): #w is the list of frequency range
    Z_d = 1 / (1 / Zcap(C_g , w) + 1 / R_i)
    Z_ion = (Zcap(C_a , w) + Zcap(C_b , w) + Z_d) #the impedance of the ionic branch
    v1 = Vb * (C_a / (C_a + C_b))
    J1 = J_s*(np.e**((v1 - 0) / (n * VT)) - np.e**((v1 - Vb) / (n * VT))) #the current densitf of the electronic branch
    #print('the power in J1 is', (v1 - Vb) / (n * VT))
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
    n1 = 0
    nhlist = [] #index of local maximum(high)
    nllist = [] #points of local minimum(low)
    seek = 'max' #will encounter a maximum first#
    while n1 < len(zlist) - 2:
        while seek == 'max':
            n1 += 1
            if -zlist[n1].imag > -zlist[n1+1].imag:
                nhlist.append(n1)
                seek = 'min'
            if n1>= len(zlist) -2:
                break
        while seek == 'min':
            if n1>= len(zlist) -2:
                #print('its bigger')
                break
            n1 += 1
            #print(n1)
            #print('zlist length', len(zlist))
            if -zlist[n1].imag < -zlist[n1+1].imag:  #note that the imaginary part should be nagative in the Nyquist plot.
                nllist.append(n1)
                seek = 'max'
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
        nhlist, nllist= find_extremum(wzlist)
        r_reci_e = wzlist[nllist[0]][1].real
        r_rec0_e = wzlist[-1][1].real
        k = r_rec0_e / r_reci_e
        klist.append(k)
    for i in range (0, len(klist)-1):
        if np.abs(klist[i+1]-klist[i]) < 0.001: #Here I used 0.001 as the threshold indicating the k value is stablised arbitrarily. Could be changed in the future
            return klist[i+1]
            break
    print('the k is not stablised for this range of bias voltage, so picked the last k')
    #################################################################################################
    #return klist[-1]
    return klist[-1]
    #return 2
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
    C_b_e = (k - 1) * C_a_e    #the estimated C_b
    C_g = C_ap[1].real       #the estiamted C_g
    print('C_a , C_b,k are',C_a_e , C_b_e ,k)
    return C_a_e , C_b_e , C_g  

#STEP 4
def find_Ri(dfs, C_a_e): 
    ############################################################################################################################
    #df = dfs[-2]#using the last dataframe to gurantee that the k is stablised
    df = dfs[-1]
    #zlist = df['impedance'].to_numpy()
    wzlist = df[['frequency','impedance']].to_numpy()
    #nhlist, nllist= find_extremum(wzlist)
    nhlist = np.array(argrelextrema(wzlist[:,1].imag, np.less))[0]
    nllist = np.array(argrelextrema(wzlist[:,1].imag, np.greater))[0]
    whlist = wzlist[nhlist][: , 0]
    #plt.plot(wzlist[:,1].real,-wzlist[:,1].imag)
    w4 = min(whlist)
    R_i_e = 1/(w4 * C_a_e).real
    print('C_a and C_a_e are', C_a,C_a_e)
    return R_i_e

#For finding the initial guess using the functions defined above
def get_init_guess(dfs):
    k = find_k(dfs) #k is obtained, Vb_wzjv_list is the list of dataframes storing the experiment data.
    print('the k is', k)
    Js_e , nA_e = find_nA_Js(dfs , k)
    C_a_e , C_b_e , C_g_e = find_Cabg(dfs , k)
    R_i_e = find_Ri(dfs,C_a_e)
    return [C_a_e , C_b_e,  R_i_e ,C_g_e ,Js_e, nA_e ]





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
    popt, pcov = curve_fit(pero_sep , wvlist_big, Zlist_big,p0 = init_guess,maxfev = 100000)   #fitting the function to the variables (wv(independent) and z(dependent)) for parameters
    return popt, pcov






#%%
#%% generating simulated sets of data
# Vblist = np.linspace(0, 1, 20)      #defining the range of the bias volatage
# w = np.logspace(-6 , 10 , 1000)     #defining the range of the frequency 
# Vb_z_list = np.empty([2 , 1000])    #initialising Vbzlist
# Vb_wzjv_list = [] #initialising the list of dataframes, each dataframe corresponds to a specific bias voltage
# for V in Vblist:
#     zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, V)
#     wzlist = np.stack((w , zlist) , axis = -1)
#     wzlist = wzlist[wzlist[:, 1].real.argsort()]     # sorting the wzlist by the real part of the impedance(preparing for the find extrema function)
#     vlist = np.ones(len(wzlist)) * V
#     jlist = np.ones(len(wzlist)) * J1
#     wz_v_jlist = np.column_stack((wzlist , vlist,jlist))
#     #v_wz_jlist = np.column_stack((v_))
#     df = pd.DataFrame(wz_v_jlist , columns=['frequency' , 'impedance' , 'bias voltage','recomb current'])
#     Vb_wzjv_list.append(df)
    


# init_guess = get_init_guess(Vb_wzjv_list)
# print('the initial guesses are', init_guess)
# popt, pcov = global_fit(Vb_wzjv_list , init_guess)
# print('the fitted parameters are',popt)
# print('the original parameters are',C_a, C_b, R_i, C_g, J_s, nA )





































#%% NOW trying to add an initial fit funtion (fitting two)



# Vblist = np.linspace(0, 1, 20)      #defining the range of the bias volatage
# w = np.logspace(-6 , 10 , 1000)     #defining the range of the frequency 
# Vb_z_list = np.empty([2 , 1000])    #initialising Vbzlist
# Vb_wzjv_list = [] #initialising the list of dataframes, each dataframe corresponds to a specific bias voltage
# for V in Vblist:
#     zlist, J1 = pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, V)
#     wzlist = np.stack((w , zlist) , axis = -1)
#     wzlist = wzlist[wzlist[:, 1].real.argsort()]     # sorting the wzlist by the real part of the impedance(preparing for the find extrema function)
#     vlist = np.ones(len(wzlist)) * V
#     jlist = np.ones(len(wzlist)) * J1
#     wz_v_jlist = np.column_stack((wzlist , vlist,jlist))
#     #v_wz_jlist = np.column_stack((v_))
#     df = pd.DataFrame(wz_v_jlist , columns=['frequency' , 'impedance' , 'bias voltage','recomb current'])
#     Vb_wzjv_list.append(df)


# df1 = Vb_wzjv_list[0]
# x = np.real(df1['impedance'])
# y = np.imag(df1['impedance'])

# plt.plot(np.real(df1['impedance'].values),-np.imag(df1['impedance']))
# plt.xlim([0,8e8])
# plt.ylim([0,8e8])
# #%%
# from scipy.signal import argrelextrema
# maxima = argrelextrema(y, np.less)

# plt.plot(x[maxima],-y[maxima],'r.')



# def circle_func(x,a,r1, mid):
#     y = np.sqrt(r1**2 - (x-a)**2)
#     return y
#     # while x < mid:

#     # while x >= mid:
#     #     y = np.sqrt(r2**2 - (x-b)**2)


        
#     # if r1**2 - (x-a)**2 <0:
#     #     y1 = 0
#     # else:
#     #     y1 = np.sqrt(r1**2 - (x-a)**2)
#     # if r2**2 - (x-b)**2 <0:
#     #     y2 = 0
#     # else:
#     #     y2 = np.sqrt(r2**2 - (x-b)**2)
#     # return y1+y2

# popt,pcov = curve_fit(lambda x,a,r1: circle_func(x,a,r1, 3.5e8), x[0:500], y[0:500],maxfev = 100000, p0 = [1.7e8,1.7e8])
# #def curve_fit()
































