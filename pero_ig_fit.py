# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:03:47 2022

@author: pokey
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


#%% defining all functions 

#For calculating impedance of a capacitor
def Zcap(c, w): #returns the impedance of a capacitor
    return 1 / (1j * w * c)


#For model of the perovskite
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
    return [C_a_e , C_b_e, C_g_e , R_i_e , nA_e ,Js_e]


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
#Vb_wzjv_list

#%% Finding the initial guess
init_guess = get_init_guess(Vb_wzjv_list)
print(init_guess)







#%% now trying to find the fitted parameters by using the initial guess











#%% FIRST trying to fit 2 sets of data in the same time by using symfit
from symfit import Parameter , Variable , Fit , parameters, variables, Model
# we need to fit w_z relation
w = np.logspace(-6 , 10 , 1000)  
zlist0 = np.array(Vb_wzjv_list[0]['impedance'].values)
zlist1 = np.array(Vb_wzjv_list[1]['impedance'].values)

C_a, C_b, R_i, C_g, J_s, n = parameters('C_a, C_b, R_i, C_g, J_s, n')
z1 , z2 , w1, w2 = variables('z1 , z2 , w1, w2')
model = Model({
    z1: pero_model(w1, C_a, C_b, R_i, C_g, J_s, n, 0)    ,                              
    z2: pero_model(w2, C_a, C_b, R_i, C_g, J_s, n, Vb_wzjv_list[1]['bias voltage'][0].real) 
})


fit = Fit(model, w1=w, w2=2, z1=zlist0, z2=zlist1)
fit_result = fit.execute()





































