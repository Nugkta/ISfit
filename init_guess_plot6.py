# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:28:02 2022

@author: pokey

In this file, the initial guess finding algorithm is written,
as well as the interactive plots and demonstration.


"""

import pero_model_fit4 as pmf
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import TextBox,Slider, Button,CheckButtons
import functools
from lmfit import report_fit
from os import listdir
from PIL import Image as PImage
import matplotlib.image as mpimg
#%%

#constants used in the model
VT = 0.026 # = kT/q
q = 1.6e-19
T = 300
kb = 1.38e-23





def get_key(val , my_dict):    #get the key from a known value in a dictionary
    for key, value in my_dict.items():
        if val == value:
            return key

class init_guess_class:       
    '''
    This is the class for storing the initial guess.
    A class is needed instead of a list, because the interactive plot need to update the initial guess in real time.
    '''
    def __init__(self): #initialising the attributes
        self.C_A = None
        self.C_ion = None
        self.R_ion = None
        self.C_g = None
        self.J_s = None
        self.nA = None
        self.R_srs = None
        self.R_shnt = 10000  #setting a large initial guess for R-shnt for the case of global fit without 0V
        
    def update_all(self, init, mode = 0): # update all the attrs in one go by inputting a list of values 
                                            #mode 1 is for global fit with 0V case.
        self.C_A = init[0]
        self.C_ion = init[1]
        self.R_ion = init[2]
        self.C_g = init[3]
        self.J_s = init[4]
        self.nA = init[5]
        self.R_srs = init[6]
        if mode == 1:
            self.R_shnt = init[7]
        
    def update_all_0V(self,init):
        self.C_ion = init[0]
        self.C_g = init[1]
        self.R_ion = init[2]
        self.J_nA = init[3]
        self.R_srs = init[4]
        self.R_shnt = init[5]
        
    def update_R_ion(self, R_ion): #update R_ion only
        self.R_ion = R_ion
        
    def values(self,mode = 0): #function for returning all the values of the initial guess
        if mode == 0:    
            return [self.C_A, 
                        self.C_ion ,
                        self.R_ion ,
                        self.C_g ,
                        self.J_s ,
                        self.nA,
                        self.R_srs,
                        self.R_shnt]
        if mode == 1: #0V individual
            return [self.C_ion ,
                        self.C_g ,
                        self.R_ion ,
                        self.J_nA,
                        self.R_srs,
                        self.R_shnt]
        
        if mode == 2: #no 0V individual
            return [self.C_A,
                self.C_ion ,
                self.R_ion ,
                        self.C_g ,
                        self.J_s,
                        self.nA,
                        self.R_srs,
                        self.R_shnt]
        
    def update_param(self,param,value): #updating a specific parameter
        setattr(self,param,value)

class fix_params():     #class that store the parameters to fix in the fit
    def __init__(self):
        self.C_A = False
        self.C_ion = False
        self.R_ion = False
        self.C_g = False
        self.J_s = False
        self.nA = False
        self.R_srs = False
        self.R_shnt = False
    def update_param(self,param,value):
        setattr(self,param,value)
    def get(self,param):
        return getattr(self, param)
    def fix_index(self):     #find the index of variables to fix
        attr_list = [self.C_A, 
                self.C_ion ,
                self.R_ion ,
                self.C_g ,
                self.J_s ,
                self.nA,
                self.R_srs,
                self.R_shnt]
        index = []
        for i in range(0,8):
            if attr_list[i] == True:
                index.append(i)
        return index #this function will return the index of the parameters that the user selected to fit.
    
def closest_node(node, nodes): #the function for finding the closest point on the line to the user selected point on the canvas
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2) 

def find_extremum(df):
    '''
    This function will plot the Nyquist plot from the dataframe of the experimental data and let the user choose the local maxima and minimum of the plot.
    And return the index of the selected points.
    '''
    #plotting the Nyquist plot
    zrlist = np.real(df['impedance'].values)
    zilist = np.imag(df['impedance'].values)
    zrilist = np.stack((zrlist , -zilist), axis = 1)
    #now letting the user to determine the rough location of the local maxima(2) and local minimum(1)
    # path = "sample_plot.png"
    # imgs = PImage.open(path)
    # imgs.show()
    
    # a = plt.figure()
    img = mpimg.imread('max_sample.png')
    # imgplot = plt.imshow(img)
    # plt.figure('The position of the minimum and the maxima')
    # plt.show()
    
    # plt.figure()
    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2,figsize =(16,7),gridspec_kw={'width_ratios': [2, 1]})
    ax1.plot(zrlist,-zilist,'.')
    ax1.set_title(' Click the maxima on this plot for the initial guesses')
    ax2.imshow(img)
    ax2.set_title('Choose the two maxima like this')
    vertices = plt.ginput(2)
    plt.close()
    
    
    img2 = mpimg.imread('min_sample.png')
    fig2, (ax3,ax4) = plt.subplots(nrows = 1, ncols = 2,figsize =(16,7),gridspec_kw={'width_ratios': [2, 1]})
    ax3.plot(zrlist,-zilist,'.')
    ax3.set_title(' click the minimum on this plot for the initial guesses')
    ax4.imshow(img2)
    ax4.set_title('Choose the minimum like this')
    mini = plt.ginput(1)
    plt.close()
    # a.close()
    # use the approximate position the user selected to find the corresponding index in the dataframe
    nhlist = []
    nllist = []
    for vert in vertices:
        num = closest_node(vert , zrilist)
        nhlist.append(num)
    minnum = closest_node(mini , zrilist)
    nllist.append(minnum)
    return nhlist,nllist


def find_point(df):
    '''
    This function will find the points that are important for the initial guess algorithm to work.
    The definition of the important point should refer to the paper.
    '''
    zlist = df['impedance'].to_numpy()
    wzlist = df[['frequency','impedance']].to_numpy()
    nhlist, nllist= find_extremum(df)
    R_n8 = wzlist[nllist[0]][1].real      #obtained the R_n8(infinity) parameter
    R_n0 = wzlist[-1][1].real             #obtained the R_n0 parameter
    whlist = np.real(wzlist[nhlist][:,0])
    w_n = min(whlist)
    w_t = wzlist[[nllist][0]][0][0].real                   #obtained the w_n parameter
    w_r = max(whlist)
    C_eff = np.real(np.imag(1/df['impedance'].values)/df['frequency'])
    # print(C_eff)
    #C_n = C_eff[-1]
    C_G = C_eff[0]
    zlist = df['impedance'].to_numpy()
    R_srs = min(np.real(zlist))

    
    return R_n8 , R_n0 , w_n , w_t , C_G ,w_r, R_srs






def find_R_ion(df):
    '''
    This is for finding R_ion using the 0V data
    '''
    #find R_ion
    zrlist = np.real(df['impedance'].values)
    zilist = np.imag(df['impedance'].values)
    zrilist = np.stack((zrlist , -zilist), axis = 1)
    img = mpimg.imread('min_sample.png')

    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2,figsize =(16,7),gridspec_kw={'width_ratios': [2, 1]})
    ax1.plot(zrlist,-zilist,'.')
    ax1.set_title(' click the minimum on this plot for the initial guesses')
    ax2.imshow(img)
    ax2.set_title('Choose the minimum like this')
    mini = plt.ginput(1)
    plt.close()
    R_ion = mini[0][0]
    return R_ion





def init_guess_find(df ,crit_points, V0 = False,df_0V = None):
    '''
    Then, by implementing the algorithm using the original data, user guessed R_ion and the obtained critical points, the initial guess
    can be obtained corresponds to the method written in the paper.
    '''
    
    
    if V0 is False: 
        R_ion = float(input(' input your guess of R_ion: '))
    if V0 is True:
        R_ion = find_R_ion(df_0V)
    R_n8 , R_n0 , w_n , w_t , C_G ,w_r, R_srs = crit_points
    J_n = np.real(df['recomb current'][0]) 
    k = R_n0 / R_n8
    nA = J_n *R_n8 *q / ( kb *T)
    C_ion = 1 / (w_n * k * R_ion)
    C_g = C_G
    C_A = C_ion / (1 - 1/k)
    C_B = 1 / (1/C_ion - 1/C_A)
    V = np.real(df['bias voltage'][0])
    J_s = J_n / np.e**((V*(1 - C_ion/C_A)*q) / (nA * kb * T))
    return C_A, C_ion, R_ion, C_g, J_s, nA, R_srs
        
def init_guess_find_0V(df):
    '''
    This function is for finding the critical point for initial guess finding for the 0V bias voltage scenerio
    '''
    #find R_srs
    zlist = df['impedance'].to_numpy()
    R_srs = abs(min(np.real(zlist)))
    
    #find R_ion
    zrlist = np.real(df['impedance'].values)
    zilist = np.imag(df['impedance'].values)
    zrilist = np.stack((zrlist , -zilist), axis = 1)
    img = mpimg.imread('min_sample_0V.png')

    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2,figsize =(16,7),gridspec_kw={'width_ratios': [2, 1]})
    ax1.plot(zrlist,-zilist,'.')
    ax1.set_title(' click the minimum on this plot for the initial guesses')
    ax2.imshow(img)
    ax2.set_title('Choose the minimum like this')
    mini = plt.ginput(1)
    plt.close()
    R_ion = mini[0][0]
    
    #Find C_g
    img = mpimg.imread('max_sample_0V.png')

    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2,figsize =(16,7),gridspec_kw={'width_ratios': [2, 1]})
    ax1.plot(zrlist,-zilist,'.')
    ax1.set_title(' click the first maximum on this plot for the initial guesses')
    ax2.imshow(img)
    ax2.set_title('Choose the first maximum like this')
    maxi = plt.ginput(1)
    plt.close()
    max_num = closest_node(maxi , zrilist)
    wzlist = df[['frequency','impedance']].to_numpy()
    w_g = np.real(wzlist[max_num][0])
    C_g = 1/(R_ion * w_g)
    
    
    #Find J_nA and R_shnt by R_n
    img = mpimg.imread('end_sample_0V.png')
    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2,figsize =(16,7),gridspec_kw={'width_ratios': [2, 1]})
    ax1.plot(zrlist,-zilist,'.')
    ax1.set_xlim(0,2.8*max(zrlist))
    ax1.set_title(' click the approximate end of this plot')
    ax2.imshow(img)
    ax2.set_title('Choose the first maximum like this')
    end = plt.ginput(1)
    plt.close()
    R_n = end[0][0]
    J_nA = kb*T / (R_n * q)
    R_shnt = R_n
    
    #Find C_ion by C_eff plot
    C_eff = np.real(np.imag(1/df['impedance'].values)/df['frequency'])
    C_ion = C_eff[-1]
    
    return C_ion, C_g, R_ion, J_nA, R_srs, R_shnt 




def init_guess_slider(df, points, R_ion):       
    '''
    The function for the slider of R_ion (known points and R_ion), different from the previous one because
    here the R_ion is define by the slider(as function input), not by user input directly.
    '''
    R_n8 , R_n0 , w_n , w_t , C_G ,w_r, R_srs = points
    J_n = np.real(df['recomb current'][0]) 
    k = R_n0 / R_n8
    nA = J_n *R_n8 *q / ( kb *T)
    # print(n,J_n ,R_n8 ,q , kb ,T)
    C_ion = 1 / (w_n * k * R_ion)
    C_g = C_G
    C_A = C_ion / (1 - 1/k)
    C_B = 1 / (1/C_ion - 1/C_A)
    V = np.real(df['bias voltage'][0])
    J_s = J_n / np.e**((V*(1 - C_ion/C_A)*q) / (nA * kb * T)) 
    return C_A, C_ion, R_ion, C_g, J_s, nA, R_srs


def get_init_guess(df): #put together crit points and init guess
    crit_points = find_point(df)
    return init_guess_find(df ,crit_points)





def find_k(dfs): #function for finding an appropriate k (ratio of C_a/(C_a +C_b) from a list of dataframes just like above
    df = dfs[-1]#because the last one has the stabliest k
    zlist = df['impedance'].to_numpy()
    wzlist = df[['frequency','impedance']].to_numpy()
    nhlist, nllist= find_extremum(df)
    r_reci_e = wzlist[nllist[0]][1].real
    r_rec0_e = wzlist[-1][1].real
    k = r_rec0_e / r_reci_e
    return k


def find_nA_Js(dfs,k, mode): #mode 0 is for global with 0V, mode 1 is for global without 0V
    # k = find_k(dfs)
    jlist = []
    vlist = []  
    for wzjvdf in dfs:
        vlist.append(wzjvdf['bias voltage'].values[-1]) #-1 because the element at -1 corresponds to lowest frequency ~steady state
        jlist.append(wzjvdf['recomb current'].values[-1])
    if mode == 0:
        log_jlist =np.log( np.array(jlist)[1:].real) #[1:]because the first element gives log0
        vlist = np.array(vlist).real[1:]
    if mode == 1:
        log_jlist =np.log( np.array(jlist).real) #[1:]because the first element gives log0
        vlist = np.array(vlist).real
    grad,b = np.polyfit(log_jlist , vlist ,1)
    log_Js = -b/grad
    Js_e = np.exp(log_Js)
    nA_e = 1/VT * grad/k
    return  Js_e,nA_e



#%% PLOTTING





wlist = np.logspace(-3, 6 ,10000)




def R_ion_Slider(init_guess, dfs,crit_points, mode = 0):
    #print('vvvvvvvvvvvvvvvvvvvvvvvv ',mode)
    df = dfs[-1]
    v = df['bias voltage'][0]
    simu_Z, simu_J1 = pmf.pero_model_ind(wlist,*init_guess.values(),v)
    fig , ((ax1 ,ax2),(ax3,ax4)) = plt.subplots(2 , 2,figsize = (16,7)) #opening the canvas for the plot of Nyquist


    #Nyquist plot
    ax3 = plt.subplot(212)
    line1, = ax3.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'x', ms=4,label = 'experiment data')
    line2, = ax3.plot(np.real(simu_Z),-np.imag(simu_Z),'r--', label = 'initial guess')
    ax3.legend()
    ax3.set_xlabel('Z\'')
    ax3.set_ylabel('Z\'\'')



    #real part plot
    line_zr, = ax1.plot(df['frequency'],np.real(df['impedance'].values),'b.',ms = 4, label = 'experiment Z\' ')
    line_zr_ig, = ax1.plot(wlist,np.real(simu_Z),'c--', label = ' initial guess Z\'')
    ax1.set_xscale('log')
    ax1.set_ylabel('Z\'')
    ax1.set_xlabel(r'frequency $\omega$')
    ax1.set_title('real Z, effective capacitance vs. frequency')
    ax1.legend(loc = 3)
    ax1.spines['left'].set_color('c')
    ax1.tick_params(axis='y', colors='c')

    #effective ccapacitance

    ax_eff = ax1.twinx()
    C_eff = np.imag(1 / df['impedance']) / df['frequency']
    C_eff_ig = np.imag(1 / simu_Z) / wlist
    line_Ceff, = ax_eff.plot(df['frequency'],C_eff,'.',ms = 4,color = 'peru', label = 'experiment effective capacitance')
    line_Ceff_ig, = ax_eff.plot(wlist , C_eff_ig,'--',ms = 4,color = 'orange', label = 'initial guess effective capacitance')
    ax_eff.set_yscale('log')
    ax_eff.set_xscale('log')
    ax_eff.set_ylabel(r'Im($Z^{-1}$)$\omega^{-1}$')
    ax_eff.legend()
    ax_eff.spines['right'].set_color('orange')
    ax_eff.tick_params(axis='y', colors='orange')


    #abs Z part plot
    line_absz, = ax2.plot(df['frequency'],np.abs(df['impedance'].values),'b.',ms = 4, label = 'experiment |Z| ')
    line_absz_ig, = ax2.plot(wlist,np.abs(simu_Z),'c--', label = ' initial guess |Z|')
    ax2.set_xscale('log')
    ax2.set_ylabel('|Z|')
    ax2.set_xlabel(r'frequency $\omega$')
    ax2.set_title(r'|Z|, $\theta$ (rad) vs. frequency')
    ax2.legend(loc = 3)
    ax2.spines['left'].set_color('c')
    ax2.tick_params(axis='y', colors='c')

    #theta plot

    ax_t = ax2.twinx()
    line_t, = ax_t.plot(df['frequency'],np.angle(df['impedance'].values),'.',ms = 4,color = 'peru', label = r'experiment $\theta$ ')
    line_t_ig, = ax_t.plot(wlist , np.angle(simu_Z),'--',ms = 4,color = 'orange', label = r'initial guess $\theta$')
    ax_t.set_xscale('log')
    ax_t.set_ylabel(r'$\theta$ (rad)')
    ax_t.legend()
    ax_t.spines['right'].set_color('orange')
    ax_t.tick_params(axis='y', colors='orange')



    #Now adding the slider function

    plt.subplots_adjust(left=0.15, bottom=.2)          #adjusting the position of the main plot to leave room for then

    ax_r = plt.axes([0.25, 0.07, 0.55, 0.03])

    R_slider = Slider(                              #the slider for the effect of the change of R_ion
        ax = ax_r, 
        label = 'R_ion',
        #setting the range of the slider, log because we need to investigate in range of orders of magnitude
        valmin = np.log(0.02 * init_guess.values()[2]),
        valmax = np.log(50 * init_guess.values()[2]),
        valinit = np.log(init_guess.values()[2]),
        )

    R_slider.valtext.set_text('%.2e'%init_guess.values()[2])


    def update_R_ion(val,points):

        R_ion = np.exp(R_slider.val)
        iglist = init_guess_slider(df,points,R_ion)
        init_guess.update_all(iglist)
        simu_Z, simu_J1 = pmf.pero_model_ind(wlist,*init_guess.values(),v)
        #first plot
        line2.set_ydata(-np.imag(simu_Z))
        line2.set_xdata(np.real(simu_Z))
        #second subplot
        line_Ceff_ig.set_ydata(np.imag(1 / simu_Z) / wlist)
        line_zr_ig.set_ydata(np.real(simu_Z))
        # #third subplot
        line_absz_ig.set_ydata(np.abs(simu_Z))
        line_t_ig.set_ydata(np.angle(simu_Z))
        #revere the log scale in slider to normal scale
        amp = np.exp(R_slider.val)
        R_slider.valtext.set_text('%.2e'%amp)
        fig.canvas.draw_idle()
        


    # R_ion = Updated()
    # def on_change(v):
    #     val = np.exp(R_slider.val)
    #     R_ion.update(val)


    R_slider.on_changed(lambda val: update_R_ion(val , crit_points))
    # R_slider.on_changed(on_change)
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button_R = Button(resetax, 'Reset', hovercolor='0.975')



    def reset_R_ion(event):
        R_slider.reset()
        fig.canvas.draw_idle()
        
        
    button_R.on_clicked(reset_R_ion)

    def submit(text,points):
        R_ion = float(text)
        iglist = init_guess_slider(df,points,R_ion)
        init_guess.update_all(iglist)
        simu_Z, simu_J1 = pmf.pero_model_ind(wlist,*init_guess.values(),v)
        #first plot
        line2.set_ydata(-np.imag(simu_Z))
        line2.set_xdata(np.real(simu_Z))
        #second subplot
        line_Ceff_ig.set_ydata(np.imag(1 / simu_Z) / wlist)
        line_zr_ig.set_ydata(np.real(simu_Z))
        # #third subplot
        line_absz_ig.set_ydata(np.abs(simu_Z))
        line_t_ig.set_ydata(np.angle(simu_Z))
        #revere the log scale in slider to normal scale
        R_slider.set_val(np.log(R_ion))
        amp = np.exp(R_slider.val)
        R_slider.valtext.set_text('%.2e'%amp)
        fig.canvas.draw_idle()
        
        

    initial_text = ""
    axbox = plt.axes([0.1, 0.07, 0.04, 0.03])
    text_box = TextBox(axbox, 'Set R_ion : ', initial=initial_text)
    text_box.on_submit(lambda text: submit(text, crit_points))
    
    # the button to proceed to the next step    
    ax_next = plt.axes([0.8, 0.925, 0.1, 0.04])    #axis of the next step pattern
    button_next = Button(ax_next , 'Next step', hovercolor='0.975')
    button_next.on_clicked(lambda event:all_param_sliders(event,init_guess, dfs, crit_points,mode))
    
    
    resetax._button = button_R
    axbox._text_box = text_box
    ax_next._button_next = button_next
    
    plt.show()

#%%

def all_param_sliders(event,init_guess, dfs, crit_points,mode = 0,refit = 0,popt = None):
    
    plt.close()

    df = dfs[-1]
    v = df['bias voltage'][0]
    if refit == 0:
        simu_Z, simu_J1 = pmf.pero_model_ind(wlist,*init_guess.values(),v)
    if refit == 1:
        if mode == 0:
            vlist = np.ones(len(wlist)) * v
            wvlist = np.stack((wlist,vlist),axis = 1) 
            new_init = popt
            print(len(popt),2222222222222222222)
            simu_Z, simu_J1 = pmf.pero_model(wvlist,*popt)
        if mode == 2:
            simu_Z, simu_J1 = pmf.pero_model_ind(wlist,*popt,v)
        
    fig, ((ax1 ,ax2),(ax3,ax4)) = plt.subplots(figsize=(16,7),ncols = 2 , nrows = 2)



    #Nyquist plot 
    ax_nyq = plt.subplot(212)
    line1, = ax_nyq.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'x', ms=4,label = 'experiment data')
    line2, = ax_nyq.plot(np.real(simu_Z),-np.imag(simu_Z),'r--',label = 'initial guess')
    ax_nyq.legend()
    ax_nyq.set_xlabel('Z\'')
    ax_nyq.set_ylabel('Z\'\'')
    plt.subplots_adjust(left=0.1, bottom=.26)


    #Z_real plot
    line_zr, = ax1.plot(df['frequency'],np.real(df['impedance'].values),'b.',ms = 4, label = 'experiment Z\' ')
    line_zr_ig, = ax1.plot(wlist,np.real(simu_Z),'c--', label = ' initial guess Z\'')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Z\'')
    ax1.set_xlabel(r'frequency $\omega$')
    ax1.set_title('real Z, effective capacitance vs. frequency')
    ax1.legend(loc = 3, fontsize = 'small')
    ax1.spines['left'].set_color('c')
    ax1.tick_params(axis='y', colors='c')

    #effective ccapacitance

    ax_eff = ax1.twinx()
    C_eff = np.imag(1 / df['impedance']) / df['frequency']
    C_eff_ig = np.imag(1 / simu_Z) / wlist
    line_Ceff, = ax_eff.plot(df['frequency'],C_eff,'.',ms = 4,color = 'peru', label = 'experiment effective capacitance')
    line_Ceff_ig, = ax_eff.plot(wlist , C_eff_ig,'--',ms = 4,color = 'orange', label = 'initial guess effective capacitance')
    ax_eff.set_yscale('log')
    ax_eff.set_xscale('log')
    ax_eff.set_ylabel(r'Im($Z^{-1}$)$\omega^{-1}$')
    ax_eff.legend(loc = 1, fontsize = 'small')
    ax_eff.spines['right'].set_color('orange')
    ax_eff.tick_params(axis='y', colors='orange')


    #abs Z part plot
    line_absz, = ax2.plot(df['frequency'],np.abs(df['impedance'].values),'b.',ms = 4, label = 'experiment |Z| ')
    line_absz_ig, = ax2.plot(wlist,np.abs(simu_Z),'c--', label = ' initial guess |Z|')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel('|Z|')
    ax2.set_xlabel(r'frequency $\omega$')
    ax2.set_title(r'|Z|, $\theta$ (rad) vs. frequency')
    ax2.legend(loc = 3, fontsize = 'small')
    ax2.spines['left'].set_color('c')
    ax2.tick_params(axis='y', colors='c')

    #theta plot

    ax_t = ax2.twinx()
    line_t, = ax_t.plot(df['frequency'],np.angle(df['impedance'].values),'.',ms = 4,color = 'peru', label = r'experiment $\theta$ ')
    line_t_ig, = ax_t.plot(wlist , np.angle(simu_Z),'--',ms = 4,color = 'orange', label = r'initial guess $\theta$')
    ax_t.set_xscale('log')
    ax_t.set_ylabel(r'$\theta$ (rad)')
    ax_t.legend(loc = 1, fontsize = 'small')
    ax_t.spines['right'].set_color('orange')
    ax_t.tick_params(axis='y', colors='orange')

    #change only the C_a in popt now as a test
    # axC_a = plt.axes([0.25, 0.5, 0.65, 0.03])
    ax_list = {} 
    ax_list_t = {} #stores axis postion for the textbox
    sliders = {}
    textboxs = {}
    param_name = ['C_A', 'C_ion', 'R_ion', 'C_g', 'J_s', 'nA' ,'R_srs']
    param_dict ={'C_A':0, 'C_ion':1, 'R_ion':2, 'C_g':3, 'J_s':4, 'nA':5,'R_srs':6}    #establish the correspondance between the order and the name of the parameters
    range_list = [(1/3 * init_guess.C_A, 3 * init_guess.C_A ),
                  (1/3 * init_guess.C_ion, 3 * init_guess.C_ion),
                  (1/10 * init_guess.R_ion, 10 * init_guess.R_ion),
                  (1/10 * init_guess.C_g, 10 * init_guess.C_g),
                  (1/2 * init_guess.J_s, 3 * init_guess.J_s),
                  (1/1.5 * init_guess.nA, 1.5 * init_guess.nA),
                  (.01 * init_guess.R_srs, 100 * init_guess.R_srs)
                  ]



    for i in range(0,7):
        ax_list[i] = plt.axes([0.25, 0.03 * (i+2)-0.02, 0.5, 0.02]) #position list for 
        ax_list_t[i] = plt.axes([0.1, 0.03 * (i+2)-0.02, 0.03, 0.02])    #position list for the textbox
        sliders[i] = Slider(
            ax = ax_list[i], 
            label = 'the value of ' + param_name[i],
            valmin = range_list[i][0],
            valmax = range_list[i][1],
            valinit = init_guess.values()[i],
            )
        textboxs[i] = TextBox(ax_list_t[i], 
                              'Set '+ param_name[i]+' : ',
                              initial='')

    sl_val_list =[]

    for key in sliders:
        sl_val_list.append(sliders[key])

    def update(val,  ):            #function called when the value of slider is updated
        vals = [i.val for i in sl_val_list]
        init_guess.update_all(vals)
        simu_Z , j = pmf.pero_model_ind(wlist,*vals,init_guess.R_shnt,v)
        line2.set_ydata(-np.imag(simu_Z))
        line2.set_xdata(np.real(simu_Z))
        #second subplot
        line_Ceff_ig.set_ydata(np.imag(1 / simu_Z) / wlist)
        line_zr_ig.set_ydata(np.real(simu_Z))
        # #third subplot
        line_absz_ig.set_ydata(np.abs(simu_Z))
        line_t_ig.set_ydata(np.angle(simu_Z))
        #revere the log scale in slider to normal scale
        fig.canvas.draw_idle()
        

    def submit_2(text,points,param,init_guess):   #param shows which parameter is updated
        new_value = float(text)                 #convert the string input in textbox to float
        init_guess.update_param(param,new_value)   #only update the value of which the textbox is updated
        simu_Z , j = pmf.pero_model_ind(wlist,*init_guess.values(),v)
        #first plot
        line2.set_ydata(-np.imag(simu_Z))
        line2.set_xdata(np.real(simu_Z))
        #second subplot
        line_Ceff_ig.set_ydata(np.imag(1 / simu_Z) / wlist)
        line_zr_ig.set_ydata(np.real(simu_Z))
        # #third subplot
        line_absz_ig.set_ydata(np.abs(simu_Z))
        line_t_ig.set_ydata(np.angle(simu_Z))
        #revere the log scale in slider to normal scale
        sliders[param_dict[param]].set_val(new_value)
        fig.canvas.draw_idle()
        
       
    # for key in textboxs:  
        # print(get_key(key,param_dict ))
    textboxs[0].on_submit(lambda text: submit_2(text, crit_points,'C_A',init_guess))
    textboxs[1].on_submit(lambda text: submit_2(text, crit_points,'C_ion',init_guess))
    textboxs[2].on_submit(lambda text: submit_2(text, crit_points,'R_ion',init_guess))
    textboxs[3].on_submit(lambda text: submit_2(text, crit_points,'C_g',init_guess))
    textboxs[4].on_submit(lambda text: submit_2(text, crit_points,'J_s',init_guess))
    textboxs[5].on_submit(lambda text: submit_2(text, crit_points,'nA',init_guess))
    textboxs[6].on_submit(lambda text: submit_2(text, crit_points,'R_srs',init_guess))
    for key in sliders:
        sliders[key].on_changed(lambda val: update(val))
    resetax = plt.axes([0.8, 0.9, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    def reset(event):
        for key in sliders:
            sliders[key].reset()
    button.on_clicked(reset)
    plt.show()

    #define bottons to fix the initial guess while doing curve_fit
    rax = plt.axes([0.85, 0.03, 0.08, 0.2])
    lines = [line1, line_zr,line_Ceff,line_absz,line_t]
    labels = param_name
    check = CheckButtons(rax, labels)
    param_to_fix = fix_params() # this stores the information of what variables to fix in the global fit

    def fix_param(label):
        index = labels.index(label)
        param_to_fix.update_param(label, not param_to_fix.get(label))

    check.on_clicked(fix_param)


    # Add a textbox as the title of the checkbottons
    text_ax = plt.axes([0.95, 0.24, 0.0, 0.0])
    label = 'Fix parameter'
    textbox = TextBox(text_ax,label)
    

    # a = plt.axes([0.1, 0.03 * (2)-0.02, 0.03, 0.02])
    # a._textbox = textboxs[0]
    ax_list_t[0]._textbox = textboxs[0]
    ax_list_t[1]._textbox = textboxs[1]
    ax_list_t[2]._textbox = textboxs[2]
    ax_list_t[3]._textbox = textboxs[3]
    ax_list_t[4]._textbox = textboxs[4]
    ax_list_t[5]._textbox = textboxs[5]
    ax_list_t[6]._textbox = textboxs[6]
    # the button to proceed to the next step    
    ax_next = plt.axes([0.8, 0.95, 0.1, 0.02])    #axis of the next step pattern
    button_next = Button(ax_next , 'Start Fitting', hovercolor='0.975')
    button_next.on_clicked(lambda event:fit_plot_comp_plots(event,param_to_fix,dfs,init_guess ,crit_points,mode))
    
    ax_next._button_next = button_next
    resetax._button = button
    rax._checkbox = check
    fig.canvas.draw_idle()







#%%
# def plot_comp_plots(event, param_to_fix,dfs,init_guess,v):
#     plt.close()
#     fix_index =  param_to_fix.fix_index()
#     #the fit it done below
#     result = pmf.global_fit(dfs , init_guess.values() , fix_index)
#     report_fit(result)
#     result_dict = result.params.valuesdict()
#     #putting the resultant parameters into the popt list
#     popt = []
#     for key in result_dict:
#         popt.append( result_dict[key])
#     z , j = pmf.pero_model(wlist,*popt,v)
#     z_ig, j_ig = pmf.pero_model(wlist,*init_guess.values(),v)

#     fig, ((ax1 ,ax2),(ax3,ax4)) = plt.subplots(figsize=(18, 12),ncols = 2 , nrows = 2)
#     fig.suptitle('Comparison between the initial guess and the fitted parameters', fontsize = 16)


def fit_plot_comp_plots(event,param_to_fix,dfs,init_guess ,crit_points,mode = 0):
    '''
    doing the fitting and plot the comparison plot with the original data and the init_guess 
    
    '''
    print('igigigigigigigigigigigigigigigigig',init_guess.values())
    plt.close()
    fix_index =  param_to_fix.fix_index()
    #the fit it done below
    print(type(dfs),type(init_guess),type(fix_index),mode)
    result = pmf.global_fit(dfs , init_guess , fix_index,mode)
    #print(type(result))
    report_fit(result)
    result_dict = result.params.valuesdict()
    #putting the resultant parameters into the popt list
    popt = []
    for key in result_dict:
        popt.append( result_dict[key])
    plot_comp(popt , init_guess, dfs,crit_points, mode)
    
    
    
    
    # for key in result_dict:
    #     popt.append( result_dict[key])
        
    # zlist_big = np.array([])   
    # wlist_big = np.array([])
    # vlist_big = np.array([])
    # for df in dfs:
    #     zlist_big = np.concatenate((zlist_big , df['impedance'].values))
    #     wlist_big = np.concatenate((wlist_big , df['frequency'].values.real))
    #     vlist_big = np.concatenate((vlist_big , df['bias voltage'].values.real))
    # wvlist_big = np.stack((wlist_big,vlist_big),axis = 1)    

    
    # z , j = pmf.pero_model(wvlist_big,*popt)
    # z_ig, j_ig = pmf.pero_model(wvlist_big,*init_guess.values(),1,1)
    
    
    # plt.figure()
    
    # plt.plot(np.real(zlist_big), -np.imag(zlist_big),'rx',ms = 4,label = 'original data')
    
    # wlist = np.logspace(-6,6,1000)
    # v_set = set(vlist_big)
    # for i in v_set:
    #     #first plot the fitted parameters
    #     vlist = np.ones(len(wlist)) * i
    #     wvlist = np.stack((wlist,vlist),axis = 1)  
    #     z_fit, j_fit = pmf.pero_model(wvlist,*popt)
    #     plt.plot(np.real(z_fit),-np.imag(z_fit),'-' , color = 'deepskyblue',label ='fitted parameters')
    #     #then plot the initial guess parameters
    #     z_ig, j_ig = pmf.pero_model(wvlist,*init_guess.values(),1,1)
    #     plt.plot(np.real(z_ig),-np.imag(z_ig),'--',color = 'darkorange',label = 'initial guess')
        
        
        
    
    # plt.legend(['original data','fitted parameters' ,'initial guess'])
    # plt.title('The Nyquist plot for comparison')
    # plt.xlabel('Z\'')
    # plt.ylabel('Z\'\'')
    
    # plt.figure()
    # plt.plot(wlist_big,np.real(zlist_big),'bx',ms = 4, label = 'actual data Z\' ')  
    # plt.xscale('log')


def plot_comp(popt , init_guess, dfs, crit_points, mode = 0 ):
    zlist_big = np.array([])   
    wlist_big = np.array([])
    vlist_big = np.array([])
    for df in dfs:
        zlist_big = np.concatenate((zlist_big , df['impedance'].values))
        wlist_big = np.concatenate((wlist_big , df['frequency'].values.real))
        vlist_big = np.concatenate((vlist_big , df['bias voltage'].values.real))
    wvlist_big = np.stack((wlist_big,vlist_big),axis = 1)    
    z_ex = zlist_big #experimental z
    
    
    
    fig, ((ax1 ,ax2),(ax3,ax4)) = plt.subplots(figsize=(16,7),ncols = 2 , nrows = 2)
    fig.suptitle('Comparison between the initial guess and the fitted parameters,ttttt', fontsize = 16)

    #The Nyquist plot
    ax_nyq = plt.subplot(212)
    line1, = ax_nyq.plot(np.real(z_ex), -np.imag(z_ex),'x', ms=4 , label = 'actual data')   #actual data line
    # line2, = ax_nyq.plot(np.real(z),-np.imag(z),'m-', label = 'fitted') #fitted parameter line
    # line3, =ax_nyq.plot(np.real(z_ig),-np.imag(z_ig),'b--', label = 'initial guess') #initial guess line
    #ax_nyq.legend()
    ax_nyq.set_xlabel('Z\'')
    ax_nyq.set_ylabel('Z\'\'')

    #Z_real plot
    line_zr, = ax1.plot(wlist_big,np.real(z_ex),'bx',ms = 4, label = 'actual data Z\' ')
    # line_zr_ig, = ax1.plot(wlist,np.real(z_ig),linestyle = 'dotted',color = 'c', label = ' initial guess Z\'')
    # line_zr_fit, = ax1.plot(wlist,np.real(z),linestyle = 'solid',color = 'm', label = ' fitted Z\'')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Z\'')
    ax1.set_xlabel(r'frequency $\omega$')
    ax1.set_title('Real Z, effective capacitance vs. frequency')
    ax1.legend(loc = 3, fontsize = 'small')
    ax1.spines['left'].set_color('c')
    ax1.tick_params(axis='y', colors='c')

    #effective ccapacitance

    ax_eff = ax1.twinx()
    C_eff = np.imag(1 / z_ex) / wlist_big


    line_Ceff, = ax_eff.plot(wlist_big,C_eff,'x',ms = 4,color = 'peru', label = 'experiment effective capacitance')
    # line_Ceff_ig, = ax_eff.plot(wlist , C_eff_ig,linestyle = 'dotted',ms = 4,color = 'orange', label = 'initial guess effective capacitance')
    # line_Ceff_fit, = ax_eff.plot(wlist , C_eff_fit,linestyle = 'solid',ms = 4,color = 'y', label = 'fitted effective capacitance')
    ax_eff.set_yscale('log')
    ax_eff.set_xscale('log')
    ax_eff.set_ylabel(r'Im($Z^{-1}$)$\omega^{-1}$')
    ax_eff.legend(loc = 1, fontsize = 'small')
    ax_eff.spines['right'].set_color('orange')
    ax_eff.tick_params(axis='y', colors='orange')


    #abs Z part plot
    line_absz, = ax2.plot(wlist_big,np.abs(z_ex),'bx',ms = 4, label = 'experiment |Z| ')
    # line_absz_ig, = ax2.plot(wlist,np.abs(z_ig),linestyle = 'dotted',color = 'c', label = ' initial guess |Z|')
    # line_absz_ig, = ax2.plot(wlist,np.abs(z),linestyle = 'solid',color = 'm', label = ' fitted |Z|')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel('|Z|')
    ax2.set_xlabel(r'frequency $\omega$')
    ax2.set_title(r'|Z|, $\theta$ (rad) vs. frequency')
    ax2.legend(loc = 3, fontsize = 'small')
    ax2.spines['left'].set_color('c')
    ax2.tick_params(axis='y', colors='c')

    #theta plot

    ax_t = ax2.twinx()
    line_t, = ax_t.plot(wlist_big,np.angle(z_ex),'x',ms = 4,color = 'peru', label = r'experiment $\theta$ ')
    # line_t_ig, = ax_t.plot(wlist , np.angle(z_ig),linestyle = 'dotted',ms = 4,color = 'orange', label = r'initial guess $\theta$')
    # line_t_ig, = ax_t.plot(wlist , np.angle(z),linestyle = 'solid',ms = 4,color = 'y', label = r'fitted $\theta$ (rad)')
    ax_t.set_xscale('log')
    ax_t.set_ylabel(r'$\theta$ (rad)')
    ax_t.legend(loc = 1, fontsize = 'small')
    ax_t.spines['right'].set_color('orange')
    ax_t.tick_params(axis='y', colors='orange')
    
    print('tetetetettttetettttetetetttet, reached')
    #doing a loop back, using fitted values as the intial guess
    ax_next = plt.axes([0.8, 0.95, 0.1, 0.02])    #axis of the next step pattern
    button_next = Button(ax_next , 'Fit again', hovercolor='0.975')
    button_next.on_clicked(lambda event:all_param_sliders(event,init_guess, dfs, crit_points,mode , refit = 1, popt = popt))
    ax_next._button_next = button_next
    
    
    v_set = set(vlist_big)
    for i in v_set:
        #first plot the fitted parameters
        vlist = np.ones(len(wlist)) * i
        wvlist = np.stack((wlist,vlist),axis = 1)  

        
        if mode ==0: #global no 0V
            
            z_fit, j_fit = pmf.pero_model(wvlist,*popt)
            C_A_0, C_ion_0, R_i, C_g, J_s, nA,  R_srs, R_shnt = init_guess.values()
            z_ig, j_ig = pmf.pero_model(wvlist,C_A_0, C_ion_0, R_i, C_g, J_s, nA, 1, R_srs, R_shnt)
            
            # C_A_0_e, C_ion_0_e, R_i_e, C_g_e, J_s_e, nA_e,  R_srs_e, R_shnt_e = popt
            # popt_t = popt
            # popt_t.pop(6)
            # init_guess.update_all(popt_t, mode = 1)
            
        if mode == 1: #0V (individual)
            z_fit = pmf.pero_model_0V(wlist,*popt)
            z_ig= pmf.pero_model_0V(wlist,*init_guess.values(mode = 1))
            # popt_t = popt
            # popt_t.pop(6)
            # init_guess.update_all(popt_t, mode = 1)
            
            
        if mode == 2:#ind without 0V
            z_fit, j_fit = pmf.pero_model_ind(wlist,*popt,vlist_big[0])
            z_ig, j_ig= pmf.pero_model_ind(wlist,*init_guess.values(mode = 2),vlist_big[0])    
            
            # popt_t = popt
            # popt_t.pop(6)
            # init_guess.update_all(popt_t, mode = 1)
        print(len(popt),11111111111)   
        line3, =ax_nyq.plot(np.real(z_ig),-np.imag(z_ig),'b--', label = 'initial guess') #initial guess line
        line2, = ax_nyq.plot(np.real(z_fit),-np.imag(z_fit),'m-', label = 'fitted') #fitted parameter line
        
        
        line_zr_ig, = ax1.plot(wlist,np.real(z_ig),linestyle = 'dotted',color = 'c', label = ' initial guess Z\'')
        line_zr_fit, = ax1.plot(wlist,np.real(z_fit),linestyle = 'solid',color = 'm', label = ' fitted Z\'')

        C_eff_ig = np.imag(1 / z_ig) / wlist
        C_eff_fit = np.imag(1 / z_fit) / wlist
        
        line_Ceff_ig, = ax_eff.plot(wlist , C_eff_ig,linestyle = 'dotted',ms = 4,color = 'orange', label = 'initial guess effective capacitance')
        line_Ceff_fit, = ax_eff.plot(wlist , C_eff_fit,linestyle = 'solid',ms = 4,color = 'y', label = 'fitted effective capacitance')
        
        line_absz_ig, = ax2.plot(wlist,np.abs(z_ig),linestyle = 'dotted',color = 'c', label = ' initial guess |Z|')
        line_absz_ig, = ax2.plot(wlist,np.abs(z_fit),linestyle = 'solid',color = 'm', label = ' fitted |Z|')
        
        line_t_ig, = ax_t.plot(wlist , np.angle(z_ig),linestyle = 'dotted',ms = 4,color = 'orange', label = r'initial guess $\theta$')
        line_t_ig, = ax_t.plot(wlist , np.angle(z_fit),linestyle = 'solid',ms = 4,color = 'y', label = r'fitted $\theta$ (rad)')

    ax_nyq.legend(['experiemental','initial guess','fitted'])
    ax1.legend(['experiemental','initial guess','fitted'],loc = 3, fontsize = 'small')
    ax_eff.legend(['experiemental','initial guess','fitted'],loc = 1, fontsize = 'small')
    ax2.legend(['experiemental','initial guess','fitted'],loc = 3, fontsize = 'small')
    ax_t.legend(['experiemental','initial guess','fitted'],loc = 1, fontsize = 'small')
    
                
    popt_t = popt.copy()
    popt_t.pop(6)
    print(len(popt),33333333333333333333333333)
    print(mode,'mmmmmmmmmmmmmmm')
    if mode == 0:
        init_guess.update_all(popt_t, mode = 1)   
    if mode == 2:
        init_guess.update_all(popt, mode = 1)   




def __main__(dfs, mode = 0):
    df = dfs[-1]#only uses the last plot to find the initial guess (becasue it has the stable shape)
    crit_points = find_point(dfs[-1]) 
    
    k = crit_points[1] / crit_points[0]
    nA_e , J_s_e = find_nA_Js(dfs, k, mode = 1)
    print('A different method(different from the built-in method in the following steps) gives estimation of nA and J_s to  %.3e %.3e'%( J_s_e,nA_e))
    ig = init_guess_find(df,crit_points) 
    
    init_guess = init_guess_class()
    init_guess.update_all(ig)
    R_ion_Slider(init_guess, dfs,crit_points, mode)


# def __main__(dfs):
#     df = dfs[-1] #Use only the last one to find initial guess, because its shape is the stablest
#     crit_points = find_point(dfs[-1]) 
#     ig = init_guess_find(df,crit_points)
#     init_guess = init_guess_class()
#     init_guess.update_all(ig)
#     fit_plot_comp_plots(dfs,init_guess) #doing the fit and plot the comparison plot











































































