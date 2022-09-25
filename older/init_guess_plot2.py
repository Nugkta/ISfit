# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:28:02 2022

@author: pokey

In this file, the initial guess finding algorithm is written,
as well as the interactive plots and demonstration.


"""

import pero_model_fit as pmf
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import TextBox,Slider, Button,CheckButtons

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
        
    def update_all(self, init): # update all the attrs in one go by inputting a list of values
        self.C_A = init[0]
        self.C_ion = init[1]
        self.R_ion = init[2]
        self.C_g = init[3]
        self.J_s = init[4]
        self.nA = init[5]
        
    def update_R_ion(self, R_ion): #update R_ion only
        self.R_ion = R_ion
        
    def values(self): #function for returning all the values of the initial guess
        return [self.C_A, 
                self.C_ion ,
                self.R_ion ,
                self.C_g ,
                self.J_s ,
                self.nA ]
    
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
                self.nA ]
        index = []
        for i in range(0,6):
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
    plt.plot(zrlist,-zilist,'.')
    plt.title('please select the maxima')
    vertices = plt.ginput(2)
    plt.close()
    plt.plot(zrlist,-zilist,'.')
    plt.title('please select the minimum')
    mini = plt.ginput(1)
    plt.close()
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
    return R_n8 , R_n0 , w_n , w_t , C_G ,w_r


def init_guess(df ,crit_points):
    '''
    Then, by implementing the algorithm using the original data, user guessed R_ion and the obtained critical points, the initial guess
    can be obtained corresponds to the method written in the paper.
    '''
    R_n8 , R_n0 , w_n , w_t , C_G ,w_r = crit_points
    R_ion = float(input('Please input your guess of R_ion: '))
    J_n = np.real(df['recomb current'][0]) 
    k = R_n0 / R_n8
    nA = J_n *R_n8 *q / ( kb *T)
    C_ion = 1 / (w_n * k * R_ion)
    C_g = C_G
    C_A = C_ion / (1 - 1/k)
    C_B = 1 / (1/C_ion - 1/C_A)
    V = np.real(df['bias voltage'][0])
    J_s = J_n / np.e**((V*(1 - C_ion/C_A)*q) / (nA * kb * T))
    return C_A, C_ion, R_ion, C_g, J_s, nA

def init_guess_slider(df, points, R_ion):       
    '''
    The function for the slider of R_ion (known points and R_ion), different from the previous one because
    here the R_ion is define by the slider(as function input), not by user input directly.
    '''
    R_n8 , R_n0 , w_n , w_t , C_G ,w_r = points
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
    return C_A, C_ion, R_ion, C_g, J_s, nA


def get_init_guess(df): #put together crit points and init guess
    crit_points = find_point(df)
    return init_guess(df ,crit_points)


def R_ion_Slider(init_guess, df, v,crit_points):
    wlist = np.logspace(-10, 10 ,10000)
    simu_Z, simu_J1 = pmf.pero_model(wlist,*init_guess.values(),v)
    fig , ((ax1 ,ax2),(ax3,ax4)) = plt.subplots(2 , 2,figsize = (20,10)) #opening the canvas for the plot of Nyquist
    
    
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
    ax2.set_title(r'|Z|, $\theta$ vs. frequency')
    ax2.legend(loc = 3)
    ax2.spines['left'].set_color('c')
    ax2.tick_params(axis='y', colors='c')
    
    #theta plot
    
    ax_t = ax2.twinx()
    line_t, = ax_t.plot(df['frequency'],np.angle(df['impedance'].values),'.',ms = 4,color = 'peru', label = r'experiment $\theta$')
    line_t_ig, = ax_t.plot(wlist , np.angle(simu_Z),'--',ms = 4,color = 'orange', label = r'initial guess $\theta$')
    ax_t.set_xscale('log')
    ax_t.set_ylabel(r'$\theta$')
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
        simu_Z, simu_J1 = pmf.pero_model(wlist,*init_guess.values(),v)
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
    button_R.on_clicked(reset_R_ion)
    
    def submit(text,points):
        R_ion = float(text)
        iglist = init_guess_slider(df,points,R_ion)
        init_guess.update_all(iglist)
        simu_Z, simu_J1 = pmf.pero_model(wlist,*init_guess.values(),v)
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
    text_box = TextBox(axbox, 'Set R_ion manually: ', initial=initial_text)
    text_box.on_submit(lambda text: submit(text, crit_points))
    plt.show()



























































