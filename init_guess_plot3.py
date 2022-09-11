# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:28:02 2022

@author: pokey

In this file, the initial guess finding algorithm is written,
as well as the interactive plots and demonstration.


"""

import pero_model_fit2 as pmf
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
    # path = "sample_plot.png"
    # imgs = PImage.open(path)
    # imgs.show()
    
    # a = plt.figure()
    img = mpimg.imread('max_sample.png')
    # imgplot = plt.imshow(img)
    # plt.figure('The position of the minimum and the maxima')
    # plt.show()
    
    # plt.figure()
    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2,figsize =(15,6),gridspec_kw={'width_ratios': [2, 1]})
    ax1.plot(zrlist,-zilist,'.')
    ax1.set_title('Please click the maxima in this plot')
    ax2.imshow(img)
    ax2.set_title('Choose the two maxima like this')
    vertices = plt.ginput(2)
    plt.close()
    
    
    img2 = mpimg.imread('min_sample.png')
    fig2, (ax3,ax4) = plt.subplots(nrows = 1, ncols = 2,figsize =(15,6),gridspec_kw={'width_ratios': [2, 1]})
    ax3.plot(zrlist,-zilist,'.')
    ax3.set_title('Please click the minimum in this plot')
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
    return R_n8 , R_n0 , w_n , w_t , C_G ,w_r


def init_guess_find(df ,crit_points):
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
    return init_guess_find(df ,crit_points)






#%% PLOTTING





wlist = np.logspace(-6, 6 ,10000)




def R_ion_Slider(init_guess, df, v,crit_points):
    simu_Z, simu_J1 = pmf.pero_model(wlist,*init_guess.values(),v)
    print(v)
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
        fig.canvas.draw_idle()
        
        
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
    
    # the button to proceed to the next step    
    ax_next = plt.axes([0.8, 0.925, 0.1, 0.04])    #axis of the next step pattern
    button_next = Button(ax_next , 'Next step', hovercolor='0.975')
    button_next.on_clicked(lambda event:all_param_sliders(event,init_guess, df, v,crit_points))
    
    
    resetax._button = button_R
    axbox._text_box = text_box
    ax_next._button_next = button_next
    
    plt.show()

#%%

def all_param_sliders(event,init_guess, df, v,crit_points):
    plt.close()
    simu_Z, simu_J1 = pmf.pero_model(wlist,*init_guess.values(),v)
    fig, ((ax1 ,ax2),(ax3,ax4)) = plt.subplots(figsize=(18, 12),ncols = 2 , nrows = 2)


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
    ax2.set_ylabel('|Z|')
    ax2.set_xlabel(r'frequency $\omega$')
    ax2.set_title(r'|Z|, $\theta$ vs. frequency')
    ax2.legend(loc = 3, fontsize = 'small')
    ax2.spines['left'].set_color('c')
    ax2.tick_params(axis='y', colors='c')

    #theta plot

    ax_t = ax2.twinx()
    line_t, = ax_t.plot(df['frequency'],np.angle(df['impedance'].values),'.',ms = 4,color = 'peru', label = r'experiment $\theta$')
    line_t_ig, = ax_t.plot(wlist , np.angle(simu_Z),'--',ms = 4,color = 'orange', label = r'initial guess $\theta$')
    ax_t.set_xscale('log')
    ax_t.set_ylabel(r'$\theta$')
    ax_t.legend(loc = 1, fontsize = 'small')
    ax_t.spines['right'].set_color('orange')
    ax_t.tick_params(axis='y', colors='orange')

    #change only the C_a in popt now as a test
    # axC_a = plt.axes([0.25, 0.5, 0.65, 0.03])
    ax_list = {} 
    ax_list_t = {} #stores axis postion for the textbox
    sliders = {}
    textboxs = {}
    param_name = ['C_A', 'C_ion', 'R_ion', 'C_g', 'J_s', 'nA' ]
    param_dict ={'C_A':0, 'C_ion':1, 'R_ion':2, 'C_g':3, 'J_s':4, 'nA':5}    #establish the correspondance between the order and the name of the parameters
    range_list = [(1/3 * init_guess.C_A, 3 * init_guess.C_A ),
                  (1/3 * init_guess.C_ion, 3 * init_guess.C_ion),
                  (1/10 * init_guess.R_ion, 10 * init_guess.R_ion),
                  (1/10 * init_guess.C_g, 10 * init_guess.C_g),
                  (1/2 * init_guess.J_s, 3 * init_guess.J_s),
                  (1/1.5 * init_guess.nA, 1.5 * init_guess.nA)
                  ]



    for i in range(0,6):
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
                              'Set '+ param_name[i]+' manually: ',
                              initial='')

    sl_val_list =[]

    for key in sliders:
        sl_val_list.append(sliders[key])

    def update(val,  ):            #function called when the value of slider is updated
        vals = [i.val for i in sl_val_list]
        init_guess.update_all(vals)
        simu_Z , j = pmf.pero_model(wlist,*vals,v)
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
        simu_Z , j = pmf.pero_model(wlist,*init_guess.values(),v)
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
    label = 'Parameters to fix during fitting'
    textbox = TextBox(text_ax,label)
    

    # a = plt.axes([0.1, 0.03 * (2)-0.02, 0.03, 0.02])
    # a._textbox = textboxs[0]
    for key in ax_list_t:
        ax_list_t[key]._textbox = textboxs[i]
    
    # the button to proceed to the next step    
    ax_next = plt.axes([0.8, 0.95, 0.1, 0.02])    #axis of the next step pattern
    button_next = Button(ax_next , 'Start Fitting', hovercolor='0.975')
    button_next.on_clicked(lambda event:plot_comp_plots(event, param_to_fix,df,init_guess,v))
    
    ax_next._button_next = button_next
    resetax._button = button
    rax._checkbox = check
    fig.canvas.draw_idle()







#%%
def plot_comp_plots(event, param_to_fix,df,init_guess,v):
    plt.close()
    fix_index =  param_to_fix.fix_index()
    result = pmf.global_fit([df] , init_guess.values() , fix_index)
    report_fit(result)
    result_dict = result.params.valuesdict()
    popt = []
    for key in result_dict:
        popt.append( result_dict[key])
    z , j = pmf.pero_model(wlist,*popt,v)
    z_ig, j_ig = pmf.pero_model(wlist,*init_guess.values(),v)

    fig, ((ax1 ,ax2),(ax3,ax4)) = plt.subplots(figsize=(18, 12),ncols = 2 , nrows = 2)
    fig.suptitle('Comparison between the initial guess and the fitted parameters', fontsize = 16)

    #The Nyquist plot
    ax_nyq = plt.subplot(212)
    line1, = ax_nyq.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'x', ms=4 , label = 'actual data')   #actual data line
    line2, = ax_nyq.plot(np.real(z),-np.imag(z),'m-', label = 'fitted') #fitted parameter line
    line3, =ax_nyq.plot(np.real(z_ig),-np.imag(z_ig),'b--', label = 'initial guess') #initial guess line
    ax_nyq.legend()
    ax_nyq.set_xlabel('Z\'')
    ax_nyq.set_ylabel('Z\'\'')

    #Z_real plot
    line_zr, = ax1.plot(df['frequency'],np.real(df['impedance'].values),'bx',ms = 4, label = 'actual data Z\' ')
    line_zr_ig, = ax1.plot(wlist,np.real(z_ig),linestyle = 'dotted',color = 'c', label = ' initial guess Z\'')
    line_zr_fit, = ax1.plot(wlist,np.real(z),linestyle = 'solid',color = 'm', label = ' fitted Z\'')
    ax1.set_xscale('log')
    ax1.set_ylabel('Z\'')
    ax1.set_xlabel(r'frequency $\omega$')
    ax1.set_title('Real Z, effective capacitance vs. frequency')
    ax1.legend(loc = 3, fontsize = 'small')
    ax1.spines['left'].set_color('c')
    ax1.tick_params(axis='y', colors='c')

    #effective ccapacitance

    ax_eff = ax1.twinx()
    C_eff = np.imag(1 / df['impedance']) / df['frequency']
    C_eff_ig = np.imag(1 / z_ig) / wlist
    C_eff_fit = np.imag(1 / z) / wlist


    line_Ceff, = ax_eff.plot(df['frequency'],C_eff,'x',ms = 4,color = 'peru', label = 'experiment effective capacitance')
    line_Ceff_ig, = ax_eff.plot(wlist , C_eff_ig,linestyle = 'dotted',ms = 4,color = 'orange', label = 'initial guess effective capacitance')
    line_Ceff_fit, = ax_eff.plot(wlist , C_eff_fit,linestyle = 'solid',ms = 4,color = 'y', label = 'fitted effective capacitance')
    ax_eff.set_yscale('log')
    ax_eff.set_xscale('log')
    ax_eff.set_ylabel(r'Im($Z^{-1}$)$\omega^{-1}$')
    ax_eff.legend(loc = 1, fontsize = 'small')
    ax_eff.spines['right'].set_color('orange')
    ax_eff.tick_params(axis='y', colors='orange')


    #abs Z part plot
    line_absz, = ax2.plot(df['frequency'],np.abs(df['impedance'].values),'bx',ms = 4, label = 'experiment |Z| ')
    line_absz_ig, = ax2.plot(wlist,np.abs(z_ig),linestyle = 'dotted',color = 'c', label = ' initial guess |Z|')
    line_absz_ig, = ax2.plot(wlist,np.abs(z),linestyle = 'solid',color = 'm', label = ' fitted |Z|')
    ax2.set_xscale('log')
    ax2.set_ylabel('|Z|')
    ax2.set_xlabel(r'frequency $\omega$')
    ax2.set_title(r'|Z|, $\theta$ vs. frequency')
    ax2.legend(loc = 3, fontsize = 'small')
    ax2.spines['left'].set_color('c')
    ax2.tick_params(axis='y', colors='c')

    #theta plot

    ax_t = ax2.twinx()
    line_t, = ax_t.plot(df['frequency'],np.angle(df['impedance'].values),'x',ms = 4,color = 'peru', label = r'experiment $\theta$')
    line_t_ig, = ax_t.plot(wlist , np.angle(z_ig),linestyle = 'dotted',ms = 4,color = 'orange', label = r'initial guess $\theta$')
    line_t_ig, = ax_t.plot(wlist , np.angle(z),linestyle = 'solid',ms = 4,color = 'y', label = r'fitted $\theta$')
    ax_t.set_xscale('log')
    ax_t.set_ylabel(r'$\theta$')
    ax_t.legend(loc = 1, fontsize = 'small')
    ax_t.spines['right'].set_color('orange')
    ax_t.tick_params(axis='y', colors='orange')
    fig.canvas.draw_idle()






def __main__(df,v):
    crit_points = find_point(df)
    ig = init_guess_find(df,crit_points) 
    init_guess = init_guess_class()
    init_guess.update_all(ig)
    R_ion_Slider(init_guess, df, v,crit_points)















































































