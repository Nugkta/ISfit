# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 02:10:50 2022

@author: Stan

This file contains 
1. the initial guess finding algorithm
2. the visualisation for the fitting result with sliders to adjust the initial guess


"""
import pero_model_fit4 as pmf
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider, Button, CheckButtons
from lmfit import report_fit
import matplotlib.image as mpimg


#%% defining other functions and classes used later in the algorithm and the  visualisation


#constants used in the model
VT = 0.026 # = kT/q
q = 1.6e-19 #elementary charge
T = 300 #room temperture
kb = 1.38e-23 #boltzmann constant


#Class for storing initial guess
class init_guess_class:       
    '''
    This is the class for storing the initial guess.
    A class is needed instead of a list, because the interactive plot need to update the initial guess in real time.
    The meaning of the variables should refer to the documents
    '''
    def __init__(self): #initialising the initial guesses for the variables
        self.C_A = None
        self.C_ion = None
        self.R_ion = None
        self.C_g = None
        self.J_s = None
        self.nA = None
        self.R_s = None
        self.R_shnt = 10000  #setting a large initial guess for R_shnt for the case of global fit without 0V
        
        
    def update_all(self, init, mode): # update all the attrs in one go by inputting a list of values for different scenerios
        if mode == 'glob_no0V' or 'ind_no0V':
            self.C_A = init[0]
            self.C_ion = init[1]
            self.R_ion = init[2]
            self.C_g = init[3]
            self.J_s = init[4]
            self.nA = init[5]
            self.R_s = init[6]
            
        if mode == 'glob_0V':
            self.C_A = init[0]
            self.C_ion = init[1]
            self.R_ion = init[2]
            self.C_g = init[3]
            self.J_s = init[4]
            self.nA = init[5]
            self.R_s = init[6]
            self.R_shnt = init[7]
            
        if mode == 'ind_0V': 
            self.C_ion = init[0]
            self.C_g = init[1]
            self.R_ion = init[2]
            self.J_nA = init[3]
            self.R_s = init[4]
            self.R_shnt = init[5]

    
    # def update_R_ion(self, R_ion): 
    #     self.R_ion = R_ion
        
    def values(self, mode): #function for returning all the values of the initial guess
        if mode == 'glob_0V' or 'glob_no0V':    
            return [self.C_A, 
                    self.C_ion ,
                    self.R_ion ,
                    self.C_g ,
                    self.J_s ,
                    self.nA,
                    self.R_s,
                    self.R_shnt]
        
        if mode == 'ind_0V': #0V individual
            return [self.C_ion ,
                    self.C_g ,
                    self.R_ion ,
                    self.J_nA,
                    self.R_s,
                    self.R_shnt]
        
        if mode == 'ind_no0V': #no 0V individual
            return [self.C_A,
                    self.C_ion ,
                    self.R_ion ,
                    self.C_g ,
                    self.J_s,
                    self.nA,
                    self.R_s,
                    self.R_shnt]
        
    def update_param(self,param,value): #updating a specific parameter
        setattr(self,param,value)




#Class of the parameters to fix during the fitting 
class fix_params():
    '''
    This is the class of the parameters to fix during the fitting.
    Meaning of the variable should refer to the document.
    '''
    
    def __init__(self):
        self.C_A = False
        self.C_ion = False
        self.R_ion = False
        self.C_g = False
        self.J_s = False
        self.nA = False
        self.R_s = False
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
                self.R_s,
                self.R_shnt]
        
        index = []
        
        for i in range(0,8):
            if attr_list[i] == True:
                index.append(i)
        return index #this function will return the index of the parameters that the user selected to fit.
    
    
    
#the function for finding the closest point on the line to the user selected point on the canvas
def closest_node(node, nodes): 
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2) 


# Function for finding the maximum of the Nyquist plot of the data.
def find_extremum(df):
    '''
    This function will plot the Nyquist plot from the dataframe of the experimental data and let the user choose the approximate local maxima and minimum of the plot manually by eye.
    Returns the index of the selected points.
    '''
    #extract the data from the dataframe
    zrlist = np.real(df['impedance'].values)
    zilist = np.imag(df['impedance'].values)
    zrilist = np.stack((zrlist , -zilist), axis = 1)
    
    # plotting the interactive Nyquist plot, and the examplary plot to instruct the user to choose the desired points
    img = mpimg.imread('max_sample.png')
    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2,figsize =(16,7),gridspec_kw={'width_ratios': [2, 1]})
    ax1.plot(zrlist,-zilist,'.')
    ax1.set_title(' Click the maxima on this plot for the initial guesses')
    ax2.imshow(img)
    ax2.set_title('Choose the two maxima like this')
    vertices = plt.ginput(2)
    plt.close()
    
    # plotting the second interactive plot
    img2 = mpimg.imread('min_sample.png')
    fig2, (ax3,ax4) = plt.subplots(nrows = 1, ncols = 2,figsize =(16,7),gridspec_kw={'width_ratios': [2, 1]})
    ax3.plot(zrlist,-zilist,'.')
    ax3.set_title(' click the minimum on this plot for the initial guesses')
    ax4.imshow(img2)
    ax4.set_title('Choose the minimum like this')
    mini = plt.ginput(1)
    plt.close()

    # convert the coordinates extracted from the interactive plots to the index in the original data.
    nhlist = []
    nllist = []
    for vert in vertices:
        num = closest_node(vert , zrilist)
        nhlist.append(num)
    minnum = closest_node(mini , zrilist)
    nllist.append(minnum)
    return nhlist, nllist















































