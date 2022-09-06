# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:09:06 2022

@author: pokey

In this file, I will try to read and fit the data stored from the paper Fig 2a by using the functions 
written previously. 
Using older inital_guess algorithm

"""

import pero_ig_fit_func_3_old_init_model as pif
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import symbols, Eq, solve
import pandas as pd
from scipy.signal import argrelextrema
import glob 
import init_guess3 as ig3
from matplotlib.widgets import TextBox,Slider, Button,CheckButtons

def get_key(val , my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
#%%

dfs = []
for file in glob.glob('paperdata/**.xlsx'):
    #plt.figure()
    df = pd.read_excel(file)
    df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]
    df['z_imag'] = -df['z_imag'].values
    dfs.append(df)
    #plt.plot(df['z_real'],-df['z_imag'],'.')



# no longer effective # (because the experimental data do not contain a 0V data set, I will generate the data first in order to test the automaticality of the function. using the fitted parameters obtained by the first three data sets)

# w = dfs[0]['frequency'].values
# zlist, J1 = pif.pero_model(w,2.03534548e-04, 8.53497697e-05, 2.59998353e+04, 4.04292147e-07, 3.36778932e-12, 1.39011012e+00, 0)
# df0 = pd.DataFrame(columns = ['frequency','z_real','z_imag','bias voltage' , 'recomb current'])
# df0['frequency'] = w
# df0['z_real'] = zlist.real
# df0['z_imag'] = zlist.imag #Note there is a minus sign here to keep it consistent with the experiment data.
# df0['bias voltage'] = np.zeros(len(w))
# df0['recomb current'] = np.ones(len(w)) * 6.1e-13
# dfs.append(df0)


#change the extracted dataframe to be the format used in the previous study (minus z_imag and complex impedance)
for df in dfs:
    df['impedance'] = df['z_real'].values + df['z_imag'].values * 1j
    #plt.plot(df['z_real'],-df['z_imag'],'.')

dfs.sort(key = lambda x: x['bias voltage'][0])  # making the dfs list sorted by the magnitude of the bias voltaege of each data set.
dfs




#%% testing individual fit
#AND adding the slider function for changing the initial guess





from matplotlib.widgets import Slider, Button
wlist = np.logspace(-6,6,1000)

a = 2 #change this to change the set of data to fit
v = [0,.795,.846,.894]
df = dfs[a]
v = v[a]
crit_points = ig3.find_point(dfs[a])

iglist = ig3.init_guess(dfs[a],crit_points) #getting the initial guess by ig3 functions
init_guess = ig3.init_guess_class()
init_guess.update_all(iglist)

print(init_guess.values())




#%% PLOTTING OUT THE INITIAL GUESS AS MODEL INPUT TO SEE DEVIATION POSSIBLY ADD SLIDER LATER
# df = dfs[a]

# # plt.plot(np.real(df['impedance']) , -np.imag(df['impedance']) , 'g.')
# # # obtaining the Nyquist plot with initial guess as input to see the goodness of fit initially
# simu_Z, simu_J1 = pif.pero_model(wlist,*init_guess.values(),v[a])
# plt.plot(simu_Z.real , -simu_Z.imag )
# plt.title('R_ion = 1e6')

#%% try to add the slider for R_ion
df = dfs[a]
v = v[a]
#%%

simu_Z, simu_J1 = pif.pero_model(wlist,*init_guess.values(),v)
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
    iglist = ig3.init_guess_slider(dfs[a],points,R_ion)
    init_guess.update_all(iglist)
    simu_Z, simu_J1 = pif.pero_model(wlist,*init_guess.values(),v)
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
    iglist = ig3.init_guess_slider(dfs[a],points,R_ion)
    init_guess.update_all(iglist)
    simu_Z, simu_J1 = pif.pero_model(wlist,*init_guess.values(),v)
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

#%%


#notet that the R_ion here causes change differenetly from the previous slider because the previous R will cause change on Jn and nA simultaneously but the R_ion is not connected to the other parameters.



#  NOTE: After sliding the R_ion, the value of it in initial guess will be updated and stored in real time.



#%%SLIDERS FOR ALL PARAMETERS IN THE INITIAL GUESSS
# NOW TRY TO ADD SLIDERS FOR ALL PARAMETERS IN THE INITIAL GUESS
# init_guess.update_all([1.2688190813857057e-08, 6.635083668405059e-09, 600000000.0, 4.24144833113637e-07, 6.669776935905864e-11, 1.5479152046376812])

simu_Z, simu_J1 = pif.pero_model(wlist,*init_guess.values(),v)
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
param_name = ['C_A', 'C_B', 'R_ion', 'C_g', 'J_s', 'nA' ]
param_dict ={'C_A':0, 'C_B':1, 'R_ion':2, 'C_g':3, 'J_s':4, 'nA':5}    #establish the correspondance between the order and the name of the parameters
range_list = [(1/3 * init_guess.C_A, 3 * init_guess.C_A ),
              (1/3 * init_guess.C_B, 3 * init_guess.C_B),
              (1/10 * init_guess.R_ion, 10 * init_guess.R_ion),
              (1/10 * init_guess.C_g, 10 * init_guess.C_g),
              (1/2 * init_guess.J_s, 3 * init_guess.J_s),
              (1/1.5 * init_guess.nA, 1.5 * init_guess.nA)
              ]



for i in range(0,6):
    ax_list[i] = plt.axes([0.25, 0.03 * (i+2)-0.02, 0.5, 0.02])
    ax_list_t[i] = plt.axes([0.1, 0.03 * (i+2)-0.02, 0.03, 0.02])
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
    # popt = np.delete(popt , i)
    # popt = np.insert(popt,i,sliders[i].val)
    vals = [i.val for i in sl_val_list]
    init_guess.update_all(vals)
    simu_Z , j = pif.pero_model(wlist,*vals,v)
    # print(vals)
    # plt.figure()
    # plt.plot(np.real(z), -np.imag(z),'x', ms=4)
    #z , j = pif.pero_model(wlist,sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val,sliders[4].val,sliders[5].val,v1)
    #z , j = pif.pero_model(wlist,*popt,v1)
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
    print(param)
    new_value = float(text)                 #convert the string input in textbox to float
    init_guess.update_param(param,new_value)   #only update the value of which the textbox is updated
    simu_Z , j = pif.pero_model(wlist,*init_guess.values(),v)
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
    # amp = np.exp(R_slider.val)
    # R_slider.valtext.set_text('%.2e'%amp)
    fig.canvas.draw_idle()
    
    
# for key in textboxs:  
    # print(get_key(key,param_dict ))
textboxs[0].on_submit(lambda text: submit_2(text, crit_points,'C_A',init_guess))
textboxs[1].on_submit(lambda text: submit_2(text, crit_points,'C_B',init_guess))
textboxs[2].on_submit(lambda text: submit_2(text, crit_points,'R_ion',init_guess))
textboxs[3].on_submit(lambda text: submit_2(text, crit_points,'C_g',init_guess))
textboxs[4].on_submit(lambda text: submit_2(text, crit_points,'J_s',init_guess))
textboxs[5].on_submit(lambda text: submit_2(text, crit_points,'nA',init_guess))
    # textboxs[key].on_submit(lambda text: submit_2(text, crit_points,'C_A',init_guess))




for key in sliders:
    #print(type(sliders[key]))
    sliders[key].on_changed(lambda val: update(val))




resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')
def reset(event):
    for key in sliders:
        sliders[key].reset()
button.on_clicked(reset)
plt.show()



#define bottons to fix the initial guess while doing curve_fit
rax = plt.axes([0.1, 0.5, 0.1, 0.3])
lines = [line1, line_zr,line_Ceff,line_absz,line_t]
labels = param_name
check = CheckButtons(rax, labels)

param_to_fix = ig3.fix_params() # this stores the information of what variables to fix in the global fit




def fix_param(label):
    index = labels.index(label)
    param_to_fix.update_param(label, not param_to_fix.get(label))
    #print(var_to_fix.get(label))

check.on_clicked(fix_param)




#%%
fix_index =  param_to_fix.fix_index()
print(fix_index)
#%% PUTTING THE FIT AND THE INIT GUESS TOGETHER TO COMPARE

fix_index =  param_to_fix.fix_index()
popt, pcov = pif.global_fit([df] , init_guess.values() , fix_index)
# df = dfs[a]
#%%
z , j = pif.pero_model(wlist,*popt,v)
z_ig, j_ig = pif.pero_model(wlist,*init_guess.values(),v)

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
print('the fitted parameters are: \n C_A is %.2e, \n C_B is %.2e, \n R_ion is %.2e, \n C_g is %.2e, \n J_s is %.2e, \n nA is %.2e.' %(popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]))






# fig, axs = plt.subplots(figsize=(5, 5),ncols = 1 , nrows = 1)
# ax = axs
# line1 = plt.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'x', ms=4)
# line2, = plt.plot(np.real(z),-np.imag(z),'r--')
# line3 = plt.plot(np.real(z_ig),-np.imag(z_ig),'b--')
#%%
print('the fitted parameters are: \n C_A is %.2e, \n C_B is %.2e, \n R_ion is %.2e, \n C_g is %.2e, \n J_s is %.2e, \n nA is %.2e.' %(popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]))










































 



#%%

popt, pcov = pif.global_fit([dfs[a]] , init_guess)
# popt, pcov = pif.global_fit([dfs[1]] , init_guess)
# popt, pcov = pif.global_fit([dfs[1]] , init_guess)
# popt, pcov = pif.global_fit([dfs[1]] , init_guess)
df = dfs[a]
v1 = v
# v = [.795,]
# v = [.864,]
# v = [.894,]
z , j = pif.pero_model(wlist,*popt,v)
fig, axs = plt.subplots(figsize=(5, 5),ncols = 1 , nrows = 1)
ax = axs
line1 = plt.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'x', ms=4)
line2, = plt.plot(np.real(z),-np.imag(z),'r--')
ax.set_xlabel('Z\'')
ax.set_ylabel('Z\'\'')
plt.subplots_adjust(left=0.25, bottom=.5)
#change only the C_a in popt now as a test
# axC_a = plt.axes([0.25, 0.5, 0.65, 0.03])
ax_list = {} 
sliders = {}
param_name = ['C_a', 'C_b', 'R_i', 'C_g', 'J_s', 'nA' ]




for i in range(0,6):
    ax_list[i] = plt.axes([0.25, 0.05 * (i+2)-0.02, 0.55, 0.03])
    sliders[i] = Slider(
        ax = ax_list[i], 
        label = 'the value of ' + param_name[i],
        valmin = 1/3 * popt[i],
        valmax = 3 * popt[i],
        valinit = popt[i],
        )

sl_val_list =[]
for key in sliders:
    sl_val_list.append(sliders[key])

def update(val,  ):
    # popt = np.delete(popt , i)
    # popt = np.insert(popt,i,sliders[i].val)
    vals = [i.val for i in sl_val_list]
    z , j = pif.pero_model(wlist,*vals,v)
    # print(vals)
    # plt.figure()
    # plt.plot(np.real(z), -np.imag(z),'x', ms=4)
    #z , j = pif.pero_model(wlist,sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val,sliders[4].val,sliders[5].val,v1)
    #z , j = pif.pero_model(wlist,*popt,v1)

    

for key in sliders:
    #print(type(sliders[key]))
    sliders[key].on_changed(lambda val: update(val))
    
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')
def reset(event):
    for key in sliders:
        sliders[key].reset()
button.on_clicked(reset)
# sliders[0].on_changed(lambda val: update(val,))
# sliders[1].on_changed(lambda val: update(val,))
# sliders[2].on_changed(lambda val: update(val,))
# sliders[3].on_changed(lambda val: update(val,))
# sliders[4].on_changed(lambda val: update(val,))
# sliders[5].on_changed(lambda val: update(val,))
#%%some other plots to examine the effectiveness of the fit

z , j = pif.pero_model(df['frequency'].values,*popt,v)
plt.plot(df['frequency'].values,np.abs(z),'--')


df = dfs[a]
plt.plot(np.real(df['frequency'].values), np.abs(df['impedance']),'.')
plt.title('freq vs. abs(z)')
plt.show()
#####################################################
#%%
z , j = pif.pero_model(df['frequency'].values,*popt,v)
plt.plot(df['frequency'].values,np.angle(z),'r--')


df = dfs[a]
plt.plot(np.real(df['frequency'].values), np.angle(df['impedance']),'g.')
plt.title(r'freq vs. $\theta$(z) ')

# plt.xlim([-300,3000])
# plt.ylim([-200,1000])







#%%Checking the values 

wlist = np.logspace(-6,6,1000)

z , j =pif.pero_model(wlist,9.248395176094367e-05, 5.459655067388845e-05, 56976.84053607925, 4.2975313043758746e-07, 1.5532621477368821e-21, 0.6192410048972403,v)
plt.figure()
plt.plot(np.real(z), -np.imag(z),'x', ms=4)










































#%%
r = pif.find_Ri(dfs,3.6e-6)








#%% trying to include the simulated 0V data
for i in range(0,4):
    df = dfs[i]
    plt.plot(np.real(df['impedance'].values), np.imag(df['impedance']),'.')

#%% testing
df = pd.read_excel('paperdata/0.01sun.xlsx')
df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]

#%% Doing the global fit now, first using paper values as initial guess




#popt,pcov = pif.global_fit(dfs,[7.2e-2, 7.2e-2, 6.7, 4.4e-4, 6.1e-9, 1.79])
v = [.795,.876,.894]
popt,pcov = pif.global_fit(dfs,[7.2e-6, 3e-6, 6.7e5, 4.4e-7, 6.1e-13, 1.79])
wlist = np.logspace(-2,5,1000)
for i in v:
    z , j = pif.pero_model(wlist,*popt,i)
    plt.plot(np.real(z),-np.imag(z),'-')

df2=[]
for file in glob.glob('paperdata/**.xlsx'):        #plotting the original points as comparison
    df = pd.read_excel(file)
    df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]
    df2.append(df)
    plt.plot(df['z_real'],df['z_imag'],'.')




#%% plotting out the parameter
w = np.logspace(-10,15,1000)
zlist, J1 = pif.pero_model(w, 7.2e-6, 3e-6, 6.7e5, 4.4e-7, 6.1e-12, 1.8,.9)
plt.plot(np.real(zlist), -np.imag(zlist),'.')


#%% After all the testing,put the simulated and actual data into the fitting funtion as a whole

init_guess = pif.get_init_guess(dfs)
init_guess2 = [2.03534548e-04, 8.53497697e-05, 2.59998353e+04, 4.04292147e-07, 3.36778932e-12, 1.39011012e+00]
print('the initial guesses are', init_guess)
popt, pcov = pif.global_fit(dfs , init_guess)
print('the fitted parameters are',popt)

v = [0,.795,.876,.894]
wlist = np.logspace(-2,5,1000)
for i in v:
    z , j = pif.pero_model(wlist,*popt,i)
    plt.plot(np.real(z),-np.imag(z),'--')

for i in range(0,4):
    df = dfs[i]
    plt.plot(np.real(df['impedance'].values), -np.imag(df['impedance']),'.')

plt.xlim([-300,3000])
plt.ylim([-200,1000])













































