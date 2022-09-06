# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:49:42 2022

@author: pokey

This is the main procedure including fucntions:
-Getting initial guess
-Slider to adjust R_ion
-Slider to adjust all parameters
-Fix certain parameter
-Comparison between initial guess and the fitted parameters.

"""
import pero_model_fit as pmf
import init_guess_plot as igp 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import glob 
from matplotlib.widgets import TextBox,Slider, Button,CheckButtons
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit


#importing example excel data
dfs = []
for file in glob.glob('paperdata/**.xlsx'): 
    df = pd.read_excel(file)
    df = df[['frequency','z_real','z_imag','bias voltage','recomb current']]
    df['z_imag'] = -df['z_imag'].values
    dfs.append(df)

for df in dfs:
    df['impedance'] = df['z_real'].values + df['z_imag'].values * 1j

dfs.sort(key = lambda x: x['bias voltage'][0])  # making the dfs list sorted by the magnitude of the bias voltaege of each data set.

wlist = np.logspace(-6,6,1000)
a = 2 #change this to change the set of data to fit
v = [0,.795,.846,.894]
df = dfs[a]
v = v[a]
#Now the df is the original data, v is the bias voltage. These are the two things to be provided by the users.


#getting the initial guess by igp functions
crit_points = igp.find_point(df)
ig = igp.init_guess(df,crit_points) 
init_guess = igp.init_guess_class()
init_guess.update_all(ig)



#%%
'''
This part is for the R_ion Slider only
'''


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
    iglist = igp.init_guess_slider(dfs[a],points,R_ion)
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
    iglist = igp.init_guess_slider(dfs[a],points,R_ion)
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



#%%
'''
This part is for all the parameters Sliders and fixed paramters plots.
'''

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
    print(param)
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
textboxs[1].on_submit(lambda text: submit_2(text, crit_points,'C_B',init_guess))
textboxs[2].on_submit(lambda text: submit_2(text, crit_points,'R_ion',init_guess))
textboxs[3].on_submit(lambda text: submit_2(text, crit_points,'C_g',init_guess))
textboxs[4].on_submit(lambda text: submit_2(text, crit_points,'J_s',init_guess))
textboxs[5].on_submit(lambda text: submit_2(text, crit_points,'nA',init_guess))

for key in sliders:
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
param_to_fix = igp.fix_params() # this stores the information of what variables to fix in the global fit

def fix_param(label):
    index = labels.index(label)
    param_to_fix.update_param(label, not param_to_fix.get(label))

check.on_clicked(fix_param)

#%% Here the parameters are fitted to the data and the result is produced
'''
Here the parameters are fitted to the data and the result is produced
'''
fix_index =  param_to_fix.fix_index()
result = pmf.global_fit([df] , init_guess.values() , fix_index)
report_fit(result)
result_dict = result.params.valuesdict()
popt = []
for key in result_dict:
    popt.append( result_dict[key])


#%% In this cell, the fitted parameter is plotted with the initial guess plot to make comparison on the effect of the fit.


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
























