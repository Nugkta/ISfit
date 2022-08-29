 # -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 00:03:18 2022

@author: pokey

Here I will try to wirte the functions for the third time revised initial guess function where the user puts in the guess for the R_ion and the vlue of JN for the algorithm to find the corresponding
initial guess set.



"""
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand
from waiting import wait
from scipy.optimize import fsolve


#%%

# def find_p(df):
#     wzlist = df[['frequency','impedance']].to_numpy()
#     zrlist = np.real(df['impedance'].values)
#     zilist = np.imag(df['impedance'].values)
#     R_n0 = zrlist[-1]   #the last element of zrlist, corresponds to the end of the curve. Might need to revise for imcomplete circle
#     nhlist = np.array(argrelextrema(zilist, np.less))
#     whlist = np.real(wzlist[nhlist][0][: , 0])
#     w0 = min(whlist)
#     C_a = 1/(R_n0 * w0)
#     #print('R is ----', R_n0)
#     print( zrlist[-1]  )
#     return C_a 

def closest_node(node, nodes): #the function for finding the closest point on the line to the user selected point
    #nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def find_extremum(df): #algorithm to find the extremums of the Nyquist plot
   # wzlist = df[['frequency','impedance']].to_numpy()
    zrlist = np.real(df['impedance'].values)
    zilist = np.imag(df['impedance'].values)
    zrilist = np.stack((zrlist , -zilist), axis = 1)
    #now letting thr user to determine the rough location of the peak and local minimum
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
    zlist = df['impedance'].to_numpy()
    wzlist = df[['frequency','impedance']].to_numpy()
    nhlist, nllist= find_extremum(df)
    R_n8 = wzlist[nllist[0]][1].real      #obtained the R_n8 parameter
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




def init_guess(df):
    R_n8 , R_n0 , w_n , w_t , C_G ,w_r = find_point(df)
    R_ion = float(input('please input your guess of R_ion: '))
    J_n = np.real(df['recomb current'][0])
    print('J_n is ', J_n)
    J_n_t = R_n8 * q / (1.93 *kb *T)
    print('theoretical J_n is:',1/J_n_t)
    k = R_n0 / R_n8
    n = J_n *R_n8 *q / ( kb *T)
    C_ion = 1 / (w_n * kb * R_ion)
    C_g = C_G
    C_A = C_ion / (1 - 1/k)
    C_B = 1 / (1/C_ion - 1/C_A)
    return C_A,C_B, C_g, R_ion, n
    







#%% FOR TESTING

#first generating the dataframe for testing

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

import pero_ig_fit_func as pif 


V = 0.8
w = np.logspace(-6 , 6 , 1000) 

zlist, J1 = pif.pero_model(w, C_a, C_b, R_i, C_g, J_s, nA, V)
wzlist = np.stack((w , zlist) , axis = -1)
wzlist = wzlist[wzlist[:, 1].real.argsort()]     # sorting the wzlist by the real part of the impedance(preparing for the find extrema function)
vlist = np.ones(len(wzlist)) * V
jlist = np.ones(len(wzlist)) * J1
wz_v_jlist = np.column_stack((wzlist , vlist,jlist))
#v_wz_jlist = np.column_stack((v_))
df = pd.DataFrame(wz_v_jlist , columns=['frequency' , 'impedance' , 'bias voltage','recomb current'])


#%% TESTING THE FIND POINT FUNCTION
R_n8 , R_n0 , w_n , w_t , C_G,w_r = find_point(df)


#%% TESTING THE INIT GUESS FUNCTION
C_A,C_B, C_g, R_ion,n = init_guess(df)



#%%

"""
the parts below this was to investigate the without-R_ion-guess method of initial guess algorithm.
outdated for now.
"""










#%%my own formula for w_t
C_ion = 1 / (1/C_a + 1/C_b)
c_ion = C_ion
R_ion = R_i
c_g = C_g
w_t_t = np.sqrt(C_g*(C_g + C_ion))/(R_i * C_g**2 + R_i*C_ion*C_g)
print(w_t_t)
w = np.logspace(-8,-2,10000)


def funcb(w):  # the function of theta vs. freqeuncy derived in matlab
    eq = -(1/(c_ion*w*(R_ion**2*c_g**2*w**2 + 1)) + (R_ion*c_g*(R_ion*c_g*w + R_ion*c_ion*w))/(c_ion*(R_ion**2*c_g**2*w**2 + 1)))/((R_ion*c_g)/(c_ion*(R_ion**2*c_g**2*w**2 + 1)) - (R_ion*c_g*w + R_ion*c_ion*w)/(c_ion*w*(R_ion**2*c_g**2*w**2 + 1)))
    return eq


plt.plot(w,funcb(w))
plt.xscale('log')
plt.yscale('log')

#%%

# plt.plot(np.real(df['frequency'].values), np.angle(np.conjugate(df['impedance']),deg = False),'.')
# plt.plot(np.real(df['frequency'].values), (np.arctan(-np.imag(df['impedance'])/ np.real(df['impedance']))),)
# plt.xscale('log')
# plt.yscale('log')

plt.plot(np.real(df['frequency'].values), np.angle(-(df['impedance']),deg = False),'.')
# plt.plot(np.real(df['frequency'].values), np.angle(np.conjugate(df['impedance']),deg = False),'.')
#plt.plot(np.real(df['frequency'].values), -(np.arctan(np.imag(df['impedance'])/ np.real(df['impedance']))),)
plt.xscale('log')
plt.yscale('log')
#%%
c = 1 + 2j
a5 = np.arctan(np.imag(c)/ np.real(c))
a6 =  np.angle(c,deg = True)
print(a5, a6)
#%%
C_ion_t = 1/(1/C_a +1/C_b)
w_t_t = 1/(R_i *np.sqrt(C_g*(C_g +C_ion_t)))
#%%#%% writing initial guess function, use w_n find c_g
n_J_n = R_n8 *q /(kb*T)
k = R_n0 / R_n8
def func(C_g):
#def func(C_ion):
    # R_ion = 1 / (w_n * k * C_ion)
    # C_g = 1/ (1 / C_G - 1/C_ion)
    # eq = (R_ion * np.sqrt(C_g * (C_g + C_ion))) - 1 / w_t
    C_ion = (1/(R_n8*w_r - 1/C_g))
    #eq=(1/k/(w_n/(R_n8*w_r - 1/C_g))*np.sqrt(np.abs(C_g*(C_g + 1/(R_n8*w_r - 1/C_g)) ))- 1/w_t)
    eq=(1/k/(w_n*C_ion))*np.sqrt((C_g*(C_g + C_ion) ))- 1/w_t 
    #print(np.sqrt(C_g * (C_g + C_ion)))
    # C_ion = (1/(1/C_G -1/C_g))
    # eq = 1/(k*w_n * C_ion) * ((C_g * (C_g + C_ion)))**0.5 - 1 / w_t
    #eq = k/(w_n/(R_n8*w_r - 1/C_g))*np.sqrt((C_g*(C_g + 1/(R_n8*w_r - 1/C_g)))) - 1/w_t
    #eq = np.abs(k/(w_n/(R_n8*w_r - 1/C_g))*np.sqrt(C_g*(C_g + 1/(R_n8*w_r - 1/C_g)) - 1/w_t))
    return eq
    # return R_ion * np.sqrt(C_g * (C_g + C_ion)) 
    
#%% writing initial guess function, use C_g find c_g
def func(C_g):
    C_ion = 1 / (1/C_G - 1/C_g)
    eq=(1/k/(w_n*C_ion))*np.sqrt((C_g*(C_g + C_ion) ))- 1/w_t 
    return eq
    
#%%
# def func(C_g):
#     C_ion = 1/(R_n8*w_r - 1/C_g)
#     eq=(C_g*(C_g + C_ion))
#     return C_ion

    
 #%%   writing initial guess function, use C_g find c_ion
def func(C_ion):
    R_ion = 1 / (w_n * k * C_ion)
    C_g = 1/ (1 / C_G - 1/C_ion)
    #eq = (R_ion * np.sqrt((C_g * (C_g + C_ion)))) - 1 / w_t
    eq = (R_ion * np.sqrt((C_g * (C_g + C_ion)))) - 1 / w_t
    return eq     
    
#%%writing function testing is C_g goes negaitve
# def func(C_ion):
#     R_ion = 1 / (w_n * k * C_ion)
#     C_g = 1/ (1 / C_G - 1/C_ion)
#     eq = (R_ion * np.sqrt((C_g * (C_g + C_ion)))) - 1 / w_t
#     return C_g    

    
#%%
root = fsolve(func, 1e-7)
print(root)
#%%
# C_g = np.logspace(-10,10,10000)
C_ion = np.logspace(-8,-2,10000)
list9 = np.ones(10000)*10
# plt.plot(C_ion , func(C_ion)-0.9
plt.plot(C_ion , func(C_ion)+10)
plt.plot(C_ion , list9) 
plt.xscale('log')
plt.yscale('log')


#%% theoretical values
C_i = 1/(1/C_a +1/C_b)
C_G_t = 1/(1/C_i +1/C_g)


plt.plot(df['frequency'], np.real(df['impedance']))

plt.xscale('log')
plt.yscale('log')






#%%
C_im =1/ (R_n8 *w_r - 1/4.7973e-05)




























#%% PROCEDURE of finding the initial guess
#这里是我在尝试用它
from scipy.optimize import fsolve



# def picking():
#     pick_extremum(zrlist,zilist, 'max')
#     pick_extremum(zrlist,zilist, 'min')


#preparing for finding the critical points on the plot 
wzlist = df[['frequency','impedance']].to_numpy()
zrlist = np.real(df['impedance'].values)
zilist = np.imag(df['impedance'].values)
zrilist = np.stack((zrlist , -zilist), axis = 1)
#now letting thr user to determine the peak and local minimum
plt.plot(zrlist,-zilist,'.')
plt.title('please select the vertices')
vertices = plt.ginput(2)
plt.close()


plt.plot(zrlist,-zilist,'.')
plt.title('please select the minimum')
mini = plt.ginput(1)
plt.close()
# use the approximate position the user selected to find the corresponding index in the dataframe
def closest_node(node, nodes):
    #nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

nhlist = []
nllist = []

for vert in vertices:
    num = closest_node(vert , zrilist)
    nhlist.append(num)

minnum = closest_node(mini , zrilist)
nllist.append(minnum)
nllist = np.asarray(nllist)
nhlist = np.asarray(nhlist)


# pick_extremum(zrlist,zilist, 'max')
# pick_extremum(zrlist,zilist, 'min')
# picking()
# def is_continue(cont):
#     if cont == 'y':
#         return True 
#     return False
# cont = input('are you ready to continue? y/n \n')
# wait(lambda: is_continue(cont), timeout_seconds=120, waiting_for="something to be ready")
# nhlist = np.loadtxt('vertices.txt',delimiter = ',')
# nllist = np.loadtxt('mins.txt',delimiter = ',')




# nhlist2 = np.array(argrelextrema(zilist, np.less))[0]
# nllist2 = np.array(argrelextrema(wzlist[:,1].imag, np.greater))[0]
whlist = np.real(wzlist[nhlist][: , 0])

# First extract all the needed information from the plot
R_n0 = zrlist[-1]   #the last element of zrlist, corresponds to the end of the curve. Might need to revise for imcomplete circle
R_n8 = wzlist[nllist[0]][1].real
wn = min(whlist)
wr = max(whlist)
wt = wzlist[[nllist][0]][0][0].real  #w_theta

#Now start to implement the finding algo
k = R_n8 / R_n0
#solving the three variable non linear equation.
def equations(p, R_n0 , R_n8 , wn , wr , wt , k ):
    C_ion ,  C_g = p
    eq1 = 1 / (R_n8 *wr - 1 / C_g) - C_ion
    eq2 = k / (wn * C_ion) * np.sqrt(C_g * (C_g + C_ion)) - 1/wt
    return (eq1 , eq2 )
#%%
C_ion , C_g = fsolve(lambda p: equations(p, R_n0 , R_n8 , wn , wr , wt , k ), (1 , 1) , maxfev= 1000000 )
    



print(C_ion , C_g)


#%% then trying out the functions
C_g = 2.8e-8  
C_ion = 1/ (1/C_a +1/C_b)

print(1 / (R_n8 *wr - 1 / C_g) - C_ion)

#%%trying the interactive point picking function

def pick_vertex(zrlist,zilist):
    myfile = open('vertices.txt' , 'w')
    myfile.close()
    # simple picking, lines, rectangles and text
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title('Please click on the vertices of the Nyquist plot', picker=True)
    ax1.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
    line, = ax1.plot(zrlist,-zilist, 'o', picker=True, pickradius=5)

    # pick the rectangle
    # ax2.bar(range(10), rand(10), picker=True)
    # for label in ax2.get_xticklabels():  # make the xtick labels pickable
    #     label.set_picker(True)
    def onpick1(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            top = (np.column_stack([xdata[ind], -ydata[ind]])[0])
            print('onpick1 line:', top)
            myfile = open('vertices.txt' , 'a')
            myfile.write(str(top)+'\n')
            myfile.close()
            #return np.column_stack([xdata[ind], ydata[ind]])
        # elif isinstance(event.artist, Rectangle):
        #     patch = event.artist
        #     print('onpick1 patch:', patch.get_path())
        # elif isinstance(event.artist, Text):
        #     text = event.artist
        #     print('onpick1 text:', text.get_text())

    fig.canvas.mpl_connect('pick_event', onpick1)

 
#pick_simple(zrlist,zilist)



#%% reading the string in file to be np array
#y = np.loadtxt('vertices.txt',delimiter = ',')
 
#nhlist = get_points('vertices.txt')



