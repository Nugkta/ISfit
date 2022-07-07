from sre_constants import BRANCH
import matplotlib.pyplot as plt
import numpy as np

#the following part is for finding the general impedance of the circuit
def find_imp (w, R1, R2, C):                    ####
    z_r = R1 + R2/(1+(w*R2*C)**2)
    z_i = -w * R2**2 * C/(1 + (w*R2*C)**2)
    return z_r, z_i


#the following part is for plotting
def plot_spec(w_h, w_l, R1, R2, C, tp):         #parameters are: high end of freq, low end of freq,   
    wlist = np.linspace(w_h, w_l, 50000)        #first resistence, second resistance, capacitance, type of plot(Nyquist or freqency)
    zrlist = []                                 #reference Note section 1
    zilist = []
    for w in wlist:
        zr, zi = find_imp(w, R1, R2, C)
        zrlist.append(zr)
        zilist.append(-zi)
    if tp == 'N':
        plt.plot(zrlist, zilist,'.')
        plt.xlabel('real z')
        plt.ylabel('imag z')
        plt.show()
        return zrlist, zilist
    if tp == 'F':
        plt.xlabel('real z')
        plt.ylabel('imag z')
        plt.plot(wlist, zilist,'.')

#plotting Nyquist plot    
a,b = plot_spec(0,100,3,4,4,'N')
a,b = plot_spec(0,100,0,4,4,'N')



#%% plotting im-freq plot
a,b = plot_spec(0,10,0,4,4,'F')






#%% testing complex number operation
a = 10
b = 5
z = (a+b*1j)
















