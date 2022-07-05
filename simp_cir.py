from sre_constants import BRANCH
import matplotlib.pyplot as plt
import numpy as np


def find_imp (w, R1, R2, C):
    z_r = R1 + R2/(1+(w*R2*C)**2)
    z_i = -w * R2**2 * C/(1 + (w*R2*C)**2)
    return z_r, z_i

def plot_spec(w_h, w_l, R1, R2, C):
    wlist = np.linspace(w_h, w_l, 50000)
    zrlist = []
    zilist = []
    for w in wlist:
        zr, zi = find_imp(w, R1, R2, C)
        zrlist.append(zr)
        zilist.append(-zi)
    plt.plot(zrlist, zilist,'.')
    plt.show()
    return zrlist, zilist

    
a ,b = plot_spec(0,100,3,4,4)
a ,b = plot_spec(0,100,0,4,4)
























