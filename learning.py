# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:04:36 2022

@author: pokey

this file is for learning the routines

"""

import numpy as np
import pandas as pd
arr2D = np.array([[11, 12, 13, 22], [21, 7, 23, 14], [31, 10, 33, 7]])

sortedArr = arr2D[arr2D[:,1].argsort()]



#%%
a = np.array([[0.46162934, 0.67833399, 0.87730068]])
b = np.array([[0.10153951, 0.70881156, 0.38736128]])


c = np.stack((a,b), axis = -1)



#%%
from sympy import symbols, Eq, solve

x = symbols('x')
eq1 = Eq(2*x**2 + x + 1, 0)


sol = solve(eq1)

#%%
a = np.array([0 ,1 ,1 ,2])
b = np.array([2])
c = np.array([a,b], dtype = object)

#%%
array = np.random.rand(5,5,2)
df = pd.DataFrame(array)

#%% https://stackoverflow.com/questions/20339234/python-and-lmfit-how-to-fit-multiple-datasets-with-shared-parameters?rq=1
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit

def func_gauss(params, x, data=[]):
    A = params['A'].value
    mu = params['mu'].value
    sigma = params['sigma'].value
    model = A*np.exp(-(x-mu)**2/(2.*sigma**2))

    if data == []:
        return model
    return data-model

x  = np.linspace( -1, 2, 100 )
data = []
for i in np.arange(5):
    params = Parameters()          #用这个parameters object 保存的parameters，除了values， 还可以有bounds， name之类的东西
    params.add( 'A'    , value=np.random.rand() )
    params.add( 'mu'   , value=np.random.rand()+0.1 )
    params.add( 'sigma', value=0.2+np.random.rand()*0.1 )
    data.append(func_gauss(params,x))
#上面相当于给data这个list生产出了五个随机生成的gaussian 的y value的array。
plt.figure()

for y in data:
    fit_params = Parameters()
    fit_params.add( 'A'    , value=0.5, min=0, max=1)
    fit_params.add( 'mu'   , value=0.4, min=0, max=1)
    fit_params.add( 'sigma', value=0.4, min=0, max=1)
    minimize(func_gauss, fit_params, args=(x, y))
    report_fit(fit_params)

    y_fit = func_gauss(fit_params,x)
    plt.plot(x,y,'o',x,y_fit,'-')
plt.show()


# ideally I would like to write:
#
# fit_params = Parameters()
# fit_params.add( 'A'    , value=0.5, min=0, max=1)
# fit_params.add( 'mu'   , value=0.4, min=0, max=1)
# fit_params.add( 'sigma', value=0.4, min=0, max=1, shared=True)
# minimize(func_gauss, fit_params, args=(x, data))
#
# or:
#
# fit_params = Parameters()
# fit_params.add( 'A'    , value=0.5, min=0, max=1)
# fit_params.add( 'mu'   , value=0.4, min=0, max=1)
#
# fit_params_shared = Parameters()
# fit_params_shared.add( 'sigma', value=0.4, min=0, max=1)
# call_function(func_gauss, fit_params, fit_params_shared, args=(x, data))




#%%
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit

def gauss(x, amp, cen, sigma):
    "basic gaussian"
    return amp*np.exp(-(x-cen)**2/(2.*sigma**2))

def gauss_dataset(params, i, x):
    """calc gaussian from params for data set i
    using simple, hardwired naming convention"""
    amp = params['amp_%i' % (i+1)].value
    cen = params['cen_%i' % (i+1)].value
    sig = params['sig_%i' % (i+1)].value
    return gauss(x, amp, cen, sig)

def objective(params, x, data):
    """ calculate total residual for fits to several data sets held
    in a 2-D array, and modeled by Gaussian functions"""
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - gauss_dataset(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

# create 5 datasets
x  = np.linspace( -1, 2, 151)
data = []
for i in np.arange(5):
    params = Parameters()
    amp   =  0.60 + 9.50*np.random.rand()
    cen   = -0.20 + 1.20*np.random.rand()
    sig   =  0.25 + 0.03*np.random.rand()
    dat   = gauss(x, amp, cen, sig) + np.random.normal(size=len(x), scale=0.1)
    data.append(dat)

# data has shape (5, 151)
data = np.array(data)
assert(data.shape) == (5, 151)

# create 5 sets of parameters, one per data set
fit_params = Parameters()
for iy, y in enumerate(data):
    fit_params.add( 'amp_%i' % (iy+1), value=0.5, min=0.0,  max=200)
    fit_params.add( 'cen_%i' % (iy+1), value=0.4, min=-2.0,  max=2.0)
    fit_params.add( 'sig_%i' % (iy+1), value=0.3, min=0.01, max=3.0)

# but now constrain all values of sigma to have the same value
# by assigning sig_2, sig_3, .. sig_5 to be equal to sig_1
for iy in (2, 3, 4, 5):
    fit_params['sig_%i' % iy].expr='sig_1'

# run the global fit to all the data sets
result = minimize(objective, fit_params, args=(x, data))
report_fit(result)

# plot the data sets and fits
plt.figure()
for i in range(5):
    y_fit = gauss_dataset(fit_params, i, x)
    plt.plot(x, data[i, :], 'o', x, y_fit, '-')

plt.show()

#%% Learning symfit
from symfit import Parameter , Variable, parameters, variables, Model
from symfit import Fit
import numpy as np
def func(x,a,b):
    return a * np.e**(x) + b*1j

a = Parameter('a' , value = 1, min = -4 , max = 10)
b = Parameter('b' )
x = Variable('x')
model = func(x,a,b)

#generating data
xdata = np.linspace(0, 100, 100) # From 0 to 100 in 100 steps
a_vec = np.random.normal(15.0, scale=2.0, size=(100,))
b_vec = np.random.normal(100.0, scale=2.0, size=(100,))
ydata = a_vec * np.e**xdata + b_vec*1j  # Point scattered around the line 5 * x + 105

fit = Fit(model , xdata, ydata)
fit_result = fit.execute()

#%%

x_1, x_2, y_1, y_2 = variables('x_1, x_2, y_1, y_2')
y0, a_1, a_2, b_1, b_2 = parameters('y0, a_1, a_2, b_1, b_2')

def func(x):
    return 2 + 3 * np.e**(- 4 * x)

model = Model({
    y_1: y0 + a_1 * np.e**(- b_1 * x_1) ,
    y_2: y0 + a_2 * np.e**(- b_2 * x_2),
})

xdata1 = np.linspace(0, 100, 100)
xdata2 = np.linspace(0, 100, 100)

ydata1 = func( xdata1)
ydata2 = func( xdata2)

fit = Fit(model, x_1=xdata1, x_2=xdata2, y_1=ydata1, y_2=ydata2)
fit_result = fit.execute()


#%%
y = lambda x: x**2 +2

#%%
import glob

txtfiles = []
for file in glob.glob("MAPIdev2p5DRIEDLiCl25C/*.txt"):
    txtfiles.append(file)

print(txtfiles)

#%% solving multivariable nonlinear equation


from scipy.optimize import fsolve
import math

def equations(p):
    x , y , z = p
    eq1 = x**2 - y+2
    eq2 = x + y 
    eq3 = z - x - y
    return (eq1, eq2, eq3)



x, y ,z  =  fsolve(equations, (1, 1,1))





#%%



def equations(p):
    x, y,z = p
    return (x+y**2-4, math.exp(x) + x*y - 3, z - x - y)

x, y,z =  fsolve(equations, (1, 1,1))

print (equations((x, y,z)))


#%% learning interactive plotting


import time

import numpy as np
import matplotlib.pyplot as plt

def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

plt.clf()
plt.setp(plt.gca(), autoscale_on=False)

tellme('You will define a triangle, click to begin')

plt.waitforbuttonpress()

while True:
    pts = []
    while len(pts) < 3:
        tellme('Select 3 corners with mouse')
        pts = np.asarray(plt.ginput(3, timeout=-1))
        if len(pts) < 3:
            tellme('Too few points, starting over')
            time.sleep(1)  # Wait a second

    ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

    tellme('Happy? Key click for yes, mouse click for no')

    if plt.waitforbuttonpress():
        break



#%%

import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import rand

fig, ax = plt.subplots()
ax.plot(rand(100), rand(100), picker=3)
# 3, for example, is tolerance for picker i.e, how far a mouse click from
# the plotted point can be registered to select nearby data point/points.

def on_pick(event):
    global points
    line = event.artist
    xdata, ydata = line.get_data()
    print('selected point is:',np.array([xdata[ind], ydata[ind]]).T)

cid = fig.canvas.mpl_connect('pick_event', on_pick)
#%%

import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import rand

a = rand(1,5)
b = rand(1,5)
plt.plot(a,b , '.')
print('please choose the values of points')
plt.title('please choose the values of points')
x,y,z = plt.ginput(3)
print('please choose another 2 points')

plt.figure()
plt.plot(a,b , '.')
plt.title('please choose again')
plt.show()
x2, y2 = plt.ginput(2)
#%%

from matplotlib.widgets import LassoSelector 

ax = plt.subplot()
ax.plot(x, y,'.')

def onselect(verts):
    print(verts)
lasso = LassoSelector(ax, onselect)





#%%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand


# Fixing random state for reproducibility
np.random.seed(19680801)


def pick_simple():
    # simple picking, lines, rectangles and text
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('click on points, rectangles or text', picker=True)
    ax1.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
    line, = ax1.plot(rand(100), 'o', picker=True, pickradius=5)

    # pick the rectangle
    ax2.bar(range(10), rand(10), picker=True)
    for label in ax2.get_xticklabels():  # make the xtick labels pickable
        label.set_picker(True)

    def onpick1(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            print('onpick1 line:', np.column_stack([xdata[ind], ydata[ind]]))
        elif isinstance(event.artist, Rectangle):
            patch = event.artist
            print('onpick1 patch:', patch.get_path())
        elif isinstance(event.artist, Text):
            text = event.artist
            print('onpick1 text:', text.get_text())

    fig.canvas.mpl_connect('pick_event', onpick1)


def pick_custom_hit():
    # picking with a custom hit test function
    # you can define custom pickers by setting picker to a callable
    # function.  The function has the signature
    #
    #  hit, props = func(artist, mouseevent)
    #
    # to determine the hit test.  if the mouse event is over the artist,
    # return hit=True and props is a dictionary of
    # properties you want added to the PickEvent attributes

    def line_picker(line, mouseevent):
        """
        Find the points within a certain distance from the mouseclick in
        data coords and attach some extra attributes, pickx and picky
        which are the data points that were picked.
        """
        if mouseevent.xdata is None:
            return False, dict()
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        maxd = 0.05
        d = np.sqrt(
            (xdata - mouseevent.xdata)**2 + (ydata - mouseevent.ydata)**2)

        ind, = np.nonzero(d <= maxd)
        if len(ind):
            pickx = xdata[ind]
            picky = ydata[ind]
            props = dict(ind=ind, pickx=pickx, picky=picky)
            return True, props
        else:
            return False, dict()

    def onpick2(event):
        print('onpick2 line:', event.pickx, event.picky)

    fig, ax = plt.subplots()
    ax.set_title('custom picker for line data')
    line, = ax.plot(rand(100), rand(100), 'o', picker=line_picker)
    fig.canvas.mpl_connect('pick_event', onpick2)


def pick_scatter_plot():
    # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)

    x, y, c, s = rand(4, 100)

    def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', ind, x[ind], y[ind])

    fig, ax = plt.subplots()
    ax.scatter(x, y, 100*s, c, picker=True)
    fig.canvas.mpl_connect('pick_event', onpick3)


def pick_image():
    # picking images (matplotlib.image.AxesImage)
    fig, ax = plt.subplots()
    ax.imshow(rand(10, 5), extent=(1, 2, 1, 2), picker=True)
    ax.imshow(rand(5, 10), extent=(3, 4, 1, 2), picker=True)
    ax.imshow(rand(20, 25), extent=(1, 2, 3, 4), picker=True)
    ax.imshow(rand(30, 12), extent=(3, 4, 3, 4), picker=True)
    ax.set(xlim=(0, 5), ylim=(0, 5))

    def onpick4(event):
        artist = event.artist
        if isinstance(artist, AxesImage):
            im = artist
            A = im.get_array()
            print('onpick4 image', A.shape)

    fig.canvas.mpl_connect('pick_event', onpick4)


if __name__ == '__main__':
    pick_simple()
    pick_custom_hit()
    pick_scatter_plot()
    pick_image()
    plt.show()

#%%
from waiting import wait

def is_continue(cont):
    if cont == 'y':
        return True 
    return False

while 
cont = input('are you ready to continue? y/n \n')

wait(lambda: is_continue(cont), timeout_seconds=120, waiting_for="something to be ready")

print('kay i know you are ready')





#%%
"""
======
Slider
======

In this example, sliders are used to control the frequency and amplitude of
a sine wave.

See :doc:`/gallery/widgets/slider_snap_demo` for an example of having
the ``Slider`` snap to discrete values.

See :doc:`/gallery/widgets/range_slider` for an example of using
a ``RangeSlider`` to define a range of values.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# The parametrized function to be plotted
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

t = np.linspace(0, 1, 1000)

# Define initial parameters
init_amplitude = 5
init_frequency = 3

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)
ax.set_xlabel('Time [s]')

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.1,
    valmax=30,
    valinit=init_frequency,
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.Button`
#    - `matplotlib.widgets.Slider`





#%%
y = 3

def add1():
    window['y'] = 20

add1()
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

t = np.arange(0.0, 2.0, 0.01)
s0 = np.sin(2*np.pi*t)
s1 = np.sin(4*np.pi*t)
s2 = np.sin(6*np.pi*t)

fig, ax = plt.subplots()
l0, = ax.plot(t, s0, visible=False, lw=2, color='k', label='2 Hz')
l1, = ax.plot(t, s1, lw=2, color='r', label='4 Hz')
l2, = ax.plot(t, s2, lw=2, color='g', label='6 Hz')
plt.subplots_adjust(left=0.2)

lines = [l0, l1, l2]

# Make checkbuttons with all plotted lines with correct visibility
rax = plt.axes([0.09, 0.4, 0.1, 0.15])
labels = [str(line.get_label()) for line in lines]
visibility = [line.get_visible() for line in lines]
check = CheckButtons(rax, labels, visibility)


def func(label):
    print(label)
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    print(not lines[index].get_visible())
    plt.draw()

check.on_clicked(func)

plt.show()




#%%
#!/usr/bin/env python
#<examples/doc_basic.py>
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np

# create data to be fitted
x = np.linspace(0, 15, 301)
data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
        np.random.normal(size=len(x), scale=0.2) )

# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """ model decaying sine wave, subtract data"""
    amp = params['amp']
    shift = params['shift']
    omega = params['omega']
    decay = params['decay']
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
    return model - data

# create a set of Parameters
params = Parameters()
params.add('amp',   value= 10,  min=0)
params.add('decay', value= 0.1)
params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
params.add('omega', value= 3.0)


# do fit, here with leastsq model
minner = Minimizer(fcn2min, params, fcn_args=(x, data))
kws  = {'options': {'maxiter':10}}
result = minner.minimize()


# calculate final result
final = data + result.residual

# write error report
report_fit(result)

# try to plot results
try:
    import pylab
    pylab.plot(x, data, 'k+')
    pylab.plot(x, final, 'r')
    pylab.show()
except:
    pass


#%%


k = 1
b = 2


def func(x, k, b):
    y = k*x + b
    return y

param= input('input the parameter to fit')

newfunc = lambda x,param :func(w,k,b)    

#我想，如果用户input了k， newfunc 就变成lambda x,k :func(w,k,b), 这里面b就用的之前定义的b = 2，不再是个variable 了，只有k还是variable，然后可以拿去fit data
#但问题是用户input的string 不能代表variable name，有没有啥办法实现

#%%
from numpy import exp, linspace, random

def gaussian(x, amp, cen, wid):
    return amp * exp(-(x-cen)**2 / wid)
gmodel = Model(gaussian)
x_eval = linspace(0, 10, 201)
y_eval = gmodel.eval(x=x_eval, cen=6.5, amp=100, wid=2.0)

plt.plot(x_eval,y_eval,'.')



#%%
    for i in fix_index:    
        pars[param_dict[i]].max = init_guess[i] *1.01
        pars[param_dict[i]].mmin = init_guess[i] *0.99

















































