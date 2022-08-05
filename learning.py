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





























