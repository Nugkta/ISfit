# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 11:29:00 2022

@author: pokey

Testing ODE SOLVER is consistent with the analytical solution


"""
import numpy as np
from scipy.integrate import solve_ivp


def find_v1(C_a, C_b, C_c, R_i, V, q_init):  
    F = lambda t, Q: (V - Q/C_a - Q/C_b - Q/C_c)/R_i      #the ODE of the ionic branch
    runtime = 1000
    t_eval = np.arange(0, runtime, 0.1)                      #times at which store in solution
    sol = solve_ivp(F, [0, runtime], [q_init], t_eval=t_eval)     #approximate the solution in interval [0, 10], initial value of q
    #print('the q is', sol.y[0][-1])
    v1 = V - sol.y[0][-1]/C_a  #the voltage connecting to thetransistor
    ####################################
    #now compare to analytica solution
    a = 1/R_i*(1/C_a + 1/C_b + 1/C_c)
    b = V/R_i
    q_a = b/a
    v1_a = V - q_a/C_a
    return v1 , v1_a

s1, s2 = find_v1(1 , 3, 2, 3, 4, 2)
print(s1, s2)







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    