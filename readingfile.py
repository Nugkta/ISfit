# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:09:58 2022

@author: hx5118
THIS file is for reading the .txt impedance spectrum data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file = open('1088_201020EISScan1B40549.txt', 'r')
flag = 0
index = 0
string = 'primary_data'
for line in file:
    index += 1
    if string in line:
        flag = 1
        break
print(index)
    





#df = pd.read_csv('1088_201020EISScan1B40549.txt',delimiter = ' ' or '  ', skiprows = 173, nrows = 39 )
z_real, z_imag, freq = np.loadtxt('1088_201020EISScan1B40549.txt', skiprows = 173, max_rows = 40, unpack = True)
plt.plot(z_real, -z_imag, '.')


































