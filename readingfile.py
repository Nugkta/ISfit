# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:09:58 2022

@author: hx5118
THIS file is for reading the .txt impedance spectrum data
"""
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
# file = open('1088_201020EISScan1B40549.txt', 'r')
# flag = 0
# index = 0
# string = 'primary_data'
# for line in file:
#     index += 1
#     if string in line:
#         flag = 1
#         break
# print(index)
    



def read_as_df(file_name , skip =171 , data_num = 41):     #the function that clean up the read data and sort by Z_real from small to big and plot the Nyquist plot
    df = pd.read_csv(file_name,sep=' ', engine="python", skiprows = skip, nrows = data_num,encoding='latin-1')
    headers = df.columns
    for i in range(0,len(df[headers[0]])):
        for j in range(0,3):
            element = df.iloc[[i]][headers[j]].values[0]
            order = 1
            while np.isnan(element):
                temp = df.loc[[i]][headers[j]].values[0]
                df.at[i,headers[j]] = df.iloc[[i]][headers[j+order]].values[0]
                df.at[i,headers[j+order]] = temp
                order+=1
                element = df.iloc[[i]][headers[j]].values[0]
    df_cleaned = df[[headers[0], headers[1],headers[2]]]
    #.rename(columns = {headers[0]:'frequency', headers[1]:'zimag',headers[2]:'zreal'}, inplace = True)
    # z_real, z_imag, freq = np.loadtxt('1088_201020EISScan1B40549.txt', skiprows = 173, max_rows = 40, unpack = True)
    # plt.plot(z_real, -z_imag, '.')
    df_cleaned = df_cleaned.rename(columns = {headers[0]: 'z_real', headers[1]: 'z_imag',headers[2] : 'frequency'})
    df_cleaned.plot(x = 'z_real', y = 'z_imag',kind = 'scatter')
    df_cleaned = df_cleaned.sort_values(by=['z_real'])
    #plt.xscale('log')
    return df_cleaned



def parse_file(file_name):      #the function that parse the file and return the rows to skip and the number of rows of data
    file = open(file_name, 'r',encoding='latin1')
    #flag = 0
    file_lines = file.readlines()
    index = 0
    string = 'primary_data'
    for line in file_lines:
        index += 1
        if string in line:
           # flag = 1
            break
    file.close()
    data_num = float(file_lines[index+1])
    skip = index + 1
    return skip , data_num


def read_imp_file(file_name): #the function that integrates the above functions, read a specific data file and return a dataframe storing z_r vs z_i vs frequency 
    skip , data_num = parse_file(file_name)
    df1 = read_as_df(file_name,skip , data_num)
    return df1 


def read_imp_folder(folder_path): #the function that find all the .txt file in a specific folder and then implement the read to df function.
    txtfiles = []
    dfs = []
    for file in glob.glob(folder_path + '*.txt'):
        txtfiles.append(file)
    print(glob.glob(folder_path + '*.txt'))
    for i in txtfiles : 
        df = read_imp_file(i)
        dfs.append(df)
    return dfs







#%% THE MAIN PROCEDURE IS HERE
#C:/Users/pokey/Documents/UROP/cir_simu/*.txt

#dfs = read_imp_folder('/Users/pokey/Documents/GitHub/perovskite_circuit')
dfs = read_imp_folder('/MAPIdev2p5DRIEDLiI25C/')


























#%%
skip , data_num = parse_file('1088_201020EISScan1B40549.txt')
df1 = read_as_df('1088_201020EISScan1B40549.txt',skip , data_num)




#%%
df = pd.read_csv('1088_201020EISScan1B40549.txt',sep=' ', engine="python", skiprows = 171, nrows = 41,encoding='latin-1')

headers = df.columns
for i in range(0,len(df[headers[0]])):
    for j in range(0,3):
        element = df.iloc[[i]][headers[j]].values[0]
        order = 1
        while np.isnan(element):
            temp = df.loc[[i]][headers[j]].values[0]
            df.at[i,headers[j]] = df.iloc[[i]][headers[j+order]].values[0]
            df.at[i,headers[j+order]] = temp
            order+=1
            element = df.iloc[[i]][headers[j]].values[0]
df_cleaned = df[[headers[0], headers[1],headers[2]]]
#.rename(columns = {headers[0]:'frequency', headers[1]:'zimag',headers[2]:'zreal'}, inplace = True)
# z_real, z_imag, freq = np.loadtxt('1088_201020EISScan1B40549.txt', skiprows = 173, max_rows = 40, unpack = True)
# plt.plot(z_real, -z_imag, '.')
df_cleaned = df_cleaned.rename(columns = {headers[0]: 'z_real', headers[1]: 'z_imag',headers[2] : 'frequency'})

df_cleaned.plot(x = 'z_real', y = 'z_imag',kind = 'scatter')
df_cleaned = df_cleaned.sort_values(by=['z_real'])
#plt.xscale('log')


#%% trying to parse the file and obtain the  number of data and the lines to skip
file = open('1203_201022EISScan1B40549.txt', 'r',encoding='latin1')
flag = 0
file_lines = file.readlines()
index = 0
string = 'primary_data'
for line in file_lines:
    index += 1
    if string in line:
        flag = 1
        break
file.close()

data_num = float(file_lines[index+1])
skip = index + 1
df_test = read_as_df('1203_201022EISScan1B40549.txt' , skip =skip , data_num = data_num)
#%%
df_test.plot(x = 'z_real', y = 'z_imag',kind = 'scatter')













































































