# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 23:33:09 2022

@author: pokey
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
    
def name_df_wz(df):       #note the sort by z_real is done here
    headers = df.columns
    df = df.rename(columns = {headers[0]: 'z_real', headers[1]: 'z_imag',headers[2] : 'frequency'})
    #df.plot(x = 'z_real', y = 'z_imag',kind = 'scatter')
    plt.plot(df['z_real'],-df['z_imag'],'.')
    df = df.sort_values(by=['z_real'])
    return df

def name_df_jv(df):
    headers = df.columns
    df = df.rename(columns = {headers[0]: 'time', headers[1]: 'current',headers[2] : 'voltage'})
    df.plot(x = 'time', y = 'current',kind = 'scatter')
    #df = df.sort_values(by=['z_real'])
    return df



def read_as_df(file_name , skip , data_num ):     #the function that clean up the read data and sort by Z_real from small to big and plot the Nyquist plot
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
    # df_cleaned = df_cleaned.rename(columns = {headers[0]: 'z_real', headers[1]: 'z_imag',headers[2] : 'frequency'})
    # df_cleaned.plot(x = 'z_real', y = 'z_imag',kind = 'scatter')
    # df_cleaned = df_cleaned.sort_values(by=['z_real'])
    #plt.xscale('log')
    return df_cleaned

def read_to_wz(file_name , skip , data_num):
   df = read_as_df(file_name , skip , data_num )
   df = name_df_wz(df)
   return df

def read_to_jv(file_name , skip , data_num):
   df = read_as_df(file_name , skip , data_num )
   df = name_df_jv(df)
   return df


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


#def read_jv_file(file_name):
    



def read_imp_file(file_name, tp): #the function that integrates the above functions, read a specific data file and return a dataframe storing z_r vs z_i vs frequency 
    skip , data_num = parse_file(file_name)
    if tp == 'wz':                  #in order to distinguish wz data from the jv data
        df1 = read_to_wz(file_name,skip , data_num)
    if tp == 'jv':
        df1 = read_to_jv(file_name,skip , data_num)
    return df1 


def comb_z(dfs):
    for df in dfs:
        z_real = np.array(df['z_real'].values)
        z_imag = np.array(df['z_imag'].values)
        df['impedance'] =  z_real + z_imag*1j
        df.drop(columns=['z_real', 'z_imag'],inplace = True)
        df = df[['impedance' , 'frequency' , 'recomb current' , 'bias voltage']]
    return dfs 




def read_imp_folder(folder_path): #the function that find all the .txt file in a specific folder and then implement the read to df function.
    txtfiles = []
    dfs = []
    for file in glob.glob(folder_path + '*.txt'):
        txtfiles.append(file)
    print(glob.glob(folder_path + '*.txt'))
    # for i in txtfiles : 
    #     if 'TRScan' in i:
    #         df = read_imp_file(i,'jv')
    #     if 'EISScan' in i:
    #         df = read_imp_file(i,'wz')
    n = 0
    #for n in range(0 , len(txtfiles)-1):             #becaause the wz and jv file appear in pairs with jv file in front
    while n < len(txtfiles):    
        df_jv = read_imp_file(txtfiles[n],'jv')
        df_wz = read_imp_file(txtfiles[n+1],'wz')
        V = np.mean(df_jv['voltage'].values[-10:])
        J1 = np.mean(df_jv['current'].values[-10:])
        vlist = np.ones(len(df_wz['frequency'])) * V
        jlist = np.ones(len(df_wz['frequency'])) * J1
        df_wz['bias voltage'] = vlist 
        df_wz['recomb current'] = jlist 
        n+=2
        dfs.append(df_wz)
    comb_z(dfs)
    
    return dfs







#%% THE MAIN PROCEDURE IS HERE
#C:/Users/pokey/Documents/UROP/cir_simu/*.txt

#dfs = read_imp_folder('C:/Users/pokey/Documents/UROP/Humidity Dark Impedance Data/MAPI\MAPIdev2p5LiI45C/')
#dfs = read_imp_folder('C:/Users/pokey/Documents/UROP\Humidity Dark Impedance Data\MAPI\MAPIdev2p5MgCl30C/')
#dfs = read_imp_folder('/MAPIdev2p5DRIEDLiI25C/')




































