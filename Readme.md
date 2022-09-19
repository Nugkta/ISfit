# A Tool For Impedance Spectrocopy Fitting
This is a step-by-step guide to how to use this impedance spectroscopy fitting tool. 

This file only contains the a general introduction and the instructions for the tool.

The fitting model is based on the model of an equivalent circuit; the theory of which **will not** be explained in this readme file, the user should refer to the supplementary material and the original paper[*reffff] for a more detailed explanation. 

## Introduction
tool for fitting data from IS.
contains 4 functions.
built in algorithm to find intial guess [refer to document]
sliders to adjust initial guess
differnt subplot showing all aspect
R_ion slider, interconnect by initial guess algo in xxx and xx case
fix parameter,
refit using fitted parameters.
Statistics of result will be given as well




## Environment
This tool requires simple environments to run. In addition to intalling some packages such as lmfit and glob, the users need to set the graphic of their IDE  to allow pop up windows for interactive plots.

## Preprocessing The Data
This tool requires the user to have the experimental data stored in Excel spreadsheets(.xlsx) with two further requirements.
1. The spreadsheets should contain the data in the form shown as follows. The users are expected to preprocess the data to this form.


    | z_real   | z_imag   | frequency | applied voltage | J_ph | J
    | -        | -        | -         | -               | -    |-
    |   num    |   num    |   num     |     num         |  num |num

    The users are expected to preprocess the data to this form. The meaning of the column headers are:
    * z_real: The real part of the impedance of the device.
    * z_imag: The imaginary  part of the impedance of the device.
    * freqeuncy: The frequency of the applied bias voltage.
    * applied voltage: The applied background voltage to the circuit
    * J_ph: The photo generation current. In principle should be obtained by the fractional intensity of the sun light and a reference current of 1 sun intensity.
    * J: The current in the circuit.



2. The spreadsheets should be stored in a folder that is under the same directory as the Main file, for the automatic data loading function to work.



## Running The Tool
1. After forking the repository from GitHub. Make sure that the IDE is opening the correct directory that contains all the files of the functions.
2. Open and run the Main8.py to load in all the modules and functions for the tool to work.
3. Run main() in the console and follow the instructions.

    __Note that__: for the case of individual_no0V and global_no0V fit, after selecting the proper points in the pop up plots, the user have to to enter the __guessed value of R_ion__ before continuing.
4. The results, including the status of the fitted parameters, the resultant values of fitting and other statistics will be printed in the console.



### List of Variables:
* 






























