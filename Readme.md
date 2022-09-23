# A Tool For Impedance Spectroscopy Fitting
This is a step-by-step guide to how to use this impedance spectroscopy fitting tool. 

This document only contains a general introduction and the instructions for the tool.

The fitting model is based on the model of an equivalent circuit; the theory of which **will not** be explained in this readme file, the user should refer to the supplementary material and the original paper[*reffff] for a more detailed explanation. 

## Introduction
This tool is for fitting the data of impedance spectroscopy, taking the complex impedance of the device as the dependent variable and the frequency of the small oscillating voltage as the independent variable. Additionally the bias voltage, steady-state current and light intensity associated with the measurement are required.

The tool can be used for 4 scenarios:
1. Doing a global fit on multiple data sets which **contain** one 0 bias voltage set.
2. Doing an individual fit on a single data set with **0** bias voltage.
3. Doing a global fit on multiple data sets that **do not contain** one 0 bias voltage set.
4. Doing an individual fit on a single data set with **non-0** bias voltage.

The fitting results of the different scenarios will give the parameters listed at the end of this document[ref to paper].

The workflow of the tool is:
1. Plot the impedance data, allowing the user to choose some critical points on the plots (and to input a guess of the ionic resistance, R_ion, in cases where no data was collected at 0 V bias). Initial guess-finding function returns a set of initial guesses used for the fit. The explanation of the initial guess-finding function can be found in the document[reffff]

2. The tool will plot the simulated impedance data corresponding to the initial guess parameters in an interactive interface. The user can then adjust all the parameters using sliders or directly modifiy values to improve the initial guess. The user selects which parameters to float or fix during the following fit.

Note: For the cases when the R_ion is needed as a guess by the user, there will be an interface to adjust the R_ion individually. Under that circumstance, the other initial guesses will be indirectly affected by the choice of R_ion because the initial-guess-finding function needs the value of R_ion to calculate the other parameters.

3. After pressing the button 'Start fitting', the tool will run the fit and print out the result directly in the console including the fitting statistics.

5. The user can choose to accept the fitted result or use the results as a new set of initial guesses and restart the interactive interface to perform the fitting again. This allows the users to test the effect of adjusting and/or constraining the parameters individually.




## Environment
This tool requires simple environments to run. The following packages must be installed for the software to run: lmfit and glob [list required packages], the users need to set the graphic of their IDE  to allow pop up windows for the interactive plots.

## Preprocessing The Data
At present this tool requires the user to have the experimental data stored in Excel spreadsheets(.xlsx) with two further requirements.
1. The spreadsheets should contain the data in the form shown as follows.


    | Z_real   | Z_imag   | angular frequency | applied voltage | J_ph | J
    |-         | -        | -                 | -               | -    |-
    |   num    |   num    |   num             |     num         |  num |num

    The users are expected to preprocess the data to this form although the column order is not important. The meaning of the column headers are:
    * z_real: The real part of the impedance of the device.
    * z_imag: The imaginary part of the impedance of the device.
    * angular frequency: The angular frequency of voltage oscillation.
    * applied voltage: The applied background voltage to the circuit.
    * J_ph: The photo-generation current density. This can be estimated using fractional intensity of the sunlight and a reference current of 1 sun intensity.
    * J: The steady-state current density through the device.

Note that a seperate spreadsheet is required for each dataset corresponding to a particular applied voltage. Consequently the "applied voltage", "J_ph", and "J" values should be identical throughout the corresponding columns in a given spreadsheet, only the "Z_real", "Z_imag", and "angular frequency" entries will vary from row to row.


2. The spreadsheets should be stored in their own folder in the same directory as the "Main" file to enable the automatic data loading function to work.



## Running The Tool
1. After forking the repository from GitHub. Make sure that the IDE opens the correct directory containing all the function files and data folders.
2. Open and run the Main8.py to load in all the modules and functions for the tool to work.
3. Follow the instructions.

    __Note that__: for the cases of the individual_no0V and global_no0V fits, after selecting the proper points in the pop up plots, the users have to enter the __guessed value of R_ion__ before continuing.
4. The results, including the status (floating or fixed, initial guess, constrained range) of the fitted parameters, the resultant values of fitting, and other statistics (including numerical fit uncertainty assuming the model is correct) will be printed in the console.



### List of Important Input Variables:
* Z_real: The real part of the impedance of the device.
* Z_imag: The imaginary part of the impedance of the device.
* J_ph: The photo generation current density. Can be estimated using the fractional intensity of the sunlight and a reference short circuit current density of 1 sun intensity.
* J: The steady-state current density through the device.

### List of Important Output Variables:
* C_A: The capacitance per unit area of the first interface.
* C_ion: The capacitance per unit area of the ionic branch (not including the geometric capacitance).
* C_g: The geometic capacitance per unit area of the perovskite bulk.
* J_s: The saturation current density of the interface dominating the observed impedance.
* nA: The ideality factor of the interface dominating the observed impedance (represented by a transistor in the model).
* R_s: The areal series resistance [resistance * area].
* R_shnt: The areal shunt resistance [resistance * area].
* J_nA: = The ratio of the saturation current density to ideality factor, J_s/n_A (this ratio cannot be separated in the case of an individual 0 V bias dataset).





























