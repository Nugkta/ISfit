# A Tool For Impedance Spectroscopy Fitting
This is a step-by-step guide to how to use this impedance spectroscopy fitting tool. 

This file only contains a general introduction and the instructions for the tool.

The fitting model is based on the model of an equivalent circuit; the theory of which **will not** be explained in this readme file, the user should refer to the supplementary material and the original paper[*reffff] for a more detailed explanation. 

## Introduction
This tool is for fitting the data of impedance spectroscopy, taking the complex impedance of the device as the independent variable and the frequency of the small oscillating voltage as the dependent variable.

The tool can be used for 4 scenarios:
1. Doing a global fit on multiple data sets which **contain** one 0 bias voltage set.
2. Doing an individual fit on a single data set with **0** bias voltage.
3. Doing a global fit on multiple data sets that **do not contain** one 0 bias voltage set.
4. Doing an individual fit on a single data set with **non-0** bias voltage.

The fitting results of the different scenarios will give different parameters[ref to paper].

The workflow of the tool is:
1. By plotting out the data in different forms and letting the user choose some critical points of the plots(and guess the R_ion for some cases), an initial guess-finding function will return a set of appropriate initial guesses to be used for the fit later. The explanation of the function can be found in the document[reffff]

2. After getting the initial guesses, the tool will plot the simulated result of the model using those as parameters, illustrated with different plots as well. At this interface, the user will be able to adjust all the parameters by sliders or direct input to improve the initial guess. Also, the user can select the parameters to fix during the fit.

Note: For the cases when the R_ion is needed as a guess by the user, there will be an interface to adjust the R_ion individually. Under that circumstance, the other initial guesses will be indirectly affected by the choice of R_ion due to the initial-guess-finding function needs the value of R_ion to work out the rest.

3. After pressing the button 'Start fitting', the tool will run the fit and print out the result directly in the console including the fitting statistics.

5. The user can choose to use the fitted result as the initial guess and restart the adjusting interface dan do the fitting again. This allows the users to test the effect of adjusting the parameters individually.




## Environment
This tool requires simple environments to run. In addition to installing some packages such as lmfit and glob, the users need to set the graphic of their IDE  to allow pop up windows for interactive plots.

## Preprocessing The Data
This tool requires the user to have the experimental data stored in Excel spreadsheets(.xlsx) with two further requirements.
1. The spreadsheets should contain the data in the form shown as follows. The users are expected to preprocess the data to this form.


    | z_real   | z_imag   | frequency | applied voltage | J_ph | J
    | -        | -        | -         | -               | -    |-
    |   num    |   num    |   num     |     num         |  num |num

    The users are expected to preprocess the data to this form. The meaning of the column headers are:
    * z_real: The real part of the impedance of the device.
    * z_imag: The imaginary part of the impedance of the device.
    * frequency: The frequency of the applied bias voltage.
    * applied voltage: The applied background voltage to the circuit
    * J_ph: The photo generation current. In principle should be obtained by the fractional intensity of the sunlight and a reference current of 1 sun intensity.
    * J: The current in the circuit.



2. The spreadsheets should be stored in a folder that is under the same directory as the Main file, for the automatic data loading function to work.



## Running The Tool
1. After forking the repository from GitHub. Make sure that the IDE is opening the correct directory that contains all the files of the functions.
2. Open and run the Main8.py to load in all the modules and functions for the tool to work.
3. Run main() in the console and follow the instructions.

    __Note that__: for the case of individual_no0V and global_no0V fit, after selecting the proper points in the pop up plots, the users have to enter the __guessed value of R_ion__ before continuing.
4. The results, including the status of the fitted parameters, the resultant values of fitting, and other statistics will be printed in the console.



### List of Important Variables:
* z_real: The real part of the impedance of the device.
* z_imag: The imaginary part of the impedance of the device.
* J_ph: The photo generation current. In principle should be obtained by the fractional intensity of the sunlight and a reference current of 1 sun intensity.
* J: The current in the circuit.
* C_A: The capacitance of the first interface.
* C_ion: The capacitance of the ionic branch(not including the geometric capacitance).
* C_g: The geometic capacitance.
* J_s: The saturation current.
* nA: The ideality factor of the transistor in the model.
* R_s: The series resistor.
* R_shnt: The shunt resistance.
* J_nA: = J_s /n_A





























