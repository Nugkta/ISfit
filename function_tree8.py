# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 02:31:44 2022

@author: pokey
This file contains the functions for different scenerios.


"""
import pero_model_fit8 as pmf
import init_guess_plot8 as igp 



def individual_no0V(dfs):
    '''
    This is for the case of individual fit and V is not 0.
    this function is the same as the original only version of fitting scenerio, so
    here I directly used the previously written function
    ''' 
    mode = 'ind_no0V'
    df = dfs[-1]#only uses the last plot to find the initial guess (becasue it has the stable shape)
    crit_points = igp.find_point(dfs[-1]) 
    ig = igp.init_guess_find(dfs,crit_points = crit_points,mode = mode) 
    init_guess = igp.init_guess_class()
    init_guess.update_all(ig, mode = mode)
    igp.R_ion_Slider(init_guess, dfs,crit_points, mode = mode)


def global_no0V(dfs):
    '''
    This is for the case of global fit and no V = 0 data
    Could also be adapted directly from the previous main function
    '''
    mode = 'glob_no0V'
    df = dfs[-1]#only uses the last plot to find the initial guess (becasue it has the stable shape)
    Vb_max = df['bias voltage'][0]
    crit_points = igp.find_point(dfs[-1]) 
    k = crit_points[1] / crit_points[0]
    nA_e , J_s_e = igp.find_nA_Js(dfs, k, mode = mode)
    print('A different method(different from the built-in method in the following steps) gives estimation of nA and J_s to  %.3e %.3e'%( J_s_e,nA_e))
    ig = igp.init_guess_find(dfs,crit_points = crit_points, mode = mode) 
    init_guess = igp.init_guess_class()
    init_guess.update_all(ig, mode = mode)
    igp.R_ion_Slider(init_guess, dfs,crit_points, mode = mode,Vb_max = Vb_max)


def individual_0V(dfs):
    
    mode = 'ind_0V'
    df = dfs[0]
    ig = igp.init_guess_find(dfs, mode = mode)
    init_guess = igp.init_guess_class()
    init_guess.update_all(ig, mode = mode)
    igp.all_param_sliders(event = None, init_guess = init_guess, dfs =dfs, mode = mode)
    # print('----------',init_guess.J_nA)
    # result = pmf.global_fit([df], init_guess, mode = 1)
    # report_fit(result)
    # result_dict = result.params.valuesdict()
    # #putting the resultant parameters into the popt list
    # popt = []
    # for key in result_dict:
    #     popt.append( result_dict[key])
    # dfs= [df]
    # igp.plot_comp(popt,init_guess, dfs, mode = 1 )


def global_0V(dfs):
    mode = 'glob_0V'
    df = dfs[-1]#only uses the last plot to find the initial guess (becasue it has the stable shape)
    crit_points = igp.find_point(dfs[-1]) 
    k = crit_points[1] / crit_points[0]
    nA_e , J_s_e = igp.find_nA_Js(dfs, k, mode = mode)
    print('A different method(different from the built-in method in the following steps) gives estimation of nA and J_s to be %.3e %.3e'%( J_s_e, nA_e))
    Vb_max = df['bias voltage'][0]
    # df = dfs[-1]#only uses the last plot to find the initial guess (becasue it has the stable shape)
    ig = igp.init_guess_find(dfs,crit_points = crit_points,V0 = True, df_0V = dfs[0], mode = mode) 
    init_guess = igp.init_guess_class()
    init_guess.update_all(ig, mode = mode)
    igp.R_ion_Slider(init_guess, dfs,crit_points = crit_points, mode = mode,Vb_max = Vb_max)
    return 























































