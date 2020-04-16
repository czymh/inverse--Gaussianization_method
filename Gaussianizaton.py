#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chenzhao

The Gaussianization process:
    
    Use the mean CDF of kappa to get the Gaussianized y fields 
    
"""
import numpy as np
import scipy.special as scisp
import scipy.interpolate as scintp


def kappa_y_relation(Kappa):
    '''
    
    Calculate the local transformation: kappa-y relation
    
    Parameters
    ----------
    Kappa : TYPE: kappa fields of (i_f,size_x,size_y) -- 2D or 3D array 
            i_f is the number of fields
        the weak lensing convergence fields

    Returns : kappa-y relation (kappa,y)
    -------
    K_re_mean : TYPE: (size_x*size_y,1) array 
        the x axis of kappa-y relation
    y : TYPE: (size_x*size_y,1) array 
        the y axis of kappa-y relation

    '''
    # get the dimension of kappa fields
    if len(Kappa.shape) == 3:
        i_fs,size_x,size_y = Kappa.shape
    elif len(Kappa.shape) == 2:    
        Kappa = Kappa.reshape(1,*Kappa.shape)
        i_fs,size_x,size_y = Kappa.shape
        
    
    K_re_mean = np.zeros((size_x*size_y,))
    for i_f in range(i_fs):
        K_re_sorted = np.squeeze(np.sort(np.reshape(Kappa[i_f,:,:],(-1,1)),axis=0))
        K_re_mean += K_re_sorted
    ## the x axis of mean CDF of kappa; the y axis is from 0 to 1.
    K_re_mean = K_re_mean/i_fs
    
    ## y = erfinv(2CDF(kappa) − 1).
    x = np.linspace(0.5/size_x/size_y,1-0.5/size_x/size_y,size_x*size_y,endpoint=True)
    y = scisp.erfinv((2*x)-1)
    
    return K_re_mean,y

def Gaussianization(Kappa,k,y):
    '''
    
    Do the Gaussianization process using the mean kappa-y relation

    Parameters
    ----------
    Kappa : TYPE: kappa fields of (i_fs,size_x,size_y) -- array 
            i_f is the number of fields
        the weak lensing convergence fields
    k : TYPE: (size_x*size_y,1) array 
        the x axis of kappa-y relation
    y : TYPE: (size_x*size_y,1) array 
        the y axis of kappa-y relation

    Returns
    -------
    Y : TYPE: the Gaussianized y fields of (i_fs,size_x,size_y) -- array
        the Gaussianized fileds 

    '''
    # get the dimension of kappa fields
    if len(Kappa.shape) == 3:
        i_fs,size_x,size_y = Kappa.shape
    elif len(Kappa.shape) == 2:
        Kappa = Kappa.reshape(1,*Kappa.shape)
        i_fs,size_x,size_y = Kappa.shape
    
    ## interpolate function: kappa-y relation
    k_y_relation = scintp.interp1d(k,y,kind='linear')

    K0_re = np.reshape(Kappa,(-1,1))
    ## deal with the boundary point in the interpolation process
    K0_re[np.where(K0_re>k.max())] = k.max()
    K0_re[np.where(K0_re<k.min())] = k.min()
    
    Y0_re = k_y_relation(K0_re)
    Y = Y0_re.reshape(i_fs,size_x,size_y)
    return Y
 
