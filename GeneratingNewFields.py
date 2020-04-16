#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chenzhao

Generate the new Gaussianized y fields and 
use the kappa-y relation to gain the realizations of weak lensing convergence: 

"""
from header import *


def CalPy_inp(y,bins):
    '''
    Parameters
    ----------
    y : TYPE: y fields of (size_x,size_y) -- array
        the Gaussianized y fields 
    bins : TYPE: int
        the bins of power spectrum
        
    Returns
    -------
    l_y0 : TYPE: (bins,) -- array
        DESCRIPTION.
    P_y0 : TYPE: (bins,) -- array
    '''
    
    size_x,size_y = y.shape
    if size_x != size_y:
        print("Must input a square filed!")    
    
    dl = (size_x/2-1)/(bins)
    n_y = np.zeros((bins))
    p_y = np.zeros((bins))
    l_y = np.zeros((bins))
    # cut bin
    YF_l0 = np.fft.fftshift(np.fft.fft2(y))
    YF_l = np.abs(YF_l0)**2
    for i1 in range(size_x):
        for i2 in range(size_y):            
            l_y0 = np.sqrt((i1-size_x/2)**2 + (i2-size_x/2)**2)
            i_y = int(np.floor((l_y0-1)/dl))
            if l_y0 < size_x/2 and l_y0 != 0 :                
                n_y[i_y] = n_y[i_y] + 1  # count numbers
                l_y[i_y] = l_y[i_y] + l_y0
                p_y[i_y] = p_y[i_y] + YF_l[i1,i2]
    p_y0 = p_y/n_y;l_y0 = l_y/n_y
    ## add the physcial unit
    p_y0 = mapsize*mapsize*p_y0/L**4
                    
    return l_y0,p_y0

def GeneratingNewYFields(Py_inp,l_min,l_max,size_f=L,seed=None):
    '''
    Parameters
    ----------
    Py_inp : TYPE: scipy.interpolate.interpolate.interp1d
        the interploate function of power spectrum of Gaussianized y fields
    l_min : float
        the min value of interploate range
    l_max : float
        the max value of interploate range
    size_f : TYPE, optional
        the size of new generating field. The default is L.
    seed : TYPE, optional
        Random seed . The default is None.

    Returns
    -------
    Y0 : TYPE: (size_f,size_f) -- array
        The new Gaussianized y field

    '''
    
    ## obtain the white noise fields
    Rand_mutiprocessing = np.random.RandomState(seed)   
    WNF = Rand_mutiprocessing.randn(size_f,size_f)
    Nl0 = np.fft.fftshift(np.fft.fft2(WNF))
    
    print("Generating the new y fields of ({:d},{:d}).....".format(size_f,size_f))
    
    Yl0 = np.zeros((size_f,size_f))*1j
    for ix in range(size_f):
        for iy in range(size_f):
            ell = np.sqrt((ix-512)*(ix-512)+(iy-512)*(iy-512))
            if ell <= l_max and ell >= l_min:
                Yl0[ix,iy] = Nl0[ix,iy]*np.sqrt(Py_inp(ell))*(size_f/mapsize)

    Y0 = np.fft.ifft2(np.fft.ifftshift(Yl0)).real
    return Y0

def InverseTrans(Y0,f_trans,y_min,y_max):
    '''
    Parameters
    ----------
    Y0 : TYPE: Y Field (size_Y,size_y) -- array
        the new y field.
    f_trans : TYPE: scipy.interpolate.interpolate.interp1d
        the inverse-transformation of kappa-y relation.
    y_min : TYPE: float
        the min value of y.
    y_max : TYPE: float
        the max value of y.

    Returns
    -------
    K0 : TYPE: (size_Y,size_Y) -- array
        the new kappa field.

    '''
    # input: Y Field (size_Y,size_y)
    size_Y,size_Y = Y0.shape 
    Y0_res = Y0.reshape((size_Y*size_Y))
    Y0_res[Y0_res>y_max] = y_max
    Y0_res[Y0_res<y_min] = y_min
    
    K0_res = f_trans(Y0_res)
    K0 = K0_res.reshape((size_Y,size_Y))
    
    return K0


