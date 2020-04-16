# -*- coding: utf-8 -*-
"""
@author: chenzhao
dimensionalless power spectrum of 2D fields
input: float field :L=1024,L=1024
output: float array : bins
"""
from header import *

def PowerSpectrum_2D(Kappa):           
    '''

    Parameters
    ----------
    Kappa : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    Kl0 = np.fft.fftshift(np.fft.fft2(Kappa))
    Kl1 = np.abs(Kl0)**2
    
    # bins : TYPE, optional
    # DESCRIPTION. The default is 30.
    bins=30
    # cut bin
    a = bins-1
    b = 1.2*np.log(2*np.pi/mapsize);    # basic frequency
    dk = (np.log(L*np.pi/mapsize)-b)/a;
    p = np.zeros((a+1))
    k = np.zeros((a+1))
    n = np.zeros((a+1))
    
    for i00 in range(L):
        for i01 in range(L):
            r0 =  np.sqrt((i00-L/2)**2 + (i01-L/2)**2)
            if (r0 < L/2) & (r0 != 0):
                l = r0*2*np.pi/mapsize
                i_l = int(np.ceil((np.log(l)-b)/dk))
                # counts
                n[ i_l ] = n[ i_l ] + 1
                # power value
                p[ i_l ] = Kl1[i00,i01] + p[ i_l ]
                # ell value
                k[ i_l ] = np.log(l) + k[ i_l ]
    
    # avoid the divide zero 
    n[n==0] = 1 
    
    p0 = p/n
    k0 = k/n
    p00 = np.zeros((bins))
    k00 = np.zeros((bins))
    
    for r in range(bins):
            k00[r] = (np.exp(k0[r]))
            ## dimensionalless
            p00[r] = mapsize*mapsize*((k00[r]*( k00[r] +1 ) )*p0[r])/(2*np.pi)/(1024**4)
   
 
           
    return(k00,p00)
