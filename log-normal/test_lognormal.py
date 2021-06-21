from nbodykit.source.catalog.file import BigFileCatalog
from nbodykit.lab import FFTPower
from nbodykit.lab import FFTCorr
from scipy.interpolate import interp1d
from nbodykit import cosmology
from functools import partial
import camb
from camb import model
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time

t1 = time.time()

cata_fa = BigFileCatalog('/home/yuyu22/testfast/B2T10/snapshot/fastpm_1.0000/',dataset='1/',header='Header/')
cata_fa.attrs

grid_mesh = 1024

print("Nmesh = ", grid_mesh)
Boxsize = 500
mesh_fa = cata_fa.to_mesh(Nmesh=grid_mesh, BoxSize=Boxsize, resampler='cic')
mesh_fa = mesh_fa.compute()
print(mesh_fa.shape)

# xi_fa = FFTCorr(mesh_fa, mode='1d')

redshift = 0.0;
cosmo = cosmology.Cosmology(h=0.6898, P_k_max=200).match(Omega0_m=0.2905)
pk_cosmo = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')

def VectorProjection(vector, direction):
    r"""
    Vector components of given vectors in a given direction.

    .. math::

        \mathbf{v}_\mathbf{d} &= (\mathbf{v} \cdot \hat{\mathbf{d}}) \hat{\mathbf{d}} \\
        \hat{\mathbf{d}} &= \frac{\mathbf{d}}{\|\mathbf{d}\|}

    Parameters
    ----------
    vector : array_like, (..., D)
        array of vectors to be projected
    direction : array_like, (D,)
        projection direction. It does not have to be normalized

    Returns
    -------
    projection : array_like, (..., D)
        vector components of the given vectors in the given direction
    """
    direction = np.asarray(direction, dtype='f8')
    direction = direction / (direction ** 2).sum() ** 0.5
    projection = (vector * direction).sum(axis=-1)
    projection = projection[:, None] * direction[None, :]

    return projection


# apply RSD along the z axis
line_of_sight = [0,0,1]
# the RSD normalization factor
rsd_factor = (1+redshift) / (100 * cosmo.efunc(redshift))
# update Position, applying RSD
cata_fa['Position'] = cata_fa['Position'] + rsd_factor * VectorProjection(cata_fa['Velocity'], line_of_sight)

mesh_rsd_fa = cata_fa.to_mesh(Nmesh=grid_mesh, BoxSize=Boxsize,resampler='cic')
mesh_rsd_fa = mesh_rsd_fa.compute()
print(mesh_rsd_fa.shape)
print(cata_fa['Velocity'].min().compute(), cata_fa['Velocity'].max().compute(), cata_fa['Velocity'].mean().compute())

del cata_fa

pk_fa = FFTPower(mesh_fa, mode='1d')
pk_rsd_fa = FFTPower(mesh_rsd_fa, mode='1d')
np.savetxt('powerspec_test/pk_fa.txt', np.array([pk_fa.power['k'][1:].real , pk_fa.power['power'][1:].real]).T)
np.savetxt('powerspec_test/pk_rsd_fa.txt', np.array([pk_rsd_fa.power['k'][1:].real , pk_rsd_fa.power['power'][1:].real]).T)

t2 = time.time()
print("time1 used: {:.1} min".format((t2-t1)/60))

### psi_x = i*kx/k^2*delta
### delta(x, y, z)

def delta2psi(delta):
          
    delta_k = np.fft.fftshift(np.fft.fftn(delta))    ### -kmax:0:kmax-1
    imag_unit = 1j
    psi = np.zeros(shape=(3, grid_mesh, grid_mesh, grid_mesh), dtype=np.complex)
     
    #### space for time
    k_x, k_y, k_z = np.meshgrid(np.arange(-int(grid_mesh/2), int(grid_mesh/2), 1),
                                            np.arange(-int(grid_mesh/2), int(grid_mesh/2), 1),
                                            np.arange(-int(grid_mesh/2), int(grid_mesh/2), 1))
    
    k_x = k_x / Boxsize; k_y = k_y / Boxsize; k_z = k_z / Boxsize
    k_square = k_x*k_x + k_y*k_y + k_z*k_z
    k_square[k_square==0] = 1
    
    psi[0, :,:,:] = imag_unit*k_x/k_square*delta_k
    psi[1, :,:,:] = imag_unit*k_y/k_square*delta_k
    psi[2, :,:,:] = imag_unit*k_z/k_square*delta_k

    psi[0, :,:,:] = np.fft.ifftn(np.fft.ifftshift(psi[0, :,:,:]))
    psi[1, :,:,:] = np.fft.ifftn(np.fft.ifftshift(psi[1, :,:,:]))
    psi[2, :,:,:] = np.fft.ifftn(np.fft.ifftshift(psi[2, :,:,:]))
    
    return psi.real

def vel_mesh2particle(pos, mesh, Boxsize, resample='cic'):
    ### from velocity field to velocity catalog of particles
    ### pos[n,3]
    ### mesh[Nmesh, Nmesh, Nmesh],  Boxsize: Mpc\h
    Nmesh = mesh.shape[-1]; n_par = pos.shape[0]
    factor = Nmesh/Boxsize
    pos = pos*factor 
    vel = np.zeros_like(pos)
    
    for i_par in range(n_par):
        
        i_xm = int(np.floor(pos[i_par, 0])) % Nmesh-1; i_xp = int(np.floor(pos[i_par, 0])+1) % Nmesh-1; 
        i_ym = int(np.floor(pos[i_par, 1])) % Nmesh-1; i_yp = int(np.floor(pos[i_par, 1])+1) % Nmesh-1;
        i_zm = int(np.floor(pos[i_par, 2])) % Nmesh-1; i_zp = int(np.floor(pos[i_par, 2])+1) % Nmesh-1;

        weight_x = 1 - (pos[i_par, 0] - i_xm)
        weight_y = 1 - (pos[i_par, 1] - i_ym)
        weight_z = 1 - (pos[i_par, 2] - i_zm)
        if resample=='ngp':
            if weight_x > 0.5:
                i_x = i_xm
            else:
                i_x = i_xp
            if weight_y > 0.5:
                i_y = i_ym
            else:
                i_y = i_yp
            if weight_z > 0.5:
                i_z = i_zm
            else:
                i_z = i_zp
            
            vel[i_par, 0] = mesh[0, i_x, i_y, i_z]
            vel[i_par, 1] = mesh[1, i_x, i_y, i_z]
            vel[i_par, 2] = mesh[2, i_x, i_y, i_z]
            
        elif resample=='cic':      
            ### x direction
            vel[i_par, 0] = mesh[0, i_xm, i_ym, i_zm]*weight_x*weight_y*weight_z + \
                            mesh[0, i_xm, i_yp, i_zm]*weight_x*(1-weight_y)*weight_z + \
                            mesh[0, i_xm, i_ym, i_zp]*weight_x*weight_y*(1-weight_z) + \
                            mesh[0, i_xm, i_yp, i_zp]*weight_x*(1-weight_y)*(1-weight_z) + \
                            mesh[0, i_xp, i_ym, i_zm]*(1-weight_x)*weight_y*weight_z + \
                            mesh[0, i_xp, i_yp, i_zm]*(1-weight_x)*(1-weight_y)*weight_z + \
                            mesh[0, i_xp, i_ym, i_zp]*(1-weight_x)*weight_y*(1-weight_z) + \
                            mesh[0, i_xp, i_yp, i_zp]*(1-weight_x)*(1-weight_y)*(1-weight_z)
            ### y direction
            vel[i_par, 1] = mesh[1, i_xm, i_ym, i_zm]*weight_x*weight_y*weight_z + \
                            mesh[1, i_xm, i_yp, i_zm]*weight_x*(1-weight_y)*weight_z + \
                            mesh[1, i_xm, i_ym, i_zp]*weight_x*weight_y*(1-weight_z) + \
                            mesh[1, i_xm, i_yp, i_zp]*weight_x*(1-weight_y)*(1-weight_z) + \
                            mesh[1, i_xp, i_ym, i_zm]*(1-weight_x)*weight_y*weight_z + \
                            mesh[1, i_xp, i_yp, i_zm]*(1-weight_x)*(1-weight_y)*weight_z + \
                            mesh[1, i_xp, i_ym, i_zp]*(1-weight_x)*weight_y*(1-weight_z) + \
                            mesh[1, i_xp, i_yp, i_zp]*(1-weight_x)*(1-weight_y)*(1-weight_z) 
            ### x direction
            vel[i_par, 2] = mesh[2, i_xm, i_ym, i_zm]*weight_x*weight_y*weight_z + \
                            mesh[2, i_xm, i_yp, i_zm]*weight_x*(1-weight_y)*weight_z + \
                            mesh[2, i_xm, i_ym, i_zp]*weight_x*weight_y*(1-weight_z) + \
                            mesh[2, i_xm, i_yp, i_zp]*weight_x*(1-weight_y)*(1-weight_z) + \
                            mesh[2, i_xp, i_ym, i_zm]*(1-weight_x)*weight_y*weight_z + \
                            mesh[2, i_xp, i_yp, i_zm]*(1-weight_x)*(1-weight_y)*weight_z + \
                            mesh[2, i_xp, i_ym, i_zp]*(1-weight_x)*weight_y*(1-weight_z) + \
                            mesh[2, i_xp, i_yp, i_zp]*(1-weight_x)*(1-weight_y)*(1-weight_z)
            
    return vel    

def vel_mesh2particle_for_para(i_core):
    return vel_mesh2particle(pos_fa[int(i_core*dnum):int((i_core+1)*dnum), :], 
                             mesh=psi1_lin_fa, Boxsize=Boxsize, resample='cic')

cata_fa = BigFileCatalog('/home/yuyu22/testfast/B2T10/snapshot/fastpm_1.0000/',dataset='1/',header='Header/')
pos_fa = cata_fa['Position'].compute()

f = cosmo.scale_independent_growth_rate(redshift)
vel_factor = f*100*cosmo.efunc(redshift)/(1+redshift)   # km/s
psi1_lin_fa = delta2psi(mesh_fa.value)*vel_factor

# print("For the vel mesh: ", psi1_lin_fa.min(), psi1_lin_fa.max(), psi1_lin_fa.mean())

ncores = 8 ## must be the 2^n
dnum = int(pos_fa.shape[0]/ncores)
pool_for = multiprocessing.Pool(ncores)
vel_fa_cata = np.array(pool_for.map(vel_mesh2particle_for_para, range(ncores)))
pool_for.close()
pool_for.join()
cata_vel1_fa = np.vstack(vel_fa_cata)

del psi1_lin_fa
del pos_fa
del vel_fa_cata
print(cata_vel1_fa.shape)
print("For the vel los catalog: ", cata_vel1_fa[:,2].min(), cata_vel1_fa[:,2].max(), cata_vel1_fa[:,2].mean())

# cata_fa = BigFileCatalog('/home/yuyu22/testfast/B2T10/snapshot/fastpm_1.0000/',dataset='1/',header='Header/')

# apply RSD along the z axis
line_of_sight = [0,0,1]
# the RSD normalization factor
rsd_factor = (1+redshift) / (100 * cosmo.efunc(redshift))
# update Position, applying RSD
cata_fa['Position'] = cata_fa['Position'] + rsd_factor * VectorProjection(cata_vel1_fa, line_of_sight)
# cata_fa['Position'] = cata_fa['Position'] + rsd_factor*cata_vel1_fa[:,2]

mesh_vel1_fa = cata_fa.to_mesh(Nmesh=grid_mesh, BoxSize=Boxsize,resampler='cic')
mesh_vel1_fa = mesh_vel1_fa.compute()
del cata_fa
print(np.min(mesh_fa), np.max(mesh_fa), mesh_fa.cmean())
print(np.min(mesh_rsd_fa), np.max(mesh_rsd_fa), mesh_rsd_fa.cmean())
print(np.min(mesh_vel1_fa), np.max(mesh_vel1_fa), mesh_vel1_fa.cmean())

pk_vel1_fa = FFTPower(mesh_vel1_fa, mode='1d')
np.savetxt('powerspec_test/pk_vel1_fa.txt', np.array([pk_vel1_fa.power['k'][1:].real , pk_vel1_fa.power['power'][1:].real]).T)

t3 = time.time()
print("vel1 used: {:.1} min".format((t3-t2)/60))



def vel2_mesh2particle_for_para(i_core):
    return vel_mesh2particle(pos_fa[int(i_core*dnum):int((i_core+1)*dnum), :], 
                             mesh=psi2_lin_fa, Boxsize=Boxsize, resample='cic')

cata_fa = BigFileCatalog('/home/yuyu22/testfast/B2T10/snapshot/fastpm_1.0000/',dataset='1/',header='Header/')
pos_fa = cata_fa['Position'].compute()

f = cosmo.scale_independent_growth_rate(redshift)
vel_factor = f*100*cosmo.efunc(redshift)/(1+redshift)   # km/s
psi2_lin_fa = delta2psi(mesh_fa.value*mesh_fa.value)*vel_factor

# print("For the vel mesh: ", psi2_lin_fa.min(), psi2_lin_fa.max(), psi2_lin_fa.mean())

ncores = 8 ## must be the 2^n
dnum = int(pos_fa.shape[0]/ncores)
pool_for = multiprocessing.Pool(ncores)
vel_fa_cata = np.array(pool_for.map(vel2_mesh2particle_for_para, range(ncores)))
pool_for.close()
pool_for.join()
cata_vel2_fa = np.vstack(vel_fa_cata)

del psi2_lin_fa
del pos_fa
del vel_fa_cata
print(cata_vel2_fa.shape)
print("For the vel los catalog: ", cata_vel2_fa[:,2].min(), cata_vel2_fa[:,2].max(), cata_vel2_fa[:,2].mean())

# apply RSD along the z axis
line_of_sight = [0,0,1]
# the RSD normalization factor
rsd_factor = (1+redshift) / (100 * cosmo.efunc(redshift))
# update Position, applying RSD
cata_fa['Position'] = cata_fa['Position'] + \
                      rsd_factor * VectorProjection((cata_vel1_fa - 2*cata_vel2_fa), line_of_sight)
# cata_fa['Position'] = cata_fa['Position'] + rsd_factor*cata_vel1_fa[:,2]

mesh_vel2_fa = cata_fa.to_mesh(Nmesh=grid_mesh, BoxSize=Boxsize,resampler='cic')
mesh_vel2_fa = mesh_vel2_fa.compute()
del cata_fa
print(np.min(mesh_fa), np.max(mesh_fa), mesh_fa.cmean())
print(np.min(mesh_rsd_fa), np.max(mesh_rsd_fa), mesh_rsd_fa.cmean())
print(np.min(mesh_vel1_fa), np.max(mesh_vel1_fa), mesh_vel1_fa.cmean())
print(np.min(mesh_vel2_fa), np.max(mesh_vel2_fa), mesh_vel2_fa.cmean())

pk_vel2_fa = FFTPower(mesh_vel2_fa, mode='1d')
np.savetxt('powerspec_test/pk_vel2_fa.txt', np.array([pk_vel2_fa.power['k'][1:].real , pk_vel2_fa.power['power'][1:].real]).T)
t4 = time.time()
print("vel2 used: {:.1} min".format((t4-t3)/60))
print("Total used: {:.1} min".format((t4-t1)/60))





