import numpy as np
import climlab
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import sys

# Load mean p, T, q
ref_p = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy') / 100  # (20, )
ref_T = np.load('/data92/garywu/2025_summer/dycore/npy_files/sst_2.5K_t_mean.npy')        # (20, 64, 128)
ref_q = np.load('/data92/garywu/2025_summer/dycore/npy_files/sst_2.5K_q_mean.npy')        # (20, 64, 128)

# Latitude and surface temperature
nlev, nlat, nlon = 20, 64, 128
lat = np.linspace(-90, 90, nlat)
lon = np.linspace(0, 360, nlon, endpoint=False)
θc = np.deg2rad(lat)
T_surf = 29. * np.exp(-(θc**2) / (2 * (26. * np.pi / 180.)**2)) + 271.

# Pre-allocate full LRF array (lat, p_resp, p_perturb)
LRF_SW_q, LRF_LW_q = np.zeros((nlat, nlev, nlev)), np.zeros((nlat, nlev, nlev))
LRF_SW_T, LRF_LW_T = np.zeros((nlat, nlev, nlev)), np.zeros((nlat, nlev, nlev))
SW_ref, LW_ref = np.zeros((nlev, nlat)), np.zeros((nlev, nlat))

# Single-latitude kernel function
def LRF_from_q(p, T, q, T_sfc):
    nlev = p.size
    state = climlab.column_state(lev=p, water_depth=1.0)
    state['Tatm'][:] = T
    state['Ts'][:] = T_sfc
    qs = climlab.utils.thermo.qsat(T, p)

    rad = climlab.radiation.RRTMG(name='rad_base', state=state,
                                  specific_humidity=q, albedo=0.3)
    rad.compute_diagnostics()
    LW_ref = rad.diagnostics['TdotLW'].copy()
    SW_ref = rad.diagnostics['TdotSW'].copy()

    kernel_LW = np.zeros((nlev, nlev))
    kernel_SW = np.zeros((nlev, nlev))

    for k in range(nlev):
        q_pert = q.copy()
        dq = qs[k] * -0.2
        q_pert[k] += dq

        rad_pert = climlab.radiation.RRTMG(name=f'Rad_pert_{k}',
                                           state=state,
                                           specific_humidity=q_pert,
                                           albedo=0.3)
        rad_pert.compute_diagnostics()
        LW_p = rad_pert.diagnostics['TdotLW'].copy()
        SW_p = rad_pert.diagnostics['TdotSW'].copy()

        kernel_LW[:, k] = (LW_p - LW_ref) / dq
        kernel_SW[:, k] = (SW_p - SW_ref) / dq

    return kernel_LW, kernel_SW


def LRF_from_T(p, T, q, T_sfc):
    nlev = p.size
    state = climlab.column_state(lev=p, water_depth=1.0)
    state['Tatm'][:] = T
    state['Ts'][:] = T_sfc
    
    rad_base = climlab.radiation.RRTMG(name='Rad_base',
                                       state=state,
                                       specific_humidity=q,
                                       albedo=0.3)

    rad_base.compute_diagnostics()
    LW_ref = rad_base.diagnostics['TdotLW'].copy()
    SW_ref = rad_base.diagnostics['TdotSW'].copy()

    kernel_LW = np.zeros((nlev, nlev))
    kernel_SW = np.zeros((nlev, nlev))
    
    dT = 1
    state_pert = climlab.column_state(lev=p, water_depth=1.0)
    state_pert['Tatm'][:] = T
    state_pert['Ts'][:]   = T_sfc

    for k in range(nlev):
        # Temp perturbation at k lev
        tmp = T.copy()
        tmp[k] += dT
        state_pert['Tatm'][:] = tmp

        # Fresh radiation model per perturbation
        rad_pert = climlab.radiation.RRTMG(name=f'Rad_pert_{k}',
                                        state=state_pert,
                                        specific_humidity=q,
                                        albedo=0.3)
        rad_pert.compute_diagnostics()
        LW_pert = rad_pert.diagnostics['TdotLW'].copy()
        SW_pert = rad_pert.diagnostics['TdotSW'].copy()

        # Compute kernel column (response at all levels to impulse at level k)
        kernel_LW[:, k] = (LW_pert - LW_ref) / dT
        kernel_SW[:, k] = (SW_pert - SW_ref) / dT
    return kernel_LW, kernel_SW




def calc_reference_heating(p, T, q, T_sfc):
    nlev = p.size
    state = climlab.column_state(lev=p, water_depth=1.0)
    state['Tatm'][:] = T
    state['Ts'][:] = T_sfc

    rad = climlab.radiation.RRTMG(name='rad_base', state=state,
                                  specific_humidity=q, albedo=0.3)
    rad.compute_diagnostics()
    LW_ref = rad.diagnostics['TdotLW'].copy()
    SW_ref = rad.diagnostics['TdotSW'].copy()
    return LW_ref, SW_ref



# Main execution
if __name__ == '__main__':

    for j in range(32, 64):
        LW_ref_tmp, SW_ref_tmp = calc_reference_heating(ref_p, ref_T[:, j, 0], ref_q[:, j, 0], T_surf[j])
        kernel_LW_q_tmp, kernel_SW_q_tmp = LRF_from_q(ref_p, ref_T[:, j, 0], ref_q[:, j, 0], T_surf[j])
        # kernel_LW_T_tmp, kernel_SW_T_tmp = LRF_from_T(ref_p, ref_T[:, j, 0], ref_q[:, j, 0], T_surf[j])  
        LRF_SW_q[j, 4:, 4:] = kernel_SW_q_tmp[4:, 4:]
        LRF_LW_q[j, 4:, 4:] = kernel_LW_q_tmp[4:, 4:]
        # LRF_SW_T[j, :, :] = kernel_SW_T_tmp
        # LRF_LW_T[j, :, :] = kernel_LW_T_tmp
        LW_ref[:, j] = LW_ref_tmp
        SW_ref[:, j] = SW_ref_tmp
        print(f'Finished latitude index {j} (lat {lat[j]:.3f}°)')

    # tile reference heating
    LW_ref = np.transpose(np.tile(LW_ref, (128, 1, 1)), (1, 2, 0))
    SW_ref = np.transpose(np.tile(SW_ref, (128, 1, 1)), (1, 2, 0))

    # Mirror Southern Hemisphere
    LRF_SW_q[:32, :, :] = np.flip(LRF_SW_q[32:, :, :], axis=0)
    LRF_LW_q[:32, :, :] = np.flip(LRF_LW_q[32:, :, :], axis=0)
    # LRF_SW_T[:32, :, :] = np.flip(LRF_SW_T[32:, :, :], axis=0)
    # LRF_LW_T[:32, :, :] = np.flip(LRF_LW_T[32:, :, :], axis=0)
    LW_ref[:, :32, :] = np.flip(LW_ref[:, 32:, :], axis=1)
    SW_ref[:, :32, :] = np.flip(SW_ref[:, 32:, :], axis=1)

    # Save
    with h5py.File("/home/garywu/summer_2025/LRF/npy files/LRF_output_sst_2.5K.dat", "w") as f:
        f.create_dataset("LRF_SW_q", data=LRF_SW_q)
        f.create_dataset("LRF_LW_q", data=LRF_LW_q)
        # f.create_dataset("LRF_SW_T", data=LRF_SW_T)
        # f.create_dataset("LRF_LW_T", data=LRF_LW_T)
        f.create_dataset("ref_heating_LW", data=LW_ref)
        f.create_dataset("ref_heating_SW", data=SW_ref)
        f.create_dataset("ref_T", data=ref_T)
        f.create_dataset("ref_q", data=ref_q)