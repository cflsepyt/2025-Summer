import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

# functions for reading dycore data
def read_Dycore_data(filepath, print_var=False):
    with h5py.File(filepath, "r") as f:
        if print_var:
            print("Available variables:", list(f.keys()))
        q   = f["grid_tracers_c_xyzt"][:]
        t   = f["grid_t_c_xyzt"][:, :, :, :]
    return q, t

def _extract_day(fp):
    m = re.search(r"_startfrom_(\d+)day", fp)
    return int(m.group(1)) if m else -1

def read_dycore_series(pattern):
    # find & sort
    files = sorted(glob.glob(pattern), key=_extract_day)[10:]

    q_all = []
    t_all = []
    for i, fp in enumerate(files):
        q_tmp, t_tmp = read_Dycore_data(fp, print_var=(i==0))
        q_all.append(q_tmp)
        t_all.append(t_tmp)
        print(f'finished reading {fp}')

    q_all   = np.concatenate(q_all, axis=0)
    t_all   = np.concatenate(t_all, axis=0)
    return q_all, t_all

# function for calculating pressure-weighted mean
def mean_pressure_weighted(p, var, axis=0):
    
    p = np.asarray(p)
    var = np.asarray(var)

    # reshape p
    shape = [1] * var.ndim
    shape[axis] = p.size
    p_bcast = p.reshape(shape)

    # numerator
    num = np.trapz(var * p_bcast, p, axis=axis)
    # denominator
    den = np.trapz(p, p)

    return num / den


def hovmoller_plot(x, y, data, lat_bound):
    data_eq = data[:, lat_bound, :]
    lat_sel = y[lat_bound]

    weights = np.cos(np.deg2rad(lat_sel))  # shape (nlat_sel,)
    weights = weights / weights.sum()

    data_hov = np.average(data_eq, axis=1, weights=weights)
    time = np.arange(data_hov.shape[0])
    plt.pcolormesh(x, time, data_hov)
    plt.savefig(f'/home/garywu/summer_2025/dycore/figures/hovmoller_cwp.png', dpi=300, bbox_inches='tight')
    return

if __name__ == '__main__':
    # lat, lon and lat_lim
    x = np.linspace(0, 360, 128, endpoint=False)
    y = np.linspace(-90, 90, 64)
    lat_lim = np.where( ( y >= -15.0 ) & ( y <= 15.0 ) )[0]

    # read dycore data
    pattern = "/data92/garywu/LRFws/HSt42_20_ws500d_gLRF/data/RH80_L20_1500day_startfrom_*.dat"
    # pattern = "/data92/Quark/LRFws/HSt42_20_ws500d_qLRF/data/RH80_L20_1500day_startfrom_*.dat"
    q, t = read_dycore_series(pattern)
    nt, nlev, nlat, nlon = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    print('finished loading data, shape:', q.shape)

    # load pressure coord.
    p = np.load('/data92/garywu/2025_summer/dycore/npy_files/sst_p_mean.npy')
    
    # calculate column water vapor
    cwp = np.trapz(q, p, axis=1) / 9.8

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(cwp[-1])
    plt.colorbar(mesh)

    plt.tight_layout()
    plt.savefig('/home/garywu/summer_2025/dycore/figures/cwp.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.plot(t.mean(axis=(0, 2, 3)))
    plt.savefig('/home/garywu/summer_2025/dycore/figures/meanT.png', dpi=300, bbox_inches='tight')