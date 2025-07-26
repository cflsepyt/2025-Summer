import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

# functions for reading dycore data
def read_Dycore_data(filepath, print_var=False):
    with h5py.File(filepath, "r") as f:
        if print_var:
            print("Available variables:", list(f.keys()))
        q   = f["grid_tracers_c_xyzt"][:, :, :, :]
        u   = f["grid_u_c_xyzt"][:, :, :, :]
        w   = f["grid_w_full_xyzt"][:, :, :, :]
    return q, u, w

def _extract_day(fp):
    m = re.search(r"_startfrom_(\d+)day", fp)
    return int(m.group(1)) if m else -1

def read_dycore_series(pattern):
    # find & sort
    files = sorted(glob.glob(pattern), key=_extract_day)[40:-1]

    q_all = []
    u_all = []
    w_all = []
    for i, fp in enumerate(files):
        q_tmp, u_tmp, w_tmp = read_Dycore_data(fp, print_var=(i==0))
        q_all.append(q_tmp)
        u_all.append(u_tmp)
        w_all.append(w_tmp)
        print(f'finished reading {fp}')

    q_all   = np.concatenate(q_all, axis=0)
    u_all   = np.concatenate(u_all, axis=0)
    w_all   = np.concatenate(w_all, axis=0)
    return q_all, u_all, w_all


def calc_composite(data, var, lat_lim):
    nt, nlev, nlat, nlon = var.shape[0], var.shape[1], var.shape[2], var.shape[3]
    # get maximum cwp index
    t0_idx, lon_idx = np.unravel_index(
        np.nanargmax(data), data.shape
        )

    # define window
    steps_per_day = 24 // 6
    window_days  = 15
    window_steps = window_days * steps_per_day
    win = np.arange(t0_idx - window_steps,
                    t0_idx + window_steps + 1)
    win = win[(win >= 0) & (win < nt)]
    day_offset = (win - t0_idx) * (6 / 24)

    # get data from the window
    var_win = var[win, ...]           # shape (n_win, nlev, nlat, nlon)
    var_sel = var_win[:, :, lat_lim, lon_idx]
    
    # (n_win, nlev)
    var_comp  = np.nanmean(var_sel, axis=2)
    var_comp -= var_comp.mean(axis=0)
    return day_offset, var_comp.T


def main():
    # lat, lon and lat_lim
    lon = np.linspace(0, 360, 128, endpoint=False)
    lat = np.linspace(-90, 90, 64)
    lat_lim = np.where( ( lat >= -15.0 ) & ( lat <= 15.0 ) )[0]

    # read dycore data
    pattern = "/data92/Quark/ctrl_2000d/HSt42_20/RH80_PR20_2000day_startfrom_*.dat"
    case_name = "control_run"
    q, u, w = read_dycore_series(pattern)
    nt, nlev, nlat, nlon = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    print('finished loading data, shape:', q.shape)

    # load pressure coord.
    p = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy')

    # calculate cwp
    cwp = np.trapz(q, p, axis=1) / 9.8
    cwp_eq = cwp[:, lat_lim, :].mean(axis=1)

    # calculate composite
    day_offset, q_comp = calc_composite(cwp_eq, q, lat_lim)
    _, u_comp = calc_composite(cwp_eq, u, lat_lim)
    _, w_comp = calc_composite(cwp_eq, w, lat_lim)
    print('composite finished, shape:', q_comp.shape)


    # plot
    t = np.linspace(-15, 15, 120)

    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(day_offset, p/100, q_comp*1000, cmap='BrBG', levels=np.linspace(-2, 2, 21), extend='both')

    skip = 3
    quiver = ax.quiver(day_offset[::skip], p/100, u_comp[:, ::skip], -w_comp[:, ::skip] * 100, scale=200)
    
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('days')
    ax.set_ylabel('hPa')
    ax.set_title('Composite qv and (u, w) anomaly')
    cbar = fig.colorbar(cf, ax=ax, shrink=0.7)
    cbar.ax.set_title('g/kg')

    plt.tight_layout()
    plt.savefig(f'/home/garywu/summer_2025/dycore/figures/comp_{case_name}.png', dpi=300, bbox_inches='tight')
    return

if __name__ == '__main__':
    main()