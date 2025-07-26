import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
from EOF import EOF

# functions for reading dycore data
def read_Dycore_data(filepath, print_var=False, lat_lim=None):
    with h5py.File(filepath, "r") as f:
        if print_var:
            print("Available variables:", list(f.keys()))
        q   = f["grid_tracers_c_xyzt"][:, :, lat_lim, :]
        w   = f["grid_w_full_xyzt"][:, :, lat_lim]
    return q, w

def _extract_day(fp):
    m = re.search(r"_startfrom_(\d+)day", fp)
    return int(m.group(1)) if m else -1

def read_dycore_series(pattern, lat_lim):
    # find & sort
    files = sorted(glob.glob(pattern), key=_extract_day)[10:]

    q_all = []
    w_all = []
    for i, fp in enumerate(files):
        q_tmp, w_tmp = read_Dycore_data(fp, print_var=(i==0), lat_lim=lat_lim)
        q_all.append(q_tmp)
        w_all.append(w_tmp)
        print(f'finished reading {fp}')

    q_all   = np.concatenate(q_all, axis=0)
    w_all   = np.concatenate(w_all, axis=0)
    return q_all, w_all

def standardize_data(data):
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1)
    data_std = (data - mean) / std
    return data_std


def calc_LIM(M1, M2, t):
    M1 = np.asarray(M1)
    M2 = np.asarray(M2)

    M = np.vstack([M1, M2])             # shape (2, N)
    dM_dt = np.vstack([
        np.gradient(M1, t, edge_order=2),
        np.gradient(M2, t, edge_order=2)
    ])                                   # shape (2, N)

    # now dot‐products on ndarrays
    C = M @ M.T                          # (2,2) covariance‐like matrix
    G = dM_dt @ M.T                      # (2,2)

    A = G @ np.linalg.inv(C)             # linear operator
    print("A =\n", A)

    eig_val, _ = np.linalg.eig(A)
    growth_rate = np.real(eig_val[0])
    freq        = np.abs(np.imag(eig_val[0]))
    print("growth_rate =", growth_rate, " freq =", freq)

    return A, growth_rate, freq


def least_square_fit(data, PC):
    beta, *_ = np.linalg.lstsq(PC, data, rcond=None)

    return beta[0, :], beta[1, :]

def process_one_case(pattern, case_name):
    # read dycore data
    q, w = read_dycore_series(pattern, lat_lim)
    print(f"[{case_name}] finish reading data, shape:", w.shape)

    # calculate column water vapor, and standardize
    cwv = np.trapz(q, p, axis=1) / 9.8
    cwv_eq = np.nanmean(cwv, axis=1)
    cwv_std = standardize_data(cwv_eq)

    # calculate EOF with cwv
    model = EOF(dataset=(cwv_std,), n_components=2, field="1D")
    model.get()

    # use cwv to regress with PC
    beta1, beta2 = least_square_fit(cwv_std, model.PC.T)

    # Plot
    x = np.linspace(0, 360, beta1.shape[0], endpoint=False)
    plt.figure(figsize=(8,3))
    plt.plot(x, beta1, color='r', label='PC1', linestyle='-')
    plt.plot(x, beta2, color='b', label='PC2', linestyle='--')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.xlim(-1, 361)
    plt.ylim(-0.4, 0.4)
    plt.xlabel('Longitude (°)')
    plt.ylabel('Regression Coeff.')
    plt.title(f'CWV regress to PC 1&2 (Case: {case_name})')
    plt.legend(loc='upper right')
    outpath = f'/home/garywu/summer_2025/EOF/figures/regress_{case_name}.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[{case_name}] saved plot to {outpath}")
    return


def main():
    # set lat, lon
    global lat_lim, p
    lat = np.linspace(-90, 90, 64)
    lat_lim = np.where((lat >= -10.0) & (lat <= 10.0))[0]
    p = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy')

    # define all the cases
    cases = [
        ("/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_200_sst1K/data/*", "LRF_LW_200_sst1K"),
        ("/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_200_sst2.5K/data/*", "LRF_LW_200_sst2.5K"),
        ("/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_300_sst1K/data/*", "LRF_LW_300_sst1K"),
        ("/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_300_sst2.5K/data/*", "LRF_LW_300_sst2.5K")
    ]

    # process and plot for each cases
    for pattern, case_name in cases:
        process_one_case(pattern, case_name)

    return

if __name__ == "__main__":
    main()