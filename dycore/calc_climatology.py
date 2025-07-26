import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np

# functions for reading dycore data
def read_Dycore_data(filepath, print_var=False):
    with h5py.File(filepath, "r") as f:
        if print_var:
            print("Available variables:", list(f.keys()))
        q   = f["grid_tracers_c_xyzt"][:, :, :, :]
        t   = f["grid_t_c_xyzt"][:, :, :, :]
    return q, t

def _extract_day(fp):
    m = re.search(r"_startfrom_(\d+)day", fp)
    return int(m.group(1)) if m else -1

def read_dycore_series(pattern):
    # find & sort
    files = sorted(glob.glob(pattern), key=_extract_day)[-40:]

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


def main():
    pattern = "/data92/garywu/sst_2000d_2.5K/HSt42_20/data/*"
    q, t = read_dycore_series(pattern)
    print('data shape:', q.shape)

    # calculate mean along time axis
    q_mean = np.nanmean(q, axis=0)
    t_mean = np.nanmean(t, axis=0)
    print('mean shape:', q_mean.shape)

    # save as .npy files
    np.save("/data92/garywu/2025_summer/dycore/npy_files/sst_2.5K_q_mean.npy", q_mean)
    np.save("/data92/garywu/2025_summer/dycore/npy_files/sst_2.5K_t_mean.npy", t_mean)
    return


if __name__ == '__main__':
    main()