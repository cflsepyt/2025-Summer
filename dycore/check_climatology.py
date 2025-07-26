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
    files = sorted(glob.glob(pattern), key=_extract_day)[-20:]

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
    # set lat, lon
    lat = np.linspace(-90, 90, 64)
    lon = np.linspace(0, 360, 128, endpoint=False)
    lat_lim = np.where( ( lat >= -15.0 ) & ( lat <= 15.0 ) )[0]

    # load pressure coord
    p = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy')

    # load dycore data
    pattern = "/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_200_sst2.5K/data/*"
    casename = "ws500d_gLRF_LW_200_sst2.5K"
    q, t = read_dycore_series(pattern)
    print("finish reading data, shape:", q.shape) # (time, lev, lat, lon)

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(lat, p/100, t.mean(axis=(0, -1)), levels=np.arange(190, 310, 10), colors='black')
    ax.clabel(cs, fontsize=10)
    
    ax.invert_yaxis()
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel('Latitude (°)')
    ax.set_title('Run 6')
    plt.savefig(f'/home/garywu/summer_2025/dycore/figures/climatology/{casename}_meanT.png', bbox_inches='tight', dpi=300)

    return

if __name__ == '__main__':
    main()