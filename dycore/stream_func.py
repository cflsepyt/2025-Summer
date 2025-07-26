import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt

# functions for reading dycore data
def read_Dycore_data(filepath, print_var=False):
    with h5py.File(filepath, "r") as f:
        if print_var:
            print("Available variables:", list(f.keys()))
        u   = f["grid_u_c_xyzt"][:]
        v   = f["grid_v_c_xyzt"][:]
    return u, v

def _extract_day(fp):
    m = re.search(r"_startfrom_(\d+)day", fp)
    return int(m.group(1)) if m else -1

def read_dycore_series(pattern):
    # find & sort
    files = sorted(glob.glob(pattern), key=_extract_day)[-20:]

    u_all = []
    v_all = []
    for i, fp in enumerate(files):
        u_tmp, v_tmp = read_Dycore_data(fp, print_var=(i==0))
        u_all.append(u_tmp)
        v_all.append(v_tmp)
        print(f'finished reading {fp}')

    u_all   = np.concatenate(u_all, axis=0)
    v_all   = np.concatenate(v_all, axis=0)
    return u_all, v_all

def main():
    # set lat, lon
    lat = np.linspace(-90, 90, 64)
    lon = np.linspace(0, 360, 128, endpoint=False)
    lat_lim = np.where( ( lat >= -15.0 ) & ( lat <= 15.0 ) )[0]

    # load pressure coord
    p = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy')

    # constants
    a = 6.371e6
    g = 9.81

    # load dycore data (15 N ~ 15 S)
    pattern = "/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_200_sst2.5K/data/*"
    casename = "ws500d_gLRF_LW_200_sst2.5K"
    u, v = read_dycore_series(pattern)
    print("finish reading data, shape:", u.shape) # (time, lev, lat, lon)

    # time & zonal mean
    u_mean = np.nanmean(u, axis=(0, -1))  # (lev, lat)
    v_mean = np.nanmean(v, axis=(0, -1))  # (lev, lat)

    # convert lat to radians
    lat_rad = np.deg2rad(lat)
    coslat = np.cos(lat_rad)

    # calculate streamfunc
    dp = np.gradient(p)  # Pa

    psi = np.zeros_like(v_mean)
    for j in range(len(lat)):
        # cumulative integral
        psi[:, j] = (2 * np.pi * a * coslat[j] / g) * np.cumsum(v_mean[:, j] * dp)

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))

    cf = ax.contourf(lat, p/100, psi, levels=np.arange(-2, 2.5, 0.5)*1e11, cmap='RdBu_r', extend='both')
    cbar = fig.colorbar(cf, ax=ax, label='kg/s')

    cs = ax.contour(lat, p/100, u_mean, levels=np.arange(-10, 50, 10), colors='black')
    ax.clabel(cs, fontsize=10)
    
    ax.invert_yaxis()
    ax.set_title('Run 6')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel('Latitude (°)')
    plt.savefig(f'/home/garywu/summer_2025/dycore/figures/streamfunc/{casename}_streamfunc.png', bbox_inches='tight', dpi=300)
    return

if __name__ == "__main__":
    main()