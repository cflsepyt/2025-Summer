import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
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

def main():
    # set lat, lon
    lat = np.linspace(-90, 90, 64)
    lon = np.linspace(0, 360, 128, endpoint=False)
    lat_lim = np.where( ( lat >= -10.0 ) & ( lat <= 10.0 ) )[0]

    # load pressure coord
    p = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy')

    # load dycore data (10 N ~ 10 S)
    pattern = "/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_300_sst2.5K/data/*"
    case_name = "ws500d_gLRF_LW_300_sst2.5K"
    q, w = read_dycore_series(pattern, lat_lim)
    print("finish reading data, shape:", w.shape) # (time, lev, lat, lon)

    # calculate cwv
    cwv = np.trapz(q, p, axis=1) / 9.8

    # EOF
    model = EOF(dataset=(cwv,), n_components=4, field="2D")
    model.get()

    print("EOFs shape:", model.EOF.shape)
    print("PCs shape:", model.PC.shape)     
    print("Explained variance:", model.explained)

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex='col', sharey='row')
    axes = axes.flatten()

    eofs_data = [model.EOF[i].reshape(8, 128) for i in range(4)]

    for i in range(4):
        cs = axes[i].contourf(
            lon, lat[lat_lim], eofs_data[i],
            cmap='RdBu_r', levels=np.linspace(-0.06, 0.06, 21), extend='both',
        )
        axes[i].set_title(f'EOF {i+1}, exp var: {model.explained[i]:.3f}', fontsize=14)

        if i in (0, 2):
            axes[i].set_ylabel('Latitude', fontsize=12)
        else:
            axes[i].set_ylabel('')

        if i in (2, 3):
            axes[i].set_xlabel('Longitude', fontsize=12)
        else:
            axes[i].set_xlabel('')

    # colorbar
    fig.tight_layout()
    cbar = fig.colorbar(
        cs, ax=axes, orientation='vertical', shrink=0.6, pad=0.04
    )

    plt.savefig(f'/home/garywu/summer_2025/EOF/figures/EOF_{case_name}.png', dpi=300, bbox_inches='tight')
    
    return

if __name__ == "__main__":
    main()