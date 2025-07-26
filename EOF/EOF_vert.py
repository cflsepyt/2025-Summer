import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    lat_lim = np.where( ( lat >= -15.0 ) & ( lat <= 15.0 ) )[0]

    # load pressure coord
    p = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy')

    # load dycore data (15 N ~ 15 S)
    pattern = "/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_300_sst1K/data/*"
    q, w = read_dycore_series(pattern, lat_lim)
    print("finish reading data, shape:", w.shape) # (time, lev, lat, lon)

    # transpose and reshape data
    w = np.transpose(-w, (1, 0, 2, 3))
    print("transposed data shape:", w.shape) # (lev, time, lat, lon)
    w = np.reshape(w, (20, -1))
    print(w.shape) # (lev, n_sample)

    # calculate anomaly along sample axis
    w_anom = w - w.mean(axis=1, keepdims=True)

    # do PCA
    pca = PCA(n_components=2)
    w_pca = pca.fit(w_anom.T)
    eof = pca.components_.T
    exp_var = pca.explained_variance_ratio_
    print(eof.shape)
    print(exp_var)

    # plot
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].plot(eof[:, 0], p)
    ax[1].plot(eof[:, 1], p)

    ax[0].set_title(f'EOF 1 (exp var: {exp_var[0]:.2f})')
    ax[1].set_title(f'EOF 2 (exp var: {exp_var[1]:.2f})')
    ax[0].invert_yaxis()
    plt.savefig('eof_w.png', bbox_inches='tight', dpi=200)

    return

if __name__ == "__main__":
    main()