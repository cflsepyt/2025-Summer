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
        u   = f["grid_u_c_xyzt"][:, :, :, :]
        w   = f["grid_w_full_xyzt"][:, :, :, :]
    return q, u, w

def _extract_day(fp):
    m = re.search(r"_startfrom_(\d+)day", fp)
    return int(m.group(1)) if m else -1

def read_dycore_series(pattern):
    # find & sort
    files = sorted(glob.glob(pattern), key=_extract_day)

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


def fft2( data ):
    
    data_fft = np.fft.fft( data, axis=0 )
    data_fft = np.fft.ifft( data_fft, axis=-1 ) * data.shape[-1]
    
    return data_fft.sum( axis=1 )

def ifft2( data ):
    
    data_ifft = np.fft.ifft( data, axis=0 )
    data_ifft = np.fft.fft( data_ifft, axis=-1 ) / data.shape[-1]
    
    return data_ifft.real


def filter_events(data):
    # get wavenumber and frequency mask
    wn = np.fft.fftfreq(data.shape[-1], d=1/data.shape[-1])
    fr = np.fft.fftfreq(data.shape[0], d=1/4)

    wnm, frm = np.meshgrid( wn, fr )

    mask = (
        ((wnm>= 2)&(wnm<= 5)&(frm>=0.02)&(frm<=0.08)) |
        ((wnm<= -2)&(wnm>= -5)&(frm<=-0.02)&(frm>=-0.08))
    )

    # calculate reconstructed signal
    data -= data.mean( axis=(0, 2), keepdims=True )

    sym = ( data + np.flip( data, axis=1 ) ) / 2

    sym_fft = fft2(sym)
    sym_fft_masked = sym_fft * mask
    reconstruct_signal = ifft2( sym_fft_masked )
    reconstruct_signal_mean = np.mean(reconstruct_signal)
    reconstruct_signal_std = np.std(reconstruct_signal)

    events = np.where((reconstruct_signal - reconstruct_signal_mean) > 2 * reconstruct_signal_std)

    return events

def main():
    # lat, lon and lat_lim
    lon = np.linspace(0, 360, 128, endpoint=False)
    lat = np.linspace(-90, 90, 64)
    lat_lim = np.where( ( lat >= -15.0 ) & ( lat <= 15.0 ) )[0]

    # read dycore data
    pattern = "/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_200_sst2.5K/data/*"
    case_name = "ws500d_gLRF_LW_200_sst2.5K"
    q, u, w = read_dycore_series(pattern)
    nt, nlev, nlat, nlon = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    print('finished loading data, shape:', q.shape)

    # load pressure coord.
    p = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy')

    # calculate cwv
    cwv = np.trapz(q, p, axis=1) / 9.8
    cwv_eq = cwv[2000:, lat_lim, :]

    # calculate 15N-15S mean for the last 500 days
    q_eq = np.mean(q[2000:, :, lat_lim, :64], axis=2)
    u_eq = np.mean(u[2000:, :, lat_lim, :64], axis=2)
    w_eq = np.mean(w[2000:, :, lat_lim, :64], axis=2) 
    print(q_eq.shape) # (time, lev, lon)

    # compute anomaly
    q_anom = q_eq - q_eq.mean(axis=(0, -1), keepdims=True)
    u_anom = u_eq - u_eq.mean(axis=(0, -1), keepdims=True)
    w_anom = w_eq - w_eq.mean(axis=(0, -1), keepdims=True)

    # load events
    center_idx = cwv_eq.shape[0] // 2
    events = filter_events(cwv_eq)
    # events_list = list(zip(events[1], events[0]))
    events_list = [(x, t) for x, t in zip(events[1], events[0]) if x < 64]

    # calculate composite
    q_comp = np.array([
        np.roll(q_anom[..., x_idx], center_idx-t_idx, axis=0)
        for i, (x_idx, t_idx) in enumerate(events_list)
        ]).mean(axis=0)[center_idx-60:center_idx+60].T
    
    u_comp = np.array([
        np.roll(u_anom[..., x_idx], center_idx-t_idx, axis=0)
        for i, (x_idx, t_idx) in enumerate(events_list)
        ]).mean(axis=0)[center_idx-60:center_idx+60].T

    w_comp = np.array([
        np.roll(w_anom[..., x_idx], center_idx-t_idx, axis=0)
        for i, (x_idx, t_idx) in enumerate(events_list)
        ]).mean(axis=0)[center_idx-60:center_idx+60].T
    
    print('composite finished, shape:', q_comp.shape)


    # plot
    t = np.linspace(-15, 15, 120)

    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(t, p/100, q_comp*1000, cmap='BrBG', 
                     levels=np.linspace(-0.5, 0.5, 21), 
                    # levels=21,
                     extend='both')

    skip = 3
    quiver = ax.quiver(t[::skip], p/100, u_comp[:, ::skip], -w_comp[:, ::skip] * 100, scale=100)
    
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('days')
    ax.set_ylabel('hPa')
    ax.set_title(f'Warm Pool Composite ({case_name[8:]})')
    cbar = fig.colorbar(cf, ax=ax, shrink=0.7)
    cbar.ax.set_title('g/kg')

    plt.tight_layout()
    plt.savefig(f'/home/garywu/summer_2025/dycore/figures/composite/comp_{case_name}.png', dpi=300, bbox_inches='tight')
    return

if __name__ == '__main__':
    main()