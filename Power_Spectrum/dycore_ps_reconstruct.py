import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

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



if __name__ == '__main__':
    # lat, lon and lat_lim
    x = np.linspace(0, 360, 128, endpoint=False)
    y = np.linspace(-90, 90, 64)
    lat_lim = np.where( ( y >= -15.0 ) & ( y <= 15.0 ) )[0]

    # read dycore data
    pattern = "/data92/Quark/LRFws/HSt42_20_ws500d_qLRF/data/RH80_L20_1500day_startfrom_*.dat"
    q, u, w = read_dycore_series(pattern)
    nt, nlev, nlat, nlon = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    print('finished loading data')

    # load pressure coord.
    p = np.load('/data92/garywu/2025_summer/dycore/npy_files/sst_p_mean.npy')
    
    # calculate column water vapor
    cwp = np.trapz(q, p, axis=1) / 9.8
    cwp_eq = cwp[2000:, lat_lim, :]

    # get wavenumber and frequency mask
    wn = np.fft.fftfreq(128, d=1/128)
    fr = np.fft.fftfreq(cwp_eq.shape[0], d=1/4)

    wnm, frm = np.meshgrid( wn, fr )

    mask = (
        ((wnm>= 0.02)&(wnm<= 2)&(frm>=0.01)&(frm<=0.06)) |
        ((wnm<= -0.02)&(wnm>= -2)&(frm<=-0.01)&(frm>=-0.06))
    )

    # calculate reconstructed signal
    cwp_eq -= cwp_eq.mean( axis=(0, 2), keepdims=True )

    sym = ( cwp_eq + np.flip( cwp_eq, axis=1 ) ) / 2

    sym_fft = fft2(sym)
    sym_fft_masked = sym_fft * mask
    reconstruct_signal = ifft2( sym_fft_masked )
    print(reconstruct_signal.shape)
    # mean and std of reconstructed signal
    reconstruct_signal_mean = np.mean(reconstruct_signal)
    reconstruct_signal_std = np.std(reconstruct_signal)
    
    # mask of significance
    significance_mask = np.where((reconstruct_signal - reconstruct_signal_mean) > 2 * reconstruct_signal_std)
    print(significance_mask)
    
    np.save("sig_mask.npy", significance_mask)
    
    # plot
    plt.pcolormesh(x, np.arange(cwp_eq.shape[0]), reconstruct_signal)
    plt.savefig('/home/garywu/summer_2025/Power_Spectrum/figures/recon_test.png', dpi=300, bbox_inches='tight')