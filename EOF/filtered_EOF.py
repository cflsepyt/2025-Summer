import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
from EOF import EOF
from scipy.signal import savgol_filter

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


def fft2( data ):
    data_fft = np.fft.fft( data, axis=0 )
    data_fft = np.fft.ifft( data_fft, axis=-1 ) * data.shape[-1]
    return data_fft

def ifft2( data ):
    data_ifft = np.fft.ifft( data, axis=0 )
    data_ifft = np.fft.fft( data_ifft, axis=-1 ) / data.shape[-1]
    return data_ifft.real


def filter_data_1d(data, 
                   wn_bound, 
                   fr_bound,
                   spd
                   ):
    # get wavenumber and frequency
    wn = np.fft.fftfreq(data.shape[-1], d=1/data.shape[-1])
    fr = np.fft.fftfreq(data.shape[0], d=1/spd)

    wnm, frm = np.meshgrid(wn, fr)

    # get max/min wavenumber and frequency
    wn_min, wn_max = wn_bound[0], wn_bound[1]
    fr_min, fr_max = fr_bound[0], fr_bound[1]

    mask = (
        ((wnm >= wn_min)&(wnm <= wn_max)&(frm >= fr_min)&(frm <= fr_max)) |
        ((wnm <= -wn_min)&(wnm >= -wn_max)&(frm <= -fr_min)&(frm >= -fr_max))
    )
    # calculate filtered signal
    sym = (data + np.flip(data, axis=1)) / 2
    sym_fft = fft2(sym)
    sym_fft_masked = sym_fft * mask
    filtered_signal = ifft2(sym_fft_masked)

    return filtered_signal


def filter_data_2d(data, lat_lim, 
                   wn_bound,
                   fr_bound,
                   spd):

    output = []

    for i in range(len(lat_lim)):
        output_tmp = filter_data_1d(data[:, i, :], wn_bound, fr_bound, spd)
        output.append(output_tmp)
    
    output = np.array(output)
    output = np.transpose(output, (1, 0, 2)) # shape: (time, lat_lim, lon)

    return output


def main():
    # # set lat_lim
    # lat = np.linspace(-90, 90, 64)
    # lat_lim = np.where((lat >= -10.0) & (lat <= 10.0))[0]
    
    # # load dycore data
    # pattern = '/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_200_sst1K/data/*'
    # q, _ = read_dycore_series(pattern, lat_lim)

    # p = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy')
    
    # # calculate cwv
    # cwv_eq = np.trapz(q, p, axis=1) / 9.8
    # cwv_eq -= cwv_eq.mean(axis=(0, 2), keepdims=True)

    # # get filtered data
    # cwv_filtered = filter_data_2d(cwv_eq, lat_lim, [1, 10], [0.01, 0.1], spd=4)
    # print(cwv_filtered.shape)

    fpath = "/work/DATA/Satellite/OLR/olr_anomaly.nc";

    with nc.Dataset( fpath, "r" ) as ds:
        dims = {
            key: ds[key][:]
            for key in ds.dimensions.keys()
        }
        
        lat_lim = np.where( ( dims["lat"] >= -5.0 ) & ( dims["lat"] <= 5.0 ) )[0]
        dims["lat"]  = dims["lat"][lat_lim]
        dims["time"] = dims["time"][:1000]
        
        olr = ds["olr"][:1000, lat_lim, :]
        olr -= olr.mean(axis=(0, 2), keepdims=True)

    olr_filtered = filter_data_2d(olr, lat_lim, [1, 10], [0.01, 0.1], spd=1)

    # EOF
    model = EOF(dataset=(olr_filtered,), n_components=2, field="2D")
    model.get()
    eof = model.EOF
    
    # project raw data anomaly onto EOF
    anom = olr.reshape(olr.shape[0], -1)
    proj = np.asarray(eof) @ np.asarray(anom.T)
    print('finish projection')

    from scipy.ndimage import uniform_filter1d

    # Smooth with moving average (e.g., window of 9 points = 2.25 days)
    window_size = 30
    proj_smooth = uniform_filter1d(proj[0], size=window_size)

    plt.figure(figsize=(8, 4))
    plt.plot(proj[0], label='EOF 1', alpha=0.5)
    plt.plot(proj_smooth, label='MA (30 days)', linewidth=2)
    plt.xlabel("Time (days)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("OLR Projected to Filtered EOF")
    plt.tight_layout()
    plt.savefig('proj_smoothed.png', dpi=300)
    plt.show()

    return



if __name__ == '__main__':
    main()