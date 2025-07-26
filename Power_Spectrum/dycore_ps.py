import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
import matsuno_plot
from scipy.signal import detrend
from matplotlib.colors import ListedColormap

# custom colormap
base_cmap = plt.cm.Reds
ncolors = 256
colors = base_cmap(np.linspace(0, 1, ncolors))

threshold = 1
vmin = 0
vmax = 3.0
norm_threshold = int((threshold - vmin) / (vmax - vmin) * ncolors)
colors[:norm_threshold] = [1, 1, 1, 1]

custom_cmap = ListedColormap(colors)


# functions for reading dycore data
def read_Dycore_data(filepath, lat_lim, print_var=True):
    with h5py.File(filepath, "r") as f:
        if print_var:
            print("Available variables:", list(f.keys()))
        q   = f["grid_t_c_xyzt"][:, :, lat_lim, :]
    return q

def _extract_day(fp):
    m = re.search(r"_startfrom_(\d+)day", fp)
    return int(m.group(1)) if m else -1

def read_dycore_series(pattern, lat_lim):
    # find & sort
    files = sorted(glob.glob(pattern), key=_extract_day)

    q_all = []
    for i, fp in enumerate(files):
        q_tmp = read_Dycore_data(fp, lat_lim, print_var=(i==0))
        q_all.append(q_tmp)
        print(f'finished reading {fp}')

    q_all   = np.concatenate(q_all, axis=0)
    return q_all

# function for calculating pressure-weighted mean
def mean_pressure_weighted(p, var, axis=0):
    
    p = np.asarray(p)
    var = np.asarray(var)

    shape = [1] * var.ndim
    shape[axis] = p.size
    p_bcast = p.reshape(shape)

    num = np.trapz(var * p_bcast, p, axis=axis)
    den = np.trapz(p, p)
    return num / den

# function for plotting dispersion curve
def draw_wk_sym_analysis(
    max_wn: float = 15,
    ax=None,
    matsuno_lines: bool = True,
    he: list = [25, 50, 150],
    meridional_modes: list = [1],
):
    matsuno_modes = matsuno_plot.matsuno_modes_wk(
        he=he, n=meridional_modes, max_wn=max_wn
    )

    if ax is None:
        ax = plt.gca()

    if matsuno_lines:
        for key in matsuno_modes:
            ax.plot(
                matsuno_modes[key]["Kelvin(he={}m)".format(key)],
                color="k",
                linestyle="--"
            )
            ax.plot(
                matsuno_modes[key]["ER(n=1,he={}m)".format(key)],
                color="k",
                linestyle="--"
            )
            ax.plot(
                matsuno_modes[key]["EIG(n=1,he={}m)".format(key)],
                color="k",
                linestyle="--"
            )
            ax.plot(
                matsuno_modes[key]["WIG(n=1,he={}m)".format(key)],
                color="k",
                linestyle="--"
            )

# functions for calculating power spectrum
def power_spec(data):
    
    data_fft = np.fft.fft(data, axis=1)
    data_fft = np.fft.ifft(data_fft, axis=3) * data.shape[3]
    
    ps = (data_fft * data_fft.conj()) / (data.shape[1] * data.shape[3])**2.0
    
    return ps.mean(axis=0).real

def background(data):
    data = data.copy()
    kernel = np.array([1, 2, 1]) / 4.0
    half_freq = data.shape[0] // 2

    for i in range(10):
        data = convolve1d(data, kernel, axis=0, mode="reflect" )
    for i in range(10):
        data[:half_freq] = convolve1d(data[:half_freq], kernel, axis=1, mode="reflect")
    for i in range(40):
        data[half_freq:] = convolve1d(data[half_freq:], kernel, axis=1, mode="reflect")

    return data


def power_spectrum_calc(data):
    """
    data: 3-D array with shape (time, lat, lon)
    """
    # equatorial latitudes
    data_eq = data.copy()
    data_eq -= data_eq.mean(axis=(0, 2), keepdims=True)

    # symmetric & antisymmetric decomposition
    sym  = (data_eq + np.flip(data_eq, axis=1)) / 2
    asym = (data_eq - np.flip(data_eq, axis=1)) / 2

    # window data
    hanning = np.hanning(96 * 4)[:, None, None]

    sym_window = []
    asym_window = []

    for i in range( data_eq.shape[0] // (96*4) ):
        sym_window.append( detrend( sym[i*48*4:i*48*4+96*4], axis=0) * hanning )
        asym_window.append( detrend( asym[i*48*4:i*48*4+96*4], axis=0) * hanning )
        
    sym_window = np.array( sym_window )
    asym_window = np.array( asym_window )

    wn = np.fft.fftshift(np.fft.fftfreq(128, d=1/128))
    fr = np.fft.fftshift(np.fft.fftfreq(96*4, d=1/4))

    sym_ps  = np.fft.fftshift(power_spec(sym_window).sum(axis=1))[fr > 0] * 2.0
    asym_ps = np.fft.fftshift(power_spec(asym_window).sum(axis=1))[fr > 0] * 2.0

    return wn, fr, sym_ps, asym_ps


def plot_power_spectrum(wn, fr, sym_ps, bg, output_dir, filename):
    
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_wk_sym_analysis(ax=ax)

    cf = ax.contourf(
        wn, fr[fr > 0], sym_ps / bg,
        levels=np.arange(0, 3.5, 0.5),
        # levels=21,
        extend="max",
        cmap=custom_cmap
        # cmap='Reds'
    )
    cbar = fig.colorbar(cf, shrink=0.8)

    R_earth = 6.371e6
    c = 8.0
    slope = c * 86400.0 / (2 * np.pi * R_earth)
    wn_line = np.array([0, ax.get_xlim()[1]])
    fr_line = slope * wn_line
    ax.plot(wn_line, fr_line,
            linestyle='--',
            linewidth=1.5,
            color='blue',
            label='6 m/s')

    ax.set_xlim(-15, 15)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel('Zonal Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Mean power spectrum ({filename})')
    
    plt.savefig(f'{output_dir}{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'figure saved as {filename}.png')
    return


def main():
    # lat, lon and lat_lim
    x = np.linspace(0, 360, 128, endpoint=False)
    y = np.linspace(-90, 90, 64)
    lat_lim = np.where( ( y >= -10.0 ) & ( y <= 10.0 ) )[0]
    
    # load global mean pressure below 200 hPa
    pglobal = np.load('/data92/garywu/2025_summer/dycore/npy_files/ctrl_p_mean.npy')[4:]

    # read dycore data
    pattern = "/data92/garywu/LRFws/HSt42_20_ws500d_gLRF_LW_200_sst2.5K/data/*"
    q = read_dycore_series(pattern, lat_lim)

    # get data from 1000d~2000d
    data = q[-3000:]
    print('data shape:', data.shape)

    # get wavenumber and frequency for plotting
    wn, fr, _, _ = power_spectrum_calc(data[:, 2, ...])
    # bg = background((sym_ps + asym_ps) / 2)

    # calculate power spectrum for all level below 200 hPa 
    ntime, nlev, nlat, nlon = data.shape

    sym_ps_all_lev = []
    asym_ps_all_lev = []

    for k in range(4, nlev):
        _, _, sym_tmp, asym_tmp = power_spectrum_calc(data[:, k, :, :])
        sym_ps_all_lev.append(sym_tmp)
        asym_ps_all_lev.append(asym_tmp)

    sym_ps_all_lev = np.array(sym_ps_all_lev)
    asym_ps_all_lev = np.array(asym_ps_all_lev)

    # calculate pressure weighted mean
    sym_ps_mean = mean_pressure_weighted(pglobal / 100, sym_ps_all_lev)
    asym_ps_mean = mean_pressure_weighted(pglobal / 100, asym_ps_all_lev)

    # calculate background
    bg = background((sym_ps_mean + asym_ps_mean) / 2)
    print('finished calculating power spectrum')
    
    # plot and save
    output_dir = '/home/garywu/summer_2025/Power_Spectrum/figures/'
    filename = 'ws500d_gLRF_LW_200_sst2.5K'
    plot_power_spectrum(wn, fr, sym_ps_mean, bg, output_dir, filename)
    return

if __name__ == '__main__':
    main()