import glob, re, h5py
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
from EOF import EOF

# functions for reading dycore data
def read_Dycore_data(filepath, print_var=False, lat_lim=None):
    """
    Read Dycore data from a given HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the Dycore HDF5 file.
    print_var : bool, optional
        If True, print the names of available variables in the HDF5 file.
    lat_lim : array-like, optional
        Range of latitudes to read from the HDF5 file.

    Returns
    -------
    tuple
        q, w : 3D arrays
            q is the water vapor mixing ratio and w is the vertical velocity.
    """
    with h5py.File(filepath, "r") as f:
        if print_var:
            print("Available variables:", list(f.keys()))
        q   = f["grid_tracers_c_xyzt"][:, :, lat_lim, :]
        w   = f["grid_w_full_xyzt"][:, :, lat_lim]
    return q, w

def _extract_day(fp):
    """
    Extract day from Dycore data filename.

    Parameters
    ----------
    fp : str
        Dycore data filename.

    Returns
    -------
    int
        Day number extracted from filename, or -1 if not found.

    Notes
    -----
    The filename should be in the format *_startfrom_<N>day*, where <N> is the
    day number.
    """
    m = re.search(r"_startfrom_(\d+)day", fp)
    return int(m.group(1)) if m else -1

def read_dycore_series(pattern, lat_lim):
    """
    Read Dycore data files and return q and w arrays in time-major order.

    Parameters
    ----------
    pattern : str
        Glob pattern for Dycore data files.
    lat_lim : array-like
        Range of latitudes to read from the HDF5 files.

    Returns
    -------
    tuple
        q, w : 3D arrays
            q is the water vapor mixing ratio and w is the vertical velocity.
    """
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


# function for standardizing a data array along time axis
def standardize_data(data, axis=0):
    """
    Standardize the data array along the specified axis.

    Parameters
    ----------
    data : array_like
        The input data array to be standardized.
    axis : int, optional
        The axis along which to compute the mean and standard deviation. Default is 0.

    Returns
    -------
    data_std : ndarray
        The standardized data array with mean 0 and standard deviation 1 along the specified axis.

    Notes
    -----
    NaN values are ignored during the calculation of mean and standard deviation.
    """
    # Calculate the mean of the data along the specified axis, ignoring NaNs
    mean = np.nanmean(data, axis=axis)
    
    # Calculate the standard deviation of the data along the specified axis, ignoring NaNs
    std = np.nanstd(data, axis=axis, ddof=1)
    
    # Standardize the data by subtracting the mean and dividing by the standard deviation
    data_std = (data - mean) / std
    
    return data_std

def calc_LIM(M1, M2, t):
    """
    Calculate the Linear Inverse Model (LIM) from two time series of mode amplitude,
    and calculate the growth rate and frequency.

    Parameters
    ----------
    M1, M2 : array_like
        Two time series of mode amplitude, shape (N,).
    t : array_like
        Time array, shape (N,).

    Returns
    -------
    A : ndarray
        The linear operator of the LIM, shape (2,2).
    growth_rate : float
        The growth rate, in days^-1.
    freq : float
        The frequency, in days^-1.

    Notes
    -----
    The LIM is calculated as A = G @ C^{-1}, where C is the covariance matrix
    of M1 and M2, and G is the covariance matrix of M1 and dM2/dt.
    The growth rate and frequency are calculated from the eigenvalues of A.
    """
    M1 = np.asarray(M1)
    M2 = np.asarray(M2)

    M = np.vstack([M1, M2])             # shape (2, N)
    dM_dt = np.vstack([
        np.gradient(M1, t, edge_order=2),
        np.gradient(M2, t, edge_order=2)
    ])                                   # shape (2, N)

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
    """
    Perform a least square regression.

    Parameters
    ----------
    data : array_like
        The data to be fitted, shape (N,).
    PC : array_like
        The principal components, shape (N, M).

    Returns
    -------
    beta1, beta2 : ndarray
        The coefficients of the least square regression, shape (M,).

    Notes
    -----
    The least square regression is performed as `data = PC @ beta + error`.
    The coefficients `beta` are calculated as the solution of the linear
    least-squares problem `minimize(||data - PC @ beta||^2)`.
    """
    beta, *_ = np.linalg.lstsq(PC, data, rcond=None)

    return beta[0, :], beta[1, :]

# function for processing a single case (dycore output)
def process_one_case(pattern, case_name):
    """
    Process a single case by reading dycore data, calculating column water vapor,
    standardizing, performing EOF analysis, and calculating the Linear Inverse Model (LIM).

    Parameters
    ----------
    pattern : str
        The file pattern to match the dycore data files.
    case_name : str
        The name of the case being processed.

    Returns
    -------
    A : ndarray
        The linear operator of the LIM.
    growth : float
        The growth rate in days^-1.
    freq : float
        The frequency in days^-1.
    """
    # Read dycore data
    q, w = read_dycore_series(pattern, lat_lim)
    print(f"[{case_name}] finish reading data, shape:", w.shape)

    # Calculate column water vapor and standardize it
    cwv = np.trapz(q, p, axis=1) / 9.8
    cwv_eq = np.nanmean(cwv, axis=1)
    cwv_std = standardize_data(cwv_eq)

    # Calculate EOF with column water vapor
    model = EOF(dataset=(cwv_std,), n_components=2, field="1D")
    model.get()

    # Get the first two principal components (Mode 1 & 2)
    M1, M2 = model.PC[0, :], model.PC[1, :]

    # Calculate LIM and extract growth rate and frequency
    t = np.arange(M1.shape[0]) * 4  # Time array with 4-day intervals
    A, growth, freq = calc_LIM(M1, M2, t)
    print(f"[{case_name}] growth={growth:.3e}, freq={freq:.3e}")

    return A, growth, freq


def main():
    """
    Main function to process all the cases.

    This function reads all the input data, standardizes them,
    performs EOF analysis, calculates the Linear Inverse Model (LIM),
    and saves the results.
    """
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

    # process and save for each case
    results = {}
    
    for pattern, case_name in cases:
        results[case_name] = process_one_case(pattern, case_name)

    np.savez("LIM_results.npz", **{
        f"{name}_A": res[0] for name, res in results.items()
    })

    np.savez(
        "LIM_results_growth_freq.npz",
        **{f"{name}_growth": res[1] for name, res in results.items()},
        **{f"{name}_freq":   res[2] for name, res in results.items()},
    )

    return

if __name__ == "__main__":
    main()
