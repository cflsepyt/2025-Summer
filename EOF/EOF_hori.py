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


class EOF:
    """
    Calculating empirical orthogonal funcitons (EOFs)
    
    Parameters
    ----------
    dataset: tuple
        A tuple with elements are variables that you want to find their EOFs
        Variables must be array like, and must be standardized
        If given more than one dataset, combined EOF will be calculated
    
    n_components: int
        Number of modes that you need

    field: str, 1D or 2D, default = 2D
        The dimension of input variable arrays
    
    **svd_args: 
        Arguments for svd calculation in sklearn.decomposition.PCA
    
    About EOFs
    ----------
    The EOFs are vectors that represent the spatial distribution with largest temporal variation.
    In short, finding EOFs is equivalent to solving an eigenvalue problem of the variance matrix. The first eigen mode
    is EOF1, the second is EOF2, and so on.
    A variance matrix is done by multiplying the input variable array and its transpose, with temporal mean is zero.

    Note that
    ---------
    Original algorithm is developed by Kai-Chih Tseng: https://kuiper2000.github.io/
    """
    def __init__(
        self,
        dataset     : tuple,
        n_components: int,
        field       : str  = "2D",
        **svd_kwargs
    ):
        self.dataset      = dataset
        self.data_arr     = None
        self.n_components = n_components
        self.field        = field
        self.pca          = None
        self.EOF          = None
        self.PC           = None
        self.explained    = None
        self._svd         = svd_kwargs
    
    def _check_dimension(self):
        """
        If the dimensions of input variables are not consistent with self.field, raise ValueError

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for sub in self.dataset:
            if (self.field == "2D" and np.ndim(sub) == 3) or (self.field == "1D" and np.ndim(sub) == 2): pass
            else:
                raise ValueError("The dimensions of input variables need to be consistent with input 'field'")

    def _single_subdataset_reshape_2D(self, subdataset: np.ndarray) -> np.ndarray:
        """
        Reshape input array with dimension (time, space) into (time*space)

        Parameters
        ----------
        subdataset: array
            The array of variable with dimension (time, space)
        
        Returns
        -------
        _subdataset_new: array
            The array of variable reshaped to dimension (time*space)
        """
        _subdataset_new = np.reshape(subdataset, (subdataset.shape[0], subdataset.shape[1]*subdataset.shape[2]))
        return _subdataset_new

    def _dataset_reshape_2D(self) -> tuple:
        """
        if there are more than two variables:
            Transfer input tuple with variable arrays into np.ndarray,
            and reshape it from dimension (var, time, space1, space2) into (time, var*space1*space2)
            Assign self.data_arr as the reshaped array
        else:
            Reshape the variable array into (time, space1*space2)
            Assign self.data_arr as the reshaped array

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if len(self.dataset) > 1:
            arr           = np.array(self.dataset)
            self.data_arr = np.reshape(np.transpose(arr, (1, 0, 2, 3)), (arr.shape[1], arr.shape[0]*arr.shape[2]*arr.shape[3]))
        else:
            self.data_arr = self._single_subdataset_reshape_2D(self.dataset[0])
    
    def _dataset_reshape_1D(self):
        """
        Same as _dataset_reshape_2D, but for 1-dimensional input variables

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if len(self.dataset) > 1:
            arr           = np.array(self.dataset)
            self.data_arr = np.reshape(np.transpose(arr, (1, 0, 2)), (arr.shape[1], arr.shape[0]*arr.shape[2]))
        else:
            self.data_arr = self.dataset[0]

    def _fit(self):
        """
        Create a PCA class and fit it with input data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pca_ = PCA(n_components = self.n_components, **self._svd, svd_solver="full", random_state=0)
        pca_.fit(self.data_arr)
        self.pca = pca_

    def _calc_EOF(self):
        """
        Calculate different EOF modes

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.EOF = self.pca.components_
    
    def _calc_PC(self):
        """
        Calculate PCs with input data and EOF modes

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PC = np.dot(self.EOF, self.data_arr.T)
        self.PC = PC
    
    def _calc_explained(self):
        """
        Calculate the explainable ratio of each given EOF modes

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.explained = self.pca.explained_variance_ratio_

    def get(self):
        """
        Call _fit() _calc_EOF() _calc_PC _calc_explained() and calculate all of them

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._check_dimension()
        if self.field == "1D":
            self._dataset_reshape_1D()
        else:
            self._dataset_reshape_2D()
        self._fit()
        self._calc_EOF()
        self._calc_PC()
        self._calc_explained()


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
    cwv_eq = np.nanmean(cwv, axis=1)

    # standardize the data
    mean_time = np.mean(cwv_eq, axis=0)
    std_time = np.std(cwv_eq, axis=0, ddof=1)
    cwv_std = (cwv_eq - mean_time) / std_time

    # EOF
    model = EOF(dataset=(cwv_std,), n_components=4, field="1D")
    model.get()

    print("EOFs shape:", model.EOF.shape)
    print("PCs shape:", model.PC.shape)     
    print("Explained variance:", model.explained)  

    # calculate dM/dt
    M1 = model.PC[0, :]
    M2 = model.PC[1, :]
    M = np.vstack([M1, M2])

    t = np.arange(M1.shape[0])
    dM_dt = np.vstack([np.gradient(M1, t), np.gradient(M2, t)])

    # calculate linear operator
    MMT = M @ M.T                     # (2,2)
    dMMT = dM_dt @ M.T                # (2,2)
    A = dMMT @ np.linalg.inv(MMT)     # (2,2)
    print(A)

    # calculate the eigenvalue of A
    eig_val, _ = np.linalg.eig(A)
    print(eig_val)
    growth_rate, freq = np.real(eig_val[0]), np.abs(np.imag(eig_val[0]))
    print(growth_rate, freq)

    # # plot
    # fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex='col', sharey='row')
    # axes = axes.flatten()

    # eofs_data = [model.EOF[i].reshape(8, 128) for i in range(4)]

    # for i in range(4):
    #     cs = axes[i].contourf(
    #         lon, lat[lat_lim], eofs_data[i],
    #         cmap='RdBu_r', levels=np.linspace(-0.06, 0.06, 21), extend='both',
    #     )
    #     axes[i].set_title(f'EOF {i+1}, exp var: {model.explained[i]:.3f}', fontsize=14)

    #     if i in (0, 2):
    #         axes[i].set_ylabel('Latitude', fontsize=12)
    #     else:
    #         axes[i].set_ylabel('')

    #     if i in (2, 3):
    #         axes[i].set_xlabel('Longitude', fontsize=12)
    #     else:
    #         axes[i].set_xlabel('')

    # # colorbar
    # fig.tight_layout()
    # cbar = fig.colorbar(
    #     cs, ax=axes, orientation='vertical', shrink=0.6, pad=0.04
    # )

    # plt.savefig(f'EOF_{case_name}.png', dpi=300, bbox_inches='tight')
    
    return

if __name__ == "__main__":
    main()