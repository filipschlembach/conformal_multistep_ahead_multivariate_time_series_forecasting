from abc import ABC, abstractmethod
import enum
import json
import logging
import numpy as np
import pandas as pd
import plotly
import os.path

import torch.utils.data


class SyntheticDatasets(str, enum.Enum):
    """
    List of available synthetic data sets.
    """
    feldman = 'feldman'
    barber_iid = 'barber_iid'
    barber_changepoints = 'barber_changepoints'
    barber_distribution_drift = 'barber_distribution_drift'
    stankeviciute = 'stankeviciute'


class RealWorldDatasets(str, enum.Enum):
    """
    List of available real world data sets.
    """
    elec2 = 'elec2'


class Dataset(torch.utils.data.Dataset):

    def __init__(self, window_l: int = -1, horizon_l: int = 1,
                 window_f: int | list = 1, horizon_f: int | list = 1, stride: int = -1,
                 params: str | dict = None):
        """
        Generates / load the data for a specific dataset and preprocesses it according to the provided arguments.
        :param window_l: number of past time-steps for every window (object), if -1 all past time-steps are returned and
                         the object size varies over time.
        :param horizon_l: number of time-steps in the prediction horizon (label)
        :param window_f: number of input (object) features or list of their names
        :param horizon_f: number of output (label) features or list of their names
        :param stride: number of time steps between the starting points of two consecutive windows,
                       if -1: stride = window_l
        :param params: additional, data set specific parameters as path to a json file or a dict
        """
        self.wl = window_l
        self.hl = horizon_l
        self.s = stride
        self._wf = window_f
        self._hf = horizon_f

        if isinstance(params, str):
            assert os.path.exists(params), f'Could not locate the dataset parameters in {params}.'
            with open(params, 'r') as params_file:
                params = json.load(params_file)
        self.params = params
        logging.debug(f'params: {self.params}')

        self.x = []  # objects
        self.y = []  # corresponding labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @property
    def wf(self) -> int:
        return self._wf if isinstance(self._wf, int) else len(self._wf)

    @property
    def hf(self) -> int:
        return self._hf if isinstance(self._hf, int) else len(self._hf)

    def w_start_idx(self, i: int) -> int:
        """
        Returns the index of the first timestep in an object in the original time series, before windowing
        :param i: index of the sample (= window) in the windowed dataset
        :return: starting index of the window in the original time series
        """
        if self.wl == -1:
            return 0
        return i * self.s

    def w_end_idx(self, i: int) -> int:
        """
        Returns the index of the first timestep after an object in the original time series, before windowing
        :param i: index of the sample (= window) in the windowed dataset
        :return: index after the window in the original time series
        """
        if self.wl == -1:
            return i * self.s
        return i * self.s + self.wl

    def h_start_idx(self, i: int) -> int:
        """
        Returns the index of the first timestep in a label in the original time series, before windowing
        :param i: index of the label (= horizon) in the windowed dataset
        :return: starting index of the horizon in the original time series
        """
        return self.w_end_idx(i)

    def h_end_idx(self, i: int) -> int:
        """
        Returns the index of the first timestep after a label in the original time series, before windowing
        :param i: index of the label (= horizon) in the windowed dataset
        :return: index after the horizon in the original time series
        """
        return self.h_start_idx(i) + self.hl

    def window(self, x_raw: np.ndarray, y_raw: np.ndarray) -> (list, list):
        """
        Creates a sliding window representation of multivariate timeseries data.
        The i-th window will be composed of
        x[i, :, :] = x_raw[i * self.s:i * self.s + self.wl, :]
        y[i, :, :] = y_raw[i * self.s + self.wl:i * self.s + self.wl + self.hl]
        Note: this method is not particularly efficient, as it creates new arrays for x and y instead of views of the
        original data set. It does however avoid undetermined behaviour when writing to overlapping windows as described
        here https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.as_strided.html.
        :param x_raw: [#time steps, self.wf]
        :param y_raw: [#time steps, self.hf]
        :return: x, y: [#windows, self.wl, self.wf], [#windows, self.hl, self.hf]
        """
        if self.wl == -1:
            nbr_w = ((x_raw.shape[0] - self.hl) // self.s) + 1  # total number of windows
        else:
            nbr_w = ((x_raw.shape[0] - self.wl - self.hl) // self.s) + 1  # total number of windows
        x = [None] * nbr_w
        y = [None] * nbr_w
        for i in range(nbr_w):
            x[i] = torch.tensor(x_raw[self.w_start_idx(i):self.w_end_idx(i), :].copy())
            y[i] = torch.tensor(y_raw[self.h_start_idx(i):self.h_end_idx(i), :].copy())
        return x, y

    def plot_samples(self, idxs: list, mode: str = 'lines') -> None:
        """
        Visualizes a list of samples from the data set.
        :param idxs: list of indexes of the samples to be visualized.
        :return: None
        """
        fig = plotly.graph_objs.Figure(layout={'title': 'Dataset Samples'})
        for i in idxs:
            for f in range(self.wf):
                fig.add_scatter(
                    x=[j for j in range(self.w_start_idx(i), self.w_end_idx(i))],
                    y=self.x[i][:, f],
                    mode=mode,
                    line={'dash': 'dot' if i % 2 == 0 else 'dash'},
                    name=f'w[{i}], x[:, {f}]'
                )
            for f in range(self.hf):
                fig.add_scatter(
                    x=[j for j in range(self.h_start_idx(i), self.h_end_idx(i))],
                    y=self.y[i][:, f],
                    mode=mode,
                    name=f'w[{i}], y[:, {f}]'
                )
        fig.show()

    def to(self, device: str) -> None:
        """
        Moves the dataset's tensors to the specified device.
        :param device: name of the device such as 'cpu' or 'cuda:0'
        :return: None
        """
        self.x = [x_i.to(device) for x_i in self.x]
        self.y = [y_i.to(device) for y_i in self.y]

    def append(self, x: torch.tensor, y: torch.tensor) -> None:
        """
        Aadds an element to the data set. It is up to the user to make sure this makes sense.
        :param x: new object
        :param y: new label
        :return:
        """
        self.x.append(x)
        self.y.append(y)

    def subset(self, idxs) -> 'Dataset':
        """
        Returns a subset of the data set.
        :param idxs: indexes of the elements that should be contained in the subset
        :return: subset of the data set containing the elements
        """
        ds = Dataset(self.wl, self.hl, self._wf, self._hf, self.s, self.params)
        ds.x = self.x[idxs]
        ds.y = self.y[idxs]
        return ds


class BarberIID(Dataset):
    """
    source: https://rinafb.github.io/code/nonexchangeable_conformal.zip
    """

    def __init__(self, window_l: int = 1, horizon_l: int = 1, window_f: int | list = 1, horizon_f: int | list = 1,
                 stride: int = 1, params: str | dict = None):
        super().__init__(window_l, horizon_l, window_f, horizon_f, stride, params)

        # default parameters from the Barber paper. Future: make these parametric.
        self.wl = 1
        self.hl = 1
        self._wf = 4  # p in the Berber version
        self._hf = 1

        t = 2000  # total number of time steps / instances in the data set. N in the Barber version.
        nf = 1.  # noise factor
        if self.params is not None:
            if 't' in self.params and (isinstance(self.params['t'], int)):
                t = self.params['t']
            self.params['t'] = t  # saving the default value if none is provided

            if 'random_seed' in self.params and (isinstance(self.params['random_seed'], int)):
                np.random.seed(self.params['random_seed'])

            if 'noise_factor' in self.params and (isinstance(self.params['noise_factor'], float)):
                nf = self.params['noise_factor']
        else:
            logging.debug('No parameters passed.')

        # i.i.d. data
        x = np.random.normal(size=(t, self.wl, self.wf))
        y = np.zeros((t, self.hl, self.hf))
        noise = np.random.normal(size=(t, self.hl, self.hf)) * nf
        beta = np.array([2, 1, 0, 0])  # magic numbers taken from the Barber paper
        y[:, 0, :] = x.dot(beta)
        y = y + noise

        self.x = [torch.tensor(x_i) for x_i in x]
        self.y = [torch.tensor(y_i) for y_i in y]


class BarberChangepoints(Dataset):
    """
    source: https://rinafb.github.io/code/nonexchangeable_conformal.zip
    """

    def __init__(self, window_l: int = 1, horizon_l: int = 1, window_f: int | list = 1, horizon_f: int | list = 1,
                 stride: int = 1, params: str | dict = None):
        super().__init__(window_l, horizon_l, window_f, horizon_f, stride, params)

        # default parameters from the Barber paper. Future: make these parametric.
        self.wl = 1
        self.hl = 1
        self._wf = 4  # p in the Berber version
        self._hf = 1

        t = 2000  # total number of time steps / instances in the data set. N in the Barber version.
        nf = 1.  # noise factor
        if self.params is not None:
            if 't' in self.params and (isinstance(self.params['t'], int)):
                t = self.params['t']
            self.params['t'] = t  # saving the default value if none is provided

            if 'random_seed' in self.params and (isinstance(self.params['random_seed'], int)):
                np.random.seed(self.params['random_seed'])

            if 'noise_factor' in self.params and (isinstance(self.params['noise_factor'], float)):
                nf = self.params['noise_factor']
        else:
            logging.debug('No parameters passed.')

        x = np.random.normal(size=(t, self.wl, self.wf))
        y = np.zeros((t, self.hl, self.hf))
        noise = np.random.normal(size=(t, self.hl, self.hf)) * nf

        changepoints = np.r_[500, 1500]
        n_changepoint = len(changepoints)
        beta = np.array([[2, 1, 0, 0], [0, -2, -1, 0], [0, 0, 2, 1]])  # magic numbers taken from the Barber paper

        for i in np.arange(n_changepoint + 1):
            if i == 0:
                ind_min = 0
            else:
                ind_min = changepoints[i - 1]
            if i == n_changepoint:
                ind_max = t
            else:
                ind_max = changepoints[i]
            y[ind_min:ind_max, 0, :] = x[ind_min:ind_max].dot(beta[i])
        y = y + noise

        self.x = [torch.tensor(x_i) for x_i in x]
        self.y = [torch.tensor(y_i) for y_i in y]


class BarberDistributionDrift(Dataset):
    """
    source: https://rinafb.github.io/code/nonexchangeable_conformal.zip
    """

    def __init__(self, window_l: int = 1, horizon_l: int = 1, window_f: int | list = 1, horizon_f: int | list = 1,
                 stride: int = 1, params: str | dict = None):
        super().__init__(window_l, horizon_l, window_f, horizon_f, stride, params)

        # default parameters from the Barber paper. Future: make these parametric.
        self.wl = 1
        self.hl = 1
        self._wf = 4  # p in the Berber version
        self._hf = 1

        t = 2000  # total number of time steps / instances in the data set. N in the Barber version.
        nf = 1.  # noise factor
        if self.params is not None:
            if 'random_seed' in self.params and (isinstance(self.params['random_seed'], int)):
                np.random.seed(self.params['random_seed'])

            if 'noise_factor' in self.params and (isinstance(self.params['noise_factor'], float)):
                nf = self.params['noise_factor']
        else:
            logging.debug('No parameters passed.')

        # i.i.d. data
        x = np.random.normal(size=(t, self.wl, self.wf))
        y = np.zeros((t, self.hl, self.hf))
        noise = np.random.normal(size=(t, self.hl, self.hf)) * nf

        beta_start = np.array([2, 1, 0, 0])
        beta_end = np.array([0, 0, 2, 1])
        beta = beta_start + np.outer(np.arange(t) / (t - 1), beta_end - beta_start)

        for i in np.arange(t):
            y[i, :, :] = x[i, :, :].dot(beta[i])
        y = y + noise

        self.x = [torch.tensor(x_i) for x_i in x]
        self.y = [torch.tensor(y_i) for y_i in y]


class Stankeviciute(Dataset):
    """
    source: https://github.com/kamilest/conformal-rnn
    Although I simplified the data generating process heavily, removing some flexibility.
    """

    def __init__(self, window_l: int = 15, horizon_l: int = 5, window_f: int | list = 1, horizon_f: int | list = 1,
                 stride: int = 20, params: str | dict = None):
        super().__init__(window_l, horizon_l, window_f, horizon_f, stride, params)

        assert self.wf == self.hf, 'This dataset requires the same number of input and output features.'

        self.n = 2000  # number of samples
        self.seed_mean = 1.
        self.seed_variance = 2.
        self.memory_factor = 0.9
        self.noise_factor = 0.1  # to 0.5, scales the noise

        if self.params is not None:
            if 'n' in self.params:
                self.n = self.params['n']
            if 'seed_mean' in self.params:
                self.seed_mean = self.params['seed_mean']
            if 'seed_variance' in self.params:
                self.seed_variance = self.params['seed_variance']
            if 'memory_factor' in self.params:
                self.memory_factor = self.params['memory_factor']
            if 'noise_factor' in self.params:
                self.noise_factor = self.params['noise_factor']

        self.x, self.y = self.generate_sequences()

    def generate_sequences(self):
        sl = self.wl + self.hl  # sequence length
        seed = np.random.normal(self.seed_mean, self.seed_variance, (self.n, sl, self.wf))  # random sequences
        s = np.zeros((self.n, sl, self.hf))  # dependent sequences
        s[:, 0] = self.memory_factor * seed[:, 0]
        for t in range(1, sl):
            s[:, t] = self.memory_factor * (s[:, t - 1] + seed[:, 0])  # autoregressive
        s = s + np.random.normal(0, self.noise_factor, (self.n, sl, self.wf))

        x = s[:, :self.wl]
        y = s[:, -self.hl:]

        return [torch.tensor(x_i) for x_i in x], [torch.tensor(y_i) for y_i in y]



class Elec2(Dataset):

    def __init__(self, window_l: int = -1, horizon_l: int = 1,
                 window_f: int | list = 1, horizon_f: int | list = 1, stride: int = -1,
                 params: str | dict = None):
        super().__init__(window_l, horizon_l, window_f, horizon_f, stride, params)
        self.raw_df = Elec2._load_from_disk(self.params['csv_path'])
        self.x, self.y = self._preprocess()

    @staticmethod
    def _load_from_disk(csv_path: str = None) -> pd.DataFrame:
        """
        Loads the csv file containing the data set and returns it as a pandas DataFrame.
        :param csv_path: path to the cav file containing the data set.
        :return: entire data set.
        """
        assert os.path.exists(csv_path), f'Could not locate the elec2 dataset in {csv_path}.'
        return pd.read_csv(csv_path)

    def _preprocess(self):
        """
        Preprocesses the ELEC2 data set by cutting it into windows of the desired dimensions.
        Since the data set was already normalized and preprocessed by someone else there is not much to do here.
        :return: windowed version of the data set.
        """
        # selecting the desired features for the objects and labels
        logging.debug(f'self.wf: {self._wf}, type(self.wf): {type(self._wf)}')
        logging.debug(f'self.hf: {self._hf}, type(self.hf): {type(self._hf)}')
        assert isinstance(self._wf, list), 'List object features explicitly for this data set.'
        assert isinstance(self._hf, list), 'List label features explicitly for this data set.'
        x_raw = self.raw_df.loc[:, self._wf].to_numpy()
        y_raw = self.raw_df.loc[:, self._hf].to_numpy()

        # trimming the data set according to the optional parameters
        start_idx = 0
        end_idx = len(x_raw)
        if 'start_idx' in self.params:
            start_idx = self.params['start_idx']
        if 'end_idx' in self.params:
            end_idx = self.params['end_idx']
        x_raw = x_raw[start_idx:end_idx]
        y_raw = y_raw[start_idx:end_idx]

        return self.window(x_raw, y_raw)


class DatasetFactory(ABC):
    known_ds = {
        # synthetic data sets
        SyntheticDatasets.barber_iid: BarberIID,
        SyntheticDatasets.barber_changepoints: BarberChangepoints,
        SyntheticDatasets.barber_distribution_drift: BarberDistributionDrift,
        SyntheticDatasets.feldman: None,  # todo: implement
        SyntheticDatasets.stankeviciute: Stankeviciute,
        # real-world data sets
        RealWorldDatasets.elec2: Elec2
    }

    @staticmethod
    def create(dataset: SyntheticDatasets | RealWorldDatasets, window_l: int = -1, horizon_l: int = 1,
               window_f: int | list = 1, horizon_f: int | list = 1, stride: int = -1,
               params: str | dict = None, **kwargs) -> Dataset:
        """
        This method instantiates the chosen dataset. The factory pattern is chosen for the convenience of using Enum to
        identify the individual datasets.
        :param dataset: dataset that is to be loaded
        :param window_l: number of past time-steps for every window (object), if -1 all past time-steps are returned and
                         the object size varies over time.
        :param horizon_l: number of time-steps in the prediction horizon (label)
        :param window_f: number of input (object) features or list of their names
        :param horizon_f: number of output (label) features or list of their names
        :param stride: number of time steps between the starting points of two consecutive windows,
                       if -1: stride = window_l
        :param params: additional, data set specific parameters as path to a json file or a dict
        :return: pytorch data set
        """
        if kwargs:
            for k, v in kwargs.items():
                logging.info(f'Parameter {k} = {v} will be ignored.')

        return DatasetFactory.known_ds[dataset](window_l, horizon_l, window_f, horizon_f, stride, params)
