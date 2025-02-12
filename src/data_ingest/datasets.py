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
    synth_changepoints = 'synth_changepoints'
    synth_distribution_drift = 'synth_distribution_drift'


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


class SyntheticChangepoints(Dataset):
    # todo: add description

    def __init__(self, window_l: int = 6, horizon_l: int = 6, window_f: int | list = 4, horizon_f: int | list = 2,
                 stride: int = 6, params: str | dict = None):
        super().__init__(window_l, horizon_l, window_f, horizon_f, stride, params)

        if not ((self.wl == self.hl) and (self.wl == self.hl)):
            raise ValueError('For this data set window length, horizon length and stride must be equal.')

        # default parameters
        self.n_days = 200
        self.time_steps_per_day = 24
        self.n_time_steps = self.time_steps_per_day * self.n_days
        self.change_points = [self.time_steps_per_day * 150, self.n_time_steps]
        self.betas = [np.array([[0.7, 0],
                                [0.3, 0],
                                [0, 0.4],
                                [0, 0.6]]),
                      np.array([[0, 0.6],
                                [0.7, 0],
                                [0.3, 0],
                                [0, 0.4]])]
        if self.params is not None:
            if 'n_days' in self.params and (isinstance(self.params['n_days'], int)):
                self.n_days = self.params['n_days']
            if 'time_steps_per_day' in self.params and (isinstance(self.params['time_steps_per_day'], int)):
                self.time_steps_per_day = self.params['time_steps_per_day']
            self.n_time_steps = self.time_steps_per_day * self.n_days
            if 'change_points' in self.params:
                self.change_points = self.params['change_points']
                self.change_points.append(self.n_time_steps)
            if 'betas' in self.params:
                self.betas = [np.array(beta) for beta in self.params['betas']]
        self.base_frequency = self.n_days

        # generating the four features of our objects.
        x_generating_process = np.zeros((self.n_time_steps, self.wf))
        x_generating_process[:, 0] = np.sin(
            np.linspace(0, 2 * np.pi * self.base_frequency, self.n_time_steps))  # sine wave daily
        x_generating_process[:, 1] = np.sin(
            np.linspace(0, 2 * np.pi * self.base_frequency / 7, self.n_time_steps))  # sine wave weekly
        x_generating_process[:, 2] = np.sign(
            -np.sin(np.linspace(0, 2 * np.pi * self.base_frequency / np.pi, self.n_time_steps)))  # square wave
        x_generating_process[:, 3] = np.array([1 if i % (7 * 24) < (5 * 24) else -1 for i in range(self.n_time_steps)])

        self.x = []
        self.y = []

        change_point_i = 0
        for slice in range(self.n_time_steps // self.s):
            t_start = slice * self.s
            t_end = (slice + 1) * self.s
            if t_start >= self.change_points[change_point_i]:
                change_point_i += 1

            # ex_i = np.cumsum(np.random.normal(size=(self.wl, self.wf)) / 10, axis=0)
            # x_i = x_generating_process[t_start:t_end, :] + ex_i
            x_i = x_generating_process[t_start:t_end, :]
            ey_i = np.cumsum(np.random.normal(size=(self.hl, self.hf)) / 10, axis=0)
            y_i = x_i.dot(self.betas[change_point_i]) + ey_i

            self.x.append(torch.tensor(x_i))
            self.y.append(torch.tensor(y_i))


class SyntheticDistributionDrift(Dataset):
    # todo: add description

    def __init__(self, window_l: int = 6, horizon_l: int = 6, window_f: int | list = 4, horizon_f: int | list = 2,
                 stride: int = 6, params: str | dict = None):
        super().__init__(window_l, horizon_l, window_f, horizon_f, stride, params)

        if not ((self.wl == self.hl) and (self.wl == self.hl)):
            raise ValueError('For this data set window length, horizon length and stride must be equal.')

        # default parameters
        self.n_days = 200
        self.time_steps_per_day = 24
        self.n_time_steps = self.time_steps_per_day * self.n_days
        self.change_points = [0, self.time_steps_per_day * 150, self.n_time_steps]
        self.betas = [np.array([[0.7, 0],
                                [0.3, 0],
                                [0, 0.4],
                                [0, 0.6]]),
                      np.array([[0.7, 0],
                                [0.3, 0],
                                [0, 0.4],
                                [0, 0.6]]),
                      np.array([[0, 0.6],
                                [0.7, 0],
                                [0.3, 0],
                                [0, 0.4]])]
        if self.params is not None:
            if 'n_days' in self.params and (isinstance(self.params['n_days'], int)):
                self.n_days = self.params['n_days']
            if 'time_steps_per_day' in self.params and (isinstance(self.params['time_steps_per_day'], int)):
                self.time_steps_per_day = self.params['time_steps_per_day']
            self.n_time_steps = self.time_steps_per_day * self.n_days
            if 'change_points' in self.params:
                self.change_points = self.params['change_points']
            if 'betas' in self.params:
                self.betas = [np.array(beta) for beta in self.params['betas']]
        self.base_frequency = self.n_days

        # generating the four features of our objects.
        x_generating_process = np.zeros((self.n_time_steps, self.wf))
        x_generating_process[:, 0] = np.sin(
            np.linspace(0, 2 * np.pi * self.base_frequency, self.n_time_steps))  # sine wave daily
        x_generating_process[:, 1] = np.sin(
            np.linspace(0, 2 * np.pi * self.base_frequency / 7, self.n_time_steps))  # sine wave weekly
        x_generating_process[:, 2] = np.sign(
            -np.sin(np.linspace(0, 2 * np.pi * self.base_frequency / np.pi, self.n_time_steps)))  # square wave
        x_generating_process[:, 3] = np.array([1 if i % (7 * 24) < (5 * 24) else -1 for i in range(self.n_time_steps)])

        # compute the interpolation for the values of beta between change points
        self.n_slices = self.n_time_steps // self.s
        n_changepoints = len(self.change_points)
        self.beta_series = np.zeros((self.n_time_steps, self.wf, self.hf))

        for ch_p_i in range(n_changepoints - 1):
            beta_start = self.betas[ch_p_i]
            beta_end = self.betas[ch_p_i + 1]
            beta_start_i = self.change_points[ch_p_i]
            beta_end_i = self.change_points[ch_p_i + 1]
            n_beta_steps = beta_end_i - beta_start_i
            self.beta_series[beta_start_i:beta_end_i, :, :] = beta_start + np.outer(
                np.arange(n_beta_steps) / (n_beta_steps + 1),
                beta_end - beta_start).reshape(
                [n_beta_steps, self.wf, self.hf])

        self.x = []
        self.y = []

        for slice in range(self.n_slices):
            t_start = slice * self.s
            t_end = (slice + 1) * self.s

            x_i = x_generating_process[t_start:t_end, :]
            ey_i = np.cumsum(np.random.normal(size=(self.hl, self.hf)) / 10, axis=0)
            y_i = x_i.dot(self.beta_series[t_start, :]) + ey_i  # todo: skipping self.s - 1 values of beta like this.

            self.x.append(torch.tensor(x_i))
            self.y.append(torch.tensor(y_i))


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
        SyntheticDatasets.synth_changepoints: SyntheticChangepoints,
        SyntheticDatasets.synth_distribution_drift: SyntheticDistributionDrift,
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
