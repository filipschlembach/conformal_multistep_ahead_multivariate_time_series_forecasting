import logging

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

from metrics.metric import Metric, RollingMetric


class MAE(Metric):
    """
    Computes the achieved coverage rate for every significance level.
    A prediction is counted as being covered, if all values of the label are within the predicted hyper rectangle.
    """

    @staticmethod
    def name() -> str:
        return 'MAE'

    @staticmethod
    def snake_name() -> str:
        return 'mae'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        y_hat_model = y_hat['model']
        assert y_hat_model.shape == y.shape, f'Shape mismatch between y_hat_model ' \
                                             f'({y_hat_model.shape}) and y ({y.shape})'
        return float(torch.mean(torch.abs(y - y_hat_model)).cpu().detach())  # / (y.shape[1] * y.shape[2])

    @classmethod
    def save(cls, result, save_path: str, max_trials: int = 100, logger: logging.Logger = None) -> None:
        cls._secure_save_npy(np.array([result]), save_path, max_trials, logger)

    @classmethod
    def load(cls, save_path: str):
        return np.load(save_path + '.npy')[0]

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        logging.warning('No representation available.')
        if display:
            print(f'mae = {result}')

    @classmethod
    def aggregate(cls, results: list):
        return np.mean(np.array(results))


class RollingMAE(RollingMetric):
    """
    Rolling average coverage rate.
    """

    @staticmethod
    def name() -> str:
        return 'Rolling MAE'

    @staticmethod
    def snake_name() -> str:
        return 'rolling_mae'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        rolling_window_length = 300
        if 'rolling_window_length' in kwargs:
            rolling_window_length = kwargs.get('rolling_window_length')
        assert rolling_window_length < len(y), 'rolling window must be smaller than total number of examples'

        y_hat_model = y_hat['model']
        rolling_mae = {
            'mae': [],
            'i': []
        }

        # computing the local mae for every window
        for i in range(len(y) - rolling_window_length):
            y_window = y[i: i + rolling_window_length]
            y_hat_window = y_hat_model[i: i + rolling_window_length]
            rolling_mae['mae'].append(torch.mean(torch.abs(y_window - y_hat_window)).detach())
            rolling_mae['i'].append(i + rolling_window_length)

        # turning the lists into numpy arrays for easier future use
        rolling_mae['mae'] = np.array(rolling_mae['mae'])
        rolling_mae['i'] = np.array(rolling_mae['i'])

        return np.array(rolling_mae['mae']), np.array(rolling_mae['i'])

    @classmethod
    def save(cls, result, save_path: str, max_trials: int = 100, logger: logging.Logger = None) -> None:
        cls._secure_save_npz({cls.snake_name(): result[0], 'window_end_idx': result[1]}, save_path,
                             max_trials, logger)

    @classmethod
    def load(cls, save_path: str):
        npz_file = np.load(save_path + '.npz')
        return npz_file[cls.snake_name()], npz_file['window_end_idx']

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        mae, window_end_idx = result
        plt.plot(window_end_idx, mae, label='mae')
        plt.title(title)
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()

    @classmethod
    def aggregate(cls, results: list):
        r_means, window_end_idxs = zip(*results)
        # sanity check
        for i in range(1, len(results)):
            assert np.all(window_end_idxs[0] == window_end_idxs[i]), f'{cls.snake_name()}: window_end_idxs do not ' \
                                                                     f'match for trial 0 and {i}'
        # aggregating by mean
        r_means = np.array(r_means)
        metric_val_mean = np.mean(r_means, axis=0)
        return metric_val_mean, window_end_idxs[0]
