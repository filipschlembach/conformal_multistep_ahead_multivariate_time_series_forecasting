import logging
from abc import abstractmethod, ABC

import numpy as np
from pathlib import Path
import torch


class Metric(ABC):
    """
    Parent class for all metrics that are used to evaluate the results produced by different conformal prediction
    methods.
    """

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        :return: descriptive human-readable name of the metric
        """
        pass

    @staticmethod
    @abstractmethod
    def snake_name() -> str:
        """
        :return: descriptive snake case name of the metric
        """
        pass

    @staticmethod
    @abstractmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        """
        Evaluates the produced prediction intervals.
        :param y: Ground truth, [batch size, hl, |hf|]
        :param y_hat: Prediction intervals, {
                        'model': Forecast of the underlying model with shape [batch size, hl, |hf|, 0 | [lower, upper]],
                        alpha[0]: Prediction intervals with shape [batch size, hl, |hf|, [lower, upper]],
                        ...
                        alpha[k]: Prediction intervals with shape [batch size, hl, |hf|, [lower, upper]]
                    }
        :param kwargs: Possible metric specific parameters
        :return: Evaluation result
        """
        pass

    @classmethod
    def save(cls, result, save_path: str, max_trials: int = 100, logger: logging.Logger = None) -> None:
        """
        Saves the result produced by the eval method to the file system.
        :param result: return object of the eval method.
        :param save_path: Path including the filename WITHOUT THE FILE TYPE EXTENSION, that is added by the save method.
        :param max_trials: maximum number of times the function tries to save the file if it fails to do so previously.
        :param logger: Loger object to be used. If none is passed a new one is created.
        :return: None
        """
        cls._secure_save_npz({'alpha': result[0], cls.snake_name(): result[1]}, save_path, max_trials, logger)

    @classmethod
    def _secure_save_npz(cls, result: dict, save_path: str, max_trials: int = 100,
                         logger: logging.Logger = None) -> None:
        # todo: documentation
        save_path = Path(save_path + '.npz')
        n_trials = 0  # how often an attempt to save the file has been made

        if logger is None:
            logger = logging.getLogger('_secure_save_npz')
        logger.debug('fisch entering _secure_save_npz')  # todo: remove after debugging

        while (n_trials >= 0) and (n_trials < max_trials):
            with save_path.open('wb') as f:
                np.savez(f, **result)
            if save_path.exists():
                logger.debug(f'successfully saved {save_path} after {n_trials}.')
                n_trials = -1
            else:
                logger.warning(f'could not save {save_path} after {n_trials}, trying again...')
                n_trials += 1
        if n_trials >= max_trials:
            logger.warning(f'could not save {save_path} after {n_trials}, reaching the max number of trials.')
        logger.debug('fisch leaving _secure_save_npz')  # todo: remove after debugging

    @classmethod
    def _secure_save_npy(cls, result: np.ndarray, save_path: str, max_trials: int = 100,
                         logger: logging.Logger = None) -> None:
        # todo: documentation
        save_path = Path(save_path + '.npy')
        n_trials = 0  # how often an attempt to save the file has been made

        if logger is None:
            logger = logging.getLogger('_secure_save_npy')
        logger.debug('fisch entering _secure_save_npy')  # todo: remove after debugging

        while (n_trials >= 0) and (n_trials < max_trials):
            with save_path.open('wb') as f:
                np.save(f, result)
            if save_path.exists():
                logger.debug(f'successfully saved {save_path} after {n_trials}.')
                n_trials = -1
            else:
                logger.warning(f'could not save {save_path} after {n_trials}, trying again...')
                n_trials += 1
        if n_trials >= max_trials:
            logger.warning(f'could not save {save_path} after {n_trials}, reaching the max number of trials.')
        logger.debug('fisch leaving _secure_save_npy')  # todo: remove after debugging

    @classmethod
    def load(cls, save_path: str):
        """
        Loads the result produced by the eval method and saved with the save method from the file system.
        :param save_path: Path including the filename WITHOUT THE FILE TYPE EXTENSION, that is added by the save method.
        :return: result object
        """
        npz_file = np.load(save_path + '.npz')
        return npz_file['alpha'], npz_file[cls.snake_name()]

    @staticmethod
    @abstractmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        """
        Provides a simple plot associated with the result produced by the eval method.
        This method is meant to gain a quick overview over the result.
        :param result: return object of the eval method.
        :param title: plot title
        :param save_path: location for the plot to be stored
        :param display: display the figure
        :return:
        """
        pass

    @classmethod
    def aggregate(cls, results: list):
        """
        Aggregates the results from multiple trials of the same experiment.
        :param results: list of return objects of the eval method.
        :return: aggregated result. The format depends on what makes sense but usually something similar to the return
                 object of the eval method.
        """
        alphas, metric_vals = zip(*results)
        # sanity check
        for i in range(1, len(results)):
            assert np.all(alphas[0] == alphas[i]), f'{cls.snake_name()}: alphas do not match for trial 0 and {i}'
        # aggregating by mean
        metric_vals = np.array(metric_vals)
        metric_val_mean = np.mean(metric_vals, axis=0)
        return alphas[0], metric_val_mean

    @staticmethod
    def stack_y_hat(y_hat: dict[str | float, torch.Tensor]) -> (torch.Tensor, list[float]):
        """
        Stacks the y_hat dict so that it is more efficient to slice.
        :param y_hat: y hat dict from the eval method
        :return: stacked y_hat values of shape [len(y_hat_keys)-1, n_samples, hl, hf, 2(lower & upper)]
                 and list of associated alphas of shape [len(y_hat_keys)-1]
        """
        y_hat_stack = []
        y_hat_keys = []
        for a in y_hat.keys():
            if a == 'model':
                continue
            y_hat_stack.append(y_hat[a])
            y_hat_keys.append(a)
        y_hat_stack = torch.stack(y_hat_stack)  # [len(y_hat_keys), n_samples, hl, hf, 2(lower & upper)]
        return y_hat_stack, y_hat_keys


class RollingMetric(Metric, ABC):

    @classmethod
    def save(cls, result, save_path: str, max_trials: int = 100, logger: logging.Logger = None) -> None:
        # assumption result = (alpha: [float], result: [], window_end_idx: [int])
        cls._secure_save_npz({'alpha': result[0], cls.snake_name(): result[1], 'window_end_idx': result[2]}, save_path,
                             max_trials, logger)

    @classmethod
    def load(cls, save_path: str):
        npz_file = np.load(save_path + '.npz')
        return npz_file['alpha'], npz_file[cls.snake_name()], npz_file['window_end_idx']

    @classmethod
    def aggregate(cls, results: list):
        alphas, metric_vals, window_end_idxs = zip(*results)
        # sanity check
        for i in range(1, len(results)):
            assert np.all(alphas[0] == alphas[i]), f'{cls.snake_name()}: alphas do not match for trial 0 and {i}'
            assert np.all(window_end_idxs[0] == window_end_idxs[i]), f'{cls.snake_name()}: window_end_idxs do not ' \
                                                                     f'match for trial 0 and {i}'
        # aggregating by mean
        metric_vals = np.array(metric_vals)
        metric_val_mean = np.mean(metric_vals, axis=0)
        return alphas[0], metric_val_mean, window_end_idxs[0]
