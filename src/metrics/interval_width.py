import logging

import torch

import matplotlib.pyplot as plt
import numpy as np
from metrics.metric import Metric, RollingMetric


class MeanIntervalWidth(Metric):
    """
    Computes the average interval width for every significance level.
    """

    @staticmethod
    def name() -> str:
        return 'Mean Interval Width'

    @staticmethod
    def snake_name() -> str:
        return 'mean_interval_width'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, miw_d = MeanIntervalWidthByDimension.eval(y, y_hat, **kwargs)
        miw = np.mean(miw_d, axis=(1, 2))
        # list of alpha values and, corresponding mean interval width for every alpha
        return alpha, miw

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        alpha, miw = result

        plt.plot([1 - a for a in alpha], [miw[i] for i in range(len(miw))],
                 label=f'interval width')
        plt.title(title)
        plt.legend()
        plt.xlim([0, 1.1])
        plt.xlabel('1 - α')
        plt.ylabel('average interval width')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()
        plt.close('all')

    @staticmethod
    def comparative_plot(results: list, labels: list[str], title: str = None, save_path: str = None,
                         display: bool = True, fig_size=(4, 4)):
        """
        Plots multiple result objects onto the same graph.
        :param results: list of tuples (alpha, cr)
        :param labels: labels associated with the different results
        :param title: title for the graph
        :param save_path: path where the graph will be saved
        :param display: show the result ot not
        :param fig_size: plt figsize
        :return:
        """
        fig = plt.figure(figsize=fig_size)
        for i in range(len(results)):
            alpha, miw = results[i]
            epsilon = 1 - alpha
            plt.plot(epsilon, miw, label=labels[i])
        if title is not None:
            plt.title(title)
        plt.legend(loc='lower center', bbox_to_anchor=(0.45, -0.4), ncol=3, fontsize='x-small')
        plt.xlabel('1 - $\\alpha$')
        plt.ylabel('mean interval width')
        plt.ylim(0)

        # plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()
        plt.close('all')


class MeanIntervalWidthByDimension(Metric):

    @staticmethod
    def name() -> str:
        return 'Mean Interval Width by Dimension'

    @staticmethod
    def snake_name() -> str:
        return 'mean_interval_width_by_dimension'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        # y_hat_stack: [|y_hat_keys|-1, n_samples, hl, hf, [lower & upper]]
        y_hat_stack, y_hat_keys = Metric.stack_y_hat(y_hat)
        lower, upper = y_hat_stack[:, :, :, :, 0], y_hat_stack[:, :, :, :, 1]
        interval_width = torch.abs(upper - lower)  # [|y_hat_keys|-1, n_samples, hl, hf]
        dimension_width = torch.mean(interval_width, dim=1)  # [|y_hat_keys|-1, hl, hf]
        # list of alpha values and, corresponding dimension-wise mean interval width for every alpha
        return np.array(y_hat_keys), np.array(dimension_width.detach())

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # horizontally stacked sub-figs, one for every feature. Inside widths for every time step.
        alpha, miw_d = result

        hf = miw_d.shape[2]
        fig, axs = plt.subplots(hf, 1, figsize=(8, 6 * hf), squeeze=False)
        for f in range(hf):
            axs[f, 0].set_xlabel('future time step')
            axs[f, 0].set_ylabel(f'feature {f}')
            for i, a in enumerate(alpha):
                axs[f, 0].plot(miw_d[i, :, f], label=f'1 - {alpha}')
        plt.title(title)
        # plt.legend()
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        if display:
            fig.show()
        plt.clf()
        plt.close('all')


class MeanIntervalWidthByFeature(Metric):

    @staticmethod
    def name() -> str:
        return 'Mean Interval Width by Feature'

    @staticmethod
    def snake_name() -> str:
        return 'mean_interval_width_by_feature'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, miw_d = MeanIntervalWidthByDimension.eval(y, y_hat, **kwargs)
        miw_f = np.mean(miw_d, axis=1)
        # list of alpha values and, corresponding time-step-wise mean interval width for every alpha
        return alpha, miw_f

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        alpha, miw_f = result

        fig = plt.figure(figsize=(8, 6))
        for i, a in enumerate(alpha):
            plt.plot(miw_f[i], label=f'1 - {a}')
        plt.xlabel('feature')
        plt.title(title)
        # plt.legend()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()
        plt.close('all')


class MeanIntervalWidthByTimeStep(Metric):

    @staticmethod
    def name() -> str:
        return 'Mean Interval Width by Time Step'

    @staticmethod
    def snake_name() -> str:
        return 'mean_interval_width_by_time_step'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, miw_d = MeanIntervalWidthByDimension.eval(y, y_hat, **kwargs)
        miw_ts = np.mean(miw_d, axis=2)
        # list of alpha values and, corresponding time-step-wise mean interval width for every alpha
        return alpha, miw_ts

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        alpha, miw_ts = result

        fig = plt.figure(figsize=(8, 6))
        for i, a in enumerate(alpha):
            plt.plot(miw_ts[i], label=f'1 - {alpha}')
        plt.xlabel('future time step')
        plt.title(title)
        # plt.legend()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()
        plt.close('all')


class RollingMeanIntervalWidth(RollingMetric):
    """
    Computes the average interval width for every significance level.
    """

    @staticmethod
    def name() -> str:
        return 'Rolling Mean Interval Width'

    @staticmethod
    def snake_name() -> str:
        return 'rolling_mean_interval_width'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, rmiw_d, window_end_idx = RollingMeanIntervalWidthByDimension.eval(y, y_hat, **kwargs)
        # rmiw_d: [|y_hat_keys|-1, n_windows, hl, hf]
        rmiw = np.mean(rmiw_d, axis=(2, 3))
        # list of alpha values and, corresponding rolling mean interval width for every alpha, window_end_idx
        return alpha, rmiw, window_end_idx

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        alpha, rmiw, window_end_idx = result

        fig = plt.figure(figsize=(8, 6))
        for i, a in enumerate(alpha):
            plt.plot(window_end_idx, rmiw[i], label=f'1 - {a}')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()
        plt.close('all')


class RollingMeanIntervalWidthByDimension(RollingMetric):

    @staticmethod
    def name() -> str:
        return 'Rolling Mean Interval Width by Dimension'

    @staticmethod
    def snake_name() -> str:
        return 'rolling_mean_interval_width_by_dimension'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        rolling_window_length = 300
        if 'rolling_window_length' in kwargs:
            rolling_window_length = kwargs.get('rolling_window_length')
        assert rolling_window_length < len(y), 'rolling window must be smaller than total number of examples'

        # y_hat_stack: [|y_hat_keys|-1, n_samples, hl, hf, [lower & upper]]
        y_hat_stack, y_hat_keys = Metric.stack_y_hat(y_hat)
        lower, upper = y_hat_stack[:, :, :, :, 0], y_hat_stack[:, :, :, :, 1]
        interval_width = torch.abs(upper - lower).detach()  # [|y_hat_keys|-1, n_samples, hl, hf]

        interval_width = np.array(interval_width)
        windowed_interval_width = np.lib.stride_tricks.sliding_window_view(interval_width,
                                                                           window_shape=rolling_window_length,
                                                                           axis=1)
        rmiw_d = np.mean(windowed_interval_width, axis=-1)  # [|y_hat_keys|-1, n_windows, hl, hf]
        window_end_idx = np.array([i for i in range(rolling_window_length - 1, len(y))])
        # list of alpha values and, corresponding dimension-wise rolling mean interval width for every alpha,
        # window_end_idx
        return np.array(y_hat_keys), rmiw_d, window_end_idx

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        logging.warning('No representation available.')


class RollingMeanIntervalWidthByFeature(RollingMetric):

    @staticmethod
    def name() -> str:
        return 'Rolling Mean Interval Width by Feature'

    @staticmethod
    def snake_name() -> str:
        return 'rolling_mean_interval_width_by_feature'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, rmiw_d, window_end_idx = RollingMeanIntervalWidthByDimension.eval(y, y_hat, **kwargs)
        # rmiw_d: [|alpha|, n_windows, hl, hf]
        rmiw_f = np.mean(rmiw_d, axis=2)  # [|alpha|, n_windows, hf]
        # list of alpha values and, corresponding rolling mean interval width for every alpha, window_end_idx
        return alpha, rmiw_f, window_end_idx

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # horizontally stacked sub-figs, one for every feature. Inside widths for every window.
        alpha, rmiw_f, window_end_idx = result

        hf = rmiw_f.shape[2]
        fig, axs = plt.subplots(1, hf, figsize=(8 * hf, 6), squeeze=False)
        for f in range(hf):
            axs[0, f].set_xlabel('window')
            axs[0, f].set_ylabel(f'feature {f}')
            for i, a in enumerate(alpha):
                axs[0, f].plot(window_end_idx, rmiw_f[i, :, f], label=f'1 - {alpha}')
        plt.title(title)
        # plt.legend()
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        if display:
            fig.show()
        plt.clf()
        plt.close('all')


class RollingMeanIntervalWidthByTimeStep(RollingMetric):

    @staticmethod
    def name() -> str:
        return 'Rolling Mean Interval Width by Time Step'

    @staticmethod
    def snake_name() -> str:
        return 'rolling_mean_interval_width_by_time_step'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, rmiw_d, window_end_idx = RollingMeanIntervalWidthByDimension.eval(y, y_hat, **kwargs)
        # rmiw_d: [|alpha|, n_windows, hl, hf]
        rmiw_ts = np.mean(rmiw_d, axis=3)  # [|alpha|, n_windows, hl]
        # list of alpha values and, corresponding rolling mean interval width for every alpha, window_end_idx
        return alpha, rmiw_ts, window_end_idx

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # horizontally stacked sub-figs, one for every time step. Inside widths for every window.
        alpha, rmiw_ts, window_end_idx = result

        hl = rmiw_ts.shape[2]
        fig, axs = plt.subplots(1, hl, figsize=(8 * hl, 6), squeeze=False)
        for ts in range(hl):
            axs[0, ts].set_xlabel('window')
            axs[0, ts].set_ylabel(f'time step {ts}')
            for i, a in enumerate(alpha):
                axs[0, ts].plot(window_end_idx, rmiw_ts[i, :, ts], label=f'1 - {alpha}')
        plt.title(title)
        # plt.legend()
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        if display:
            fig.show()
        plt.clf()
        plt.close('all')
