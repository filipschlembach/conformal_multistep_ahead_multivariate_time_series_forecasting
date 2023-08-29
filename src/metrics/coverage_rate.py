import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch

from metrics.metric import Metric, RollingMetric


class CoverageRate(Metric):
    """
    Computes the achieved coverage rate for every significance level.
    A prediction is counted as being covered, if all values of the label are within the predicted hyper rectangle.
    """

    @staticmethod
    def name() -> str:
        return 'Coverage Rate'

    @staticmethod
    def snake_name() -> str:
        return 'coverage_rate'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, coverage_d = CoverageRateByDimension.raw_coverage(y, y_hat, **kwargs)
        coverage_ts = np.all(coverage_d, axis=3)  # [|alpha|, batch size, hl, |hf|] -> [|alpha|, batch size, hl]
        coverage = np.all(coverage_ts, axis=2)  # [|alpha|, batch size, hl] -> [|alpha|, batch size]
        cr = np.mean(coverage, axis=1)  # [|alpha|, batch size] -> [|alpha|]
        # list of alpha values and, corresponding coverage rate for every alpha
        return alpha, cr

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        alpha, cr_d = result
        epsilon = 1 - alpha
        fig = plt.figure(figsize=(4, 4))
        plt.plot(epsilon, cr_d,
                 label=f'empirical coverage rate')
        plt.plot([i for i in range(2)], [i for i in range(2)], label='targeted coverage rate')
        plt.title(title)
        plt.legend()
        plt.xlim([0, 1.1])
        plt.ylim([0, 1.1])
        plt.xlabel('1 - α')
        plt.ylabel('coverage rate')
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
            alpha, cr_d = results[i]
            epsilon = 1 - alpha
            plt.plot(epsilon, cr_d, label=labels[i])
        plt.plot([i for i in range(2)], [i for i in range(2)], label=labels[-1])
        if title is not None:
            plt.title(title)
        plt.legend(loc='lower center', bbox_to_anchor=(0.45, -0.4), ncol=3, fontsize='x-small')
        plt.xlim([0, 1.1])
        plt.ylim([0, 1.1])
        plt.xlabel('1 - $\\alpha$')
        plt.ylabel('coverage rate')
        # plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()
        plt.close('all')


class CoverageRateByDimension(Metric):
    """
    Coverage rate for every dimension in the label space.
    """

    @staticmethod
    def name() -> str:
        return 'Coverage Rate By Dimension'

    @staticmethod
    def snake_name() -> str:
        return 'coverage_rate_by_dimension'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, coverage_d = CoverageRateByDimension.raw_coverage(y, y_hat, **kwargs)
        # computing the average coverage for every dimension on the label space
        cr_d = np.sum(coverage_d, axis=1) / coverage_d.shape[1]  # [|alpha|, hl, hf]
        # list of alpha values and, corresponding dimension-wise coverage rate
        return alpha, cr_d

    @staticmethod
    def raw_coverage(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        """
        Evaluates the coverage for every dimension and every example
        :param y: Ground truth, [batch size, hl, |hf|]
        :param y_hat: Prediction intervals, {
                        'model': Forecast of the underlying model with shape [batch size, hl, |hf|, 0 | [lower, upper]],
                        alpha[0]: Prediction intervals with shape [batch size, hl, |hf|, [lower, upper]],
                        ...
                        alpha[k]: Prediction intervals with shape [batch size, hl, |hf|, [lower, upper]]
                    }
        :param kwargs: Possible metric specific parameters
        :return: alpha[0:k], coverage as boolean np array of shape [|alpha|, batch size, hl, |hf|]
        """
        y_hat_stack, y_hat_keys = Metric.stack_y_hat(y_hat)
        alpha = np.array(y_hat_keys)
        lower, upper = y_hat_stack[:, :, :, :, 0], y_hat_stack[:, :, :, :, 1]  # [|alpha|, n_samples, hl, hf]

        coverage_d = torch.logical_and(y >= lower, y <= upper)  # [|alpha|, n_samples, hl, hf]
        coverage_d = np.array(coverage_d.detach())

        return alpha, coverage_d

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # sub-figs, one for every dimenstion
        alpha, cr_d = result
        epsilon = 1 - alpha

        hl = cr_d.shape[1]
        hf = cr_d.shape[2]
        fig, axs = plt.subplots(hf, hl, figsize=(4 * hl, 4 * hf), squeeze=False)
        for f in range(hf):
            for t in range(hl):
                axs[f, t].plot(epsilon, cr_d[:, t, f], label=f't {t}, f {f}')
                axs[f, t].plot([i for i in range(2)], [i for i in range(2)], label='target')
                axs[f, t].set_xlabel('1 - α')
                axs[f, t].set_ylabel('coverage rate')
                axs[f, t].legend()
                axs[f, t].set_xlim([0, 1.1])
                axs[f, t].set_ylim([0, 1.1])
        fig.suptitle(title)
        if save_path is not None:
            fig.savefig(save_path)
        if display:
            fig.show()
        plt.clf()
        plt.close('all')


class CoverageRateByFeature(Metric):

    @staticmethod
    def name() -> str:
        return 'Coverage Rate By Feature'

    @staticmethod
    def snake_name() -> str:
        return 'coverage_rate_by_feature'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, coverage_d = CoverageRateByDimension.raw_coverage(y, y_hat, **kwargs)
        coverage_f = np.all(coverage_d, axis=2)  # [|alpha|, batch size, hl, |hf|] -> [|alpha|, batch size, |hf|]
        cr_f = np.mean(coverage_f, axis=1)  # [|alpha|, batch size, |hf|] -> [|alpha|, |hf|]
        # list of alpha values and, corresponding time-step-wise coverage rate for every alpha
        return alpha, cr_f

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # sub-figs, one for every feature
        alpha, cr_d = result
        epsilon = 1 - alpha

        hf = cr_d.shape[1]
        fig, axs = plt.subplots(1, hf, figsize=(4 * hf, 4), squeeze=False)
        for f in range(hf):
            axs[0, f].plot(epsilon, cr_d[:, f], label=f'f {f}')
            axs[0, f].plot([i for i in range(2)], [i for i in range(2)], label='target')
            axs[0, f].set_xlabel('1 - α')
            axs[0, f].set_ylabel('coverage rate')
            axs[0, f].legend()
            axs[0, f].set_xlim([0, 1.1])
            axs[0, f].set_ylim([0, 1.1])
        fig.suptitle(title)
        if save_path is not None:
            fig.savefig(save_path)
        if display:
            fig.show()
        plt.clf()
        plt.close('all')


class CoverageRateByTimeStep(Metric):

    @staticmethod
    def name() -> str:
        return 'Coverage Rate By Time Step'

    @staticmethod
    def snake_name() -> str:
        return 'coverage_rate_by_time_step'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, coverage_d = CoverageRateByDimension.raw_coverage(y, y_hat, **kwargs)
        coverage_ts = np.all(coverage_d, axis=3)  # [|alpha|, batch size, hl, |hf|] -> [|alpha|, batch size, hl]
        cr_ts = np.mean(coverage_ts, axis=1)  # [|alpha|, batch size, hl] -> [|alpha|, hl]
        # list of alpha values and, corresponding feature-wise coverage rate for every alpha
        return alpha, cr_ts

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # sub-figs, one for every time step
        alpha, cr_d = result
        epsilon = 1 - alpha

        hl = cr_d.shape[1]
        fig, axs = plt.subplots(1, hl, figsize=(4 * hl, 4), squeeze=False)
        for t in range(hl):
            axs[0, t].plot(epsilon, cr_d[:, t], label=f't {t}')
            axs[0, t].plot([i for i in range(2)], [i for i in range(2)], label='target')
            axs[0, t].set_xlabel('1 - α')
            axs[0, t].set_ylabel('coverage rate')
            axs[0, t].legend()
            axs[0, t].set_xlim([0, 1.1])
            axs[0, t].set_ylim([0, 1.1])
        fig.suptitle(title)
        if save_path is not None:
            fig.savefig(save_path)
        if display:
            fig.show()
        plt.clf()
        plt.close('all')


class RollingCoverageRate(RollingMetric):
    """
    Rolling average coverage rate.
    """

    @staticmethod
    def name() -> str:
        return 'Rolling Coverage Rate'

    @staticmethod
    def snake_name() -> str:
        return 'rolling_coverage_rate'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        # wc_d: [|alpha|, n_windows, hl, |hf|, r_window_length]
        alpha, wc_d, w_end_idx = RollingCoverageRateByDimension.raw_windowed_coverage(y, y_hat, **kwargs)
        # [|alpha|, n_windows, hl, |hf|, r_window_length] -> [|alpha|, n_windows, hl, r_window_length]
        wc_ts = np.all(wc_d, axis=3)
        wc = np.all(wc_ts, axis=2)  # [|alpha|, n_windows, hl, r_window_length] -> [|alpha|, n_windows, r_window_length]
        rcr = np.mean(wc, axis=-1)  # [|alpha|, n_windows] -> [|alpha|, n_windows]
        # list of alpha values and, corresponding rolling coverage rate for every alpha, window_end_idx
        return alpha, rcr, w_end_idx

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        alpha, rcr, window_end_idx = result

        fig = plt.figure(figsize=(8, 6))
        for i, a in enumerate(alpha):
            rcr_traces = plt.plot(window_end_idx, rcr[i], label=f'α: {a}')
            plt.plot(window_end_idx, [1 - a for i in range(len(window_end_idx))],
                     color=rcr_traces[0].get_color(), linestyle='dotted', alpha=0.7)
        plt.title(title)
        plt.xlabel('window')
        plt.ylabel('coverage rate')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()
        plt.close('all')

    @staticmethod
    def comparative_plot(results: list, labels: list[str], a: float = 0.9, title: str = None, save_path: str = None,
                         display: bool = True, fig_size=(4, 3), alpha_prime: float | None = None):
        """
        Plots multiple result objects onto the same graph.
        :param results: list of tuples (alpha, cr)
        :param labels: labels associated with the different results
        :param a: alpha value that is compared, only one can be chosen.
        :param title: title for the graph
        :param save_path: path where the graph will be saved
        :param display: show the result ot not
        :param fig_size: plt figsize
        :param alpha_prime: alternative value for alpha in the plot
        :return:
        """
        fig = plt.figure(figsize=fig_size)
        for i in range(len(results)):
            alpha, rcr, window_end_idx = results[i]
            epsilon = 1 - alpha
            for j, al in enumerate(alpha):
                if al == a:
                    plt.plot(window_end_idx, rcr[j], label=labels[i])
        plt.plot(window_end_idx, [1 - a for _ in range(len(window_end_idx))], label=f'$\\alpha$ = {a}',
                 color='gray', linestyle='dotted', alpha=0.7)
        if alpha_prime is not None:
            plt.plot(window_end_idx, [1 - alpha_prime for _ in range(len(window_end_idx))],
                     label=f'$\\alpha\'$ = {alpha_prime:.4f}',
                     color='gray', linestyle='dashed', alpha=0.7)
        if title is not None:
            plt.title(title)
        plt.legend(fontsize='x-small')
        plt.ylim([0, 1.1])
        plt.xlabel('example')  # 'window'
        plt.ylabel('coverage rate')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()
        plt.close('all')


class RollingCoverageRateByDimension(RollingMetric):
    """
    Coverage rate for every dimension in the label space.
    """

    @staticmethod
    def name() -> str:
        return 'Rolling Coverage Rate By Dimension'

    @staticmethod
    def snake_name() -> str:
        return 'rolling_coverage_rate_by_dimension'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        # wc_d: [|alpha|, n_windows, hl, |hf|, r_window_length]
        alpha, wc_d, w_end_idx = RollingCoverageRateByDimension.raw_windowed_coverage(y, y_hat, **kwargs)
        # [|alpha|, n_windows, hl, |hf|, r_window_length] -> [|alpha|, n_windows, hl, |hf|]
        rcr_d = np.mean(wc_d, axis=-1)
        # list of alpha values and, corresponding dimension-wise rolling coverage rate for every alpha, w_end_idx
        return alpha, rcr_d, w_end_idx

    @staticmethod
    def raw_windowed_coverage(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        """
        Evaluates the coverage for every dimension and every example and slices the results according to a moving window
        scheme.
        :param y: Ground truth, [batch size, hl, |hf|]
        :param y_hat: Prediction intervals, {
                        'model': Forecast of the underlying model with shape [batch size, hl, |hf|, 0 | [lower, upper]],
                        alpha[0]: Prediction intervals with shape [batch size, hl, |hf|, [lower, upper]],
                        ...
                        alpha[k]: Prediction intervals with shape [batch size, hl, |hf|, [lower, upper]]
                    }
        :param kwargs: Possible metric specific parameters
        :return: alpha[0:k],
                 coverage as boolean np array of shape [|alpha|, n_windows, hl, |hf|, r_window_length],
                 end indexes of the corresponding rolling window
        """
        rolling_window_length = 300
        if 'rolling_window_length' in kwargs:
            rolling_window_length = kwargs.get('rolling_window_length')
        assert rolling_window_length < len(y), 'rolling window must be smaller than total number of examples'

        alpha, coverage_d = CoverageRateByDimension.raw_coverage(y, y_hat, **kwargs)
        windowed_coverage_d = sliding_window_view(coverage_d, window_shape=rolling_window_length,
                                                  axis=1)  # [|alpha|, n_windows, hl, |hf|, r_window_length]
        window_end_idx = np.array([i for i in range(rolling_window_length - 1, len(y))])

        return alpha, windowed_coverage_d, window_end_idx

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # sub-figs, one for every dimenstion
        alpha, rcr_d, window_end_idx = result

        hl = rcr_d.shape[2]
        hf = rcr_d.shape[3]
        fig, axs = plt.subplots(hf, hl, figsize=(8 * hl, 6 * hf), squeeze=False)
        for f in range(hf):
            for t in range(hl):
                for i, a in enumerate(alpha):
                    axs[f, t].plot(window_end_idx, rcr_d[i, :, t, f], label=f'α: {a}')
                axs[f, t].set_title(f't {t}, f {f}')
                axs[f, t].set_xlabel('window')
                axs[f, t].set_ylabel('coverage rate')
                axs[f, t].legend()
                axs[f, t].set_ylim([0, 1.1])
        fig.suptitle(title)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        if display:
            fig.show()
        plt.clf()
        plt.close('all')


class RollingCoverageRateByFeature(RollingMetric):
    """
    Coverage rate for every dimension in the label space.
    """

    @staticmethod
    def name() -> str:
        return 'Rolling Coverage Rate By Feature'

    @staticmethod
    def snake_name() -> str:
        return 'rolling_coverage_rate_by_feature'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        # wc_d: [|alpha|, n_windows, hl, |hf|, r_window_length]
        alpha, wc_d, w_end_idx = RollingCoverageRateByDimension.raw_windowed_coverage(y, y_hat, **kwargs)
        # [|alpha|, n_windows, hl, |hf|, r_window_length] -> [|alpha|, n_windows, |hf|, r_window_length]
        wc_f = np.all(wc_d, axis=2)
        rcr_f = np.mean(wc_f, axis=-1)  # [|alpha|, n_windows, |hf|, r_window_length] -> [|alpha|, n_windows, hf]
        # list of alpha values and, corresponding rolling coverage rate for every feature and every alpha,
        # window_end_idx
        return alpha, rcr_f, w_end_idx

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # sub-figs, one for every dimenstion
        alpha, rcr_d, window_end_idx = result

        hf = rcr_d.shape[2]
        fig, axs = plt.subplots(1, hf, figsize=(8 * hf, 6), squeeze=False)
        for f in range(hf):
            for i, a in enumerate(alpha):
                axs[0, f].plot(window_end_idx, rcr_d[i, :, f], label=f'α: {a}')
            axs[0, f].set_title(f'f {f}')
            axs[0, f].set_xlabel('window')
            axs[0, f].set_ylabel('coverage rate')
            axs[0, f].legend()
            axs[0, f].set_ylim([0, 1.1])
        fig.suptitle(title)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        if display:
            fig.show()
        plt.clf()
        plt.close('all')


class RollingCoverageRateByTimeStep(RollingMetric):
    """
    Coverage rate for every dimension in the label space.
    """

    @staticmethod
    def name() -> str:
        return 'Rolling Coverage Rate By Time Step'

    @staticmethod
    def snake_name() -> str:
        return 'rolling_coverage_rate_by_time_step'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        # wc_d: [|alpha|, n_windows, hl, |hf|, r_window_length]
        alpha, wc_d, w_end_idx = RollingCoverageRateByDimension.raw_windowed_coverage(y, y_hat, **kwargs)
        # [|alpha|, n_windows, hl, |hf|, r_window_length] -> [|alpha|, n_windows, hl, r_window_length]
        wc_ts = np.all(wc_d, axis=3)
        rcr_ts = np.mean(wc_ts, axis=-1)  # [|alpha|, n_windows, hl, r_window_length] -> [|alpha|, n_windows, hl]
        # list of alpha values and, corresponding rolling coverage rate for every feature and every alpha,
        # window_end_idx
        return alpha, rcr_ts, w_end_idx

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # sub-figs, one for every dimenstion
        alpha, rcr_d, window_end_idx = result

        hl = rcr_d.shape[2]
        fig, axs = plt.subplots(1, hl, figsize=(8 * hl, 6), squeeze=False)
        for t in range(hl):
            for i, a in enumerate(alpha):
                axs[0, t].plot(window_end_idx, rcr_d[i, :, t], label=f'α: {a}')
            axs[0, t].set_title(f't {t}')
            axs[0, t].set_xlabel('window')
            axs[0, t].set_ylabel('coverage rate')
            axs[0, t].legend()
            axs[0, t].set_ylim([0, 1.1])
        fig.suptitle(title)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        if display:
            fig.show()
        plt.clf()
        plt.close('all')


class CoveredDimensions(Metric):
    """
    Computes the number of covered dimensions for every alpha and every example in the provided batch.
    """

    @staticmethod
    def name() -> str:
        return 'Covered Dimensions'

    @staticmethod
    def snake_name() -> str:
        return 'covered_dimensions'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, coverage_d = CoverageRateByDimension.raw_coverage(y, y_hat, **kwargs)
        n_covered_ts = np.sum(coverage_d, axis=3)  # [|alpha|, batch size, hl, |hf|] -> [|alpha|, batch size, hl]
        n_covered_d = np.sum(n_covered_ts, axis=2)  # [|alpha|, batch size, hl] -> [|alpha|, batch size]
        return alpha, n_covered_d

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # Box plot for evey alpha.
        alpha, n_covered_d = result
        plt.boxplot(n_covered_d.T, labels=alpha)
        plt.xlabel('$\\alpha$')
        plt.ylabel('covered dimensions')
        plt.title(title)
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
        # todo: implement similar to the RollingCoverageRate one where one alpha needs to be selected. Thec plot all
        #  box plots for that value of alpha.
        raise 'Not yet implemented.'
