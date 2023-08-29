import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics.metric import Metric


class InfinityCount(Metric):
    @staticmethod
    def name() -> str:
        return 'Infinity Count'

    @staticmethod
    def snake_name() -> str:
        return 'infinity_count'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        alpha, inf_count_d = InfinityCountByDimension.eval(y, y_hat, **kwargs)  # [|alpha|, hl, hf]
        inf_count_ts = np.sum(inf_count_d, axis=2)  # [|alpha|, hl]
        inf_count = np.sum(inf_count_ts, axis=1)  # [|alpha|]
        return alpha, inf_count

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        alpha, inf_count = result

        # plt.bar([1 - a for a in alpha], [inf_count[i] for i in range(len(inf_count))])  # label=f'interval width'
        plt.plot([1 - a for a in alpha], [inf_count[i] for i in range(len(inf_count))])  # label=f'interval width'
        plt.title(title)
        # plt.legend()
        plt.xlim([0, 1.1])
        plt.xlabel('1 - α')
        plt.ylabel('number of infinite intervals')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if display:
            plt.show()
        plt.clf()
        plt.close('all')


class InfinityCountByDimension(Metric):

    @staticmethod
    def name() -> str:
        return 'Infinity Count By Dimension'

    @staticmethod
    def snake_name() -> str:
        return 'infinity_count_by_dimension'

    @staticmethod
    def eval(y: torch.Tensor, y_hat: dict[str | float, torch.Tensor], **kwargs):
        y_hat_stack, y_hat_keys = Metric.stack_y_hat(y_hat)  # [|alpha|, batch_size, hl, |hf|]
        alpha = np.array(y_hat_keys)  # [|alpha|]
        lower, upper = y_hat_stack[:, :, :, :, 0], y_hat_stack[:, :, :, :, 1]  # [|alpha|, batch_size, hl, |hf|]

        inf_d = upper == torch.inf  # [|alpha|, batch_size, hl, |hf|]
        inf_count_d = torch.sum(inf_d, dim=1)  # [|alpha|, n_samples, hl, |hf|]
        logging.debug(f'inf_count_d.shape = {inf_count_d.shape}')

        return alpha, np.array(inf_count_d.cpu().detach())

    @staticmethod
    def simple_plot(result, title: str, save_path: str = None, display: bool = True):
        # come up with an intuitive way to visualize this.
        pass
