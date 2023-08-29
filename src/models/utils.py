# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license
import enum
import logging
from typing import Type

import torch

from models.lin_reg import LinearRegression
from models.rnn import PointRNN


class KnownModels(str, enum.Enum):
    """
    List of available underlying models.
    """
    lin_reg = 'lin_reg'
    point_rnn = 'point_rnn'

    @classmethod
    def get_model_class(cls, model_type: 'KnownModels') -> Type[torch.nn.Module]:
        if model_type == cls.lin_reg:
            return LinearRegression
        if model_type == cls.point_rnn:
            return PointRNN
        logging.error(f'Model type {model_type} not found.')


def select_device():
    """
    Selects the device the model will be trained on.
    :return: device name to be used by PyTorch
    """
    device = 'cpu'
    logging.info(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.debug(f"CUDA version: {torch.version.cuda}")
        cuda_id = torch.cuda.current_device()
        logging.info(f"ID of current CUDA device: {cuda_id}")
        logging.info(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
        device = f'cuda:{cuda_id}'
    return device
