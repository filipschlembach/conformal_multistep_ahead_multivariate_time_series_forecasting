import enum
from abc import ABC, abstractmethod
import logging
import numpy as np
import torch
from typing import Type, Callable

from data_ingest.datasets import Dataset


class ConformalPredictor(ABC):

    def __int__(self, known_examples: Dataset, model_class: Type[torch.nn.Module] = None,
                model_params: dict = None, alpha: list[float] = None, **kwargs):
        """
        Instantiates the conformal predictor.
        :param known_examples: training split of the data set, not identical to proper training set in general
        :param model_class: underlying model class
        :param model_params: parameters used to fit the model
        :param alpha: significance levels
        :return:
        """
        if alpha is None:
            alpha = [0.05]
        self.known_examples: Dataset = known_examples
        self.model_class: Type[torch.nn] = model_class
        self.model_params: dict = model_params
        self.alpha: list[float] = alpha

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str | float, torch.Tensor]:
        """
        Generates prediction intervals for the given samples.
        :param x: Tensor of shape [batch size, window length, #window features]
        :return: {
                    'model': Forecast of the underlying model with shape [batch size, hl, |hf|, 0 | [lower, upper]],
                    alpha[0]: Prediction intervals with shape [batch size, hl, |hf|, [lower, upper]],
                    ...
                    alpha[k]: Prediction intervals with shape [batch size, hl, |hf|, [lower, upper]]
                }
        """
        pass

    @abstractmethod
    def nonconformity(self, output, calibration_example):
        """
        Measures the nonconformity between output and target time series.
        :param output: the prediction given by the auxiliary forecasting model
        :param calibration_example: the tuple consisting of calibration example's input sequence and ground truth
                                    forecast
        :return: nonconformiry score
        """
        pass

    @staticmethod
    def weighted_quantile(values: np.ndarray, epsilons: float | np.ndarray,
                          weights: np.ndarray = None) -> np.ndarray:
        """
        Very close to numpy.quantile, but supports weights. Behaves like interpolation='higher' in numpy.
        source: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
        :param values: numpy array with data, generally nonconformity measures (ex. ncf) in this context.
        :param epsilons: numpy array or int with as many quantiles as needed (multiple values for epsilon = 1 - alpha)
        :param weights: numpy array of the same length as `values` or len(values) + 1
        :return: numpy array with computed quantiles
        """

        assert isinstance(values, np.ndarray), '"values" should be a numpy ndarray'
        assert values.ndim == 1, '"values" needs to be a 1D array.'

        if not isinstance(epsilons, np.ndarray):
            epsilons = np.array(epsilons)
        assert epsilons.ndim == 1, '"epsilons" needs to be a 1D array.'
        epsilons = np.sort(epsilons)  # increases computational efficiency a little later on

        # following section inspired by Barber et al. - 2022 - Conformal prediction beyond exchangeability
        n = len(values)

        if weights is None:
            weights = np.ones(n + 1)
        elif len(weights) == n:
            weights = np.r_[weights, 1]
        else:
            assert isinstance(weights, np.ndarray), '"weights" should be a numpy ndarray'
            assert weights.ndim == 1, '"weights" needs to be a 1D array.'
            assert len(weights) == n + 1, '"weights" needs to be of length n or n + 1'
        weights = weights / np.sum(weights)

        assert np.all(epsilons >= 0) and np.all(epsilons <= 1), '"epsilons" values should be in [0, 1]'

        # sort values and weights in ascending order of values
        sorter = np.argsort(values)
        sorted_values = values[sorter]
        sorted_weights = weights[:-1][sorter]  # there are n + 1 weights but only n values, last weight is discarded

        cdf = np.cumsum(sorted_weights)  # cumulative distribution function

        logging.debug(f'sorted_weights: {sorted_weights}')
        logging.debug(f'sorted_values: {sorted_values}')
        logging.debug(f'epsilons: {epsilons}')

        # compute the value for every epsilon
        # because the epsilon are sorted, they can be compared to the cdf in increasing order
        quantile_values = np.zeros(len(epsilons))
        epsilon_i = 0
        cdf_i = 0
        while cdf_i < len(cdf):
            # logging.debug(f'cdf_i: {cdf_i}')
            if epsilons[epsilon_i] <= cdf[cdf_i]:
                quantile_values[epsilon_i] = sorted_values[cdf_i]
                if epsilon_i < len(epsilons) - 1:
                    epsilon_i += 1
                    # if the cdf has large jumps multiple epsilons might lie between two consecutive values of the cdf
                    cdf_i = cdf_i - 1
            cdf_i = cdf_i + 1

        for e_j in range(epsilon_i, len(epsilons)):
            quantile_values[e_j] = np.inf

        return quantile_values

    class WeightFunctions:
        """
        Weight generating functions for the weighted quantile function
        """

        class Name(str, enum.Enum):
            constant = 'constant'
            exponential = 'exponential'
            hard_cutoff = 'hard_cutoff'
            inverse = 'inverse'
            linear = 'linear'
            soft_cutoff = 'soft_cutoff'

        @classmethod
        def get(cls, name: Name, fct_params: dict = None) -> Callable[[int], list[float]]:
            """
            Returns the weight generating function given a name.
            :param name: name of the weight generating function
            :param fct_params: parameters of the weight generating function
            :return:
            """
            if fct_params is None:
                fct_params = {}
            if name == cls.Name.constant:
                return cls.get_constant(**fct_params)
            if name == cls.Name.exponential:
                return cls.get_exponential(**fct_params)
            if name == cls.Name.hard_cutoff:
                return cls.get_hard_cutoff(**fct_params)
            if name == cls.Name.inverse:
                return cls.get_inverse(**fct_params)
            if name == cls.Name.linear:
                return cls.get_linear(**fct_params)
            if name == cls.Name.soft_cutoff:
                return cls.get_soft_cutoff(**fct_params)

        @staticmethod
        def get_constant(const: float = 1.):
            """
            Constant weights.
            :param const: value of the weights
            :return: weight function
            """
            return lambda n: [const for _ in range(n)]

        @staticmethod
        def get_exponential(factor: float = 1, beta: float = 0.05):
            """
            Exponentially increasing weights.
            :param factor: factor scaling the weights
            :param beta: changes how aggressively older weights are discounted.
            :return: weight function
            """
            # return lambda n: [1000. * np.exp((i - n) / (n / 15)) for i in range(n)]  # used in the paper
            # return lambda n: [factor * np.exp((i - n) / n) for i in range(n)]
            logging.debug(f'factor: {factor}, beta: {beta}')
            return lambda n: [factor * np.exp(beta * (i - n)) for i in range(n)]

        @staticmethod
        def get_hard_cutoff(factor: float = 1., cutoff: int = 50):
            """
            Soft cutoff assigns high weights to the past 'cutoff' time steps and zero weights before that
            :param factor: scaling of the function
            :param cutoff: after which point very low weights will be assigned
            :return: weight function
            """
            if cutoff < 0:
                raise ValueError('cutoff needs to be > 0')
            return lambda n: [0. for _ in range(n - cutoff)] + [factor for _ in range(cutoff)]

        @staticmethod
        def get_inverse(factor: float = 1.):
            """
            Decreasing weights at the rate of 1/t
            :param factor: factor scaling the weights
            :return: weight function
            """
            return lambda n: [factor / (n - i) for i in range(n)]

        @staticmethod
        def get_linear(val: float = 1.):
            """
            Linearly increasing weights.
            :param val: factor scaling the weights
            :return: weight function
            """
            # return [1000. / (n - i) for i in range(n)]  # used in the paper, definitely not linear
            return lambda n: [val * (i + 1) / n for i in range(n)]

        @staticmethod
        def get_soft_cutoff(factor: float = 0.5, cutoff: int = 50, softness: float = 10.):
            """
            Soft cutoff assigns high weights to the past 'cutoff' time steps and much lower weights before that
            :param factor: scaling of the function
            :param cutoff: after which point very low weights will be assigned
            :param softness: softer transition for larger values
            :return: weight function
            """
            logging.info(f'factor: {factor}, cutoff: {cutoff}, softness: {softness}')
            if cutoff < 0:
                raise ValueError('cutoff needs to be > 0')
            if softness < 0:
                raise ValueError('softness needs to be > 0')
            return lambda n: [factor * ((i - n + cutoff) / (softness + np.abs(i - n + cutoff)) + 1) for i in range(n)]
