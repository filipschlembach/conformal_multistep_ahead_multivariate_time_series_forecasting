import enum
import logging
import numpy as np
import time
import torch
import torch.nn.functional
import torch.utils.data
from typing import Type, Callable

from conformal_predictors.cp import ConformalPredictor
from data_ingest.datasets import Dataset


class ICP(ConformalPredictor):
    """
    This class implements an Inductive Conformal Predictor (ICP) using the residuals as nonconformity core.
    After initializing an ICP instance the modus operandi depends on the setting the ICP is used in.
    The most common settings will be laid out here.
    It is important to note, that the setting is determined by the user through the use of the ICP. The ICP instance
    is ignorant to the setting it is used in.


    Off-Line Setting
    ----------------

    In the off-line setting the underlying model is fitted once to the proper training set and the ICP is calibrated on
    the calibration set. All subsequent prediction intervals are constructed using the nonconformity measures from this
    calibration operation. No information that may become available in the future is taken into account.
    1. icp = ICP(...)    # instantiate the ICP with all necessary arguments
    2. icp.fit(...)      # fit the underlying model to the proper training set
    3. icp.cal(...)      # calibrate the ICP on the calibration set
    4. icp.forward(...)  # generate prediction intervals for as many future examples as necessary


    Semi-Off-Line Setting
    ---------------------

    In the semi-off-line setting the underlying model is fitted once to the proper training set and the ICP is
    calibrated on the calibration set.
    After a prediction interval has been constructed, the true label needs to be made available to the ICP so the
    example can be added to the calibration set (i.e. it's nonconformity score to the set of known nonconformity
    scores).
    This way, every prediction interval is constructed using the nonconformity measures from the initial calibration
    set as well as all new nonconformity scores that have been added since.
    1. icp = ICP(...)                      # instantiate the ICP with all necessary arguments
    2. icp.fit(...)                        # fit the underlying model to the proper training set
    3. icp.cal(...)                        # calibrate the ICP on the calibration set
    4. while there are new objects x_i
    4.1 icp.forward(x_i)                   # generate a prediction interval for a new object x_i
    4.2 icp.add_cal_example(z_i, y_hat_i)  # add the object i to the calibration set once the true label is available.
                                           #  This operation automatically re-calibrates the ICP instance.


    On-Line Setting
    ---------------

    In the on-line setting the underlying model is fitted and the ICP is calibrated for every new example the ICP is
    presented with.
    After a prediction interval has been constructed, the true label needs to be made available to the ICP so the
    example can be added to the calibration set (i.e. it's nonconformity score to the set of known nonconformity
    scores).
    This way, every prediction interval is constructed using the nonconformity measures from the  current calibration
    set that may include nonconformity scores from examples that were not available initially.
    Assuming a computationally expensive model fitting operating, this setting the most expensive one.
    1. icp = ICP(...)                  # instantiate the ICP with all necessary arguments
    2. while there are new objects x_i
    2.1 icp.fit(...)                   # fit the underlying model to the proper training set
    2.2 icp.cal(...)                   # calibrate the ICP on the calibration set
    2.3 icp.forward(x_i)               # generate a prediction interval for a new object x_i
    2.4 icp.add_example(z_i, y_hat_i)  # add the object i to the set of known examples once the true label is available.

    General Remarks
    ---------------

    - the ICP assumes that calibrate() was called at least once when executing forward(). It is the user's
    responsibility to make sure that this is actually the case.

    - Make sure, that when using the weighted quantile function 'random_split' is set to false so that the split between
    the proper training set and the calibration set happens sequentially on the time axis.
    Exceptions to this rule may apply. Make sure you understand the theory well before disregarding this remark.

    - When comparing ICPs with different settings for 'weight_f' or 'correction', the computational efficiency can be
    improved by training the underlying model only once and sharing it between the different ICP instances.
    If the inference step is also computationally expensive for the chosen underlying model, the predictions it makes
    may be added to other ICP instances using the icp.add_cal_example(z_i, y_hat_i) or icp.add_example(z_i, y_hat_i)
    methods.

    - For the situation where the underlying model is inaccessible, but its predictions are accessible the model fitting
    and calibration phases in the previous examples can be replaced with calls to the add_example(...) method.

    - Why do I have to provide the model's prediction when calling add_example(z_i, y_hat_i)?
    Because the delay between the object and it's true label becoming accessible may be greater than one time step.
    Instead of making an assumption here and, for instance, caching the last prediction and using that one, the
    responsibility of managing new true labels and adding them to the set of known examples is transferred to the user.
    """

    class Corrections(str, enum.Enum):
        """
        List of available family-wise error corrections for the quantile function.
        """
        no = 'no'  # applies no correction: alpha' = alpha
        bonferroni = 'bonferroni'  # applies bonferroni correction: alpha' = alpha / h
        independent = 'independent'  # assumes independence of errors: alpha' = 1 - (1 - alpha)^(1 / h)

    def __init__(self, known_examples: Dataset,
                 model_class: Type[torch.nn.Module] = None,
                 model_params: dict = None,
                 alpha: list[float] = None,
                 proper_training_set_size: float | int = 0.5,
                 random_split: bool = True,
                 correction: str = 'bonferroni',
                 device: str = None,
                 weight_f: Callable[[int], list[float]] | None = None,
                 **kwargs):
        """
        Implements an on-line inductive conformal predictor.
        :param known_examples: training split of the data set
        :param model_class: underlying model class
        :param model_params: parameters used to fit the model
        :param alpha: significance levels
        :param proper_training_set_size: size of the proper training set.
                                         If an integer is given it determines the number of examples in the proper
                                         training set. If a float is given, it determines the ratio of known examples
                                         that are used in the proper training set.
        :param random_split: if false, the split between the proper training set and the calibration set does not
                             shuffle the examples
        :param correction: for multistep ahead and / or multivariate predictions, how the independent predictions are
                           combined. Possible values are
                           - 'no': No correction is applied, individual features and time steps are processed
                                   independently using the targeted error rate
                           - 'bonferroni': Bonferroni correction is applied
                           - 'independent': Correction as if the errors per feature and time step were independent
        :param device: cuda device. If none cpu is used.
        :param weight_f: function that generates n weights for the weighted quantile function given the number n
        :param kwargs:
        :return:
        """
        super(ICP, self).__int__(known_examples=known_examples, model_class=model_class,
                                 model_params=model_params, alpha=alpha)
        if kwargs:
            for k, v in kwargs.items():
                logging.info(f'Parameter {k} = {v} will be ignored.')

        # attributes passed by the constructor
        self.proper_training_set_size: float | int = proper_training_set_size
        self.random_split: bool = random_split
        self.correction: str = correction
        self.alpha = np.array(alpha)
        self.q: dict = {a: None for a in alpha}  # q-values associated with a given alpha. [horizon, output_size]
        self.weight_f: Callable[[int], list[float]] | None = weight_f

        # computed attributes
        self.proper_train_set: Dataset | None = None
        self.cal_set: Dataset | None = None
        self.cal_y_hat: list[torch.Tensor] | None = None  # y_hat for elements in the cal_set
        self.ncs: list[torch.Tensor] = []  # nonconformity scores for elements in the cal_set

        self.model: torch.nn = None  # instance of the provided model class
        self.device = device

        if self.random_split and self.weight_f is not None:
            logging.warning('For weighted quantiles the split between the proper training set and the calibration set'
                            'should be sequential. Consider setting random_split = False.')

    def forward(self, x: torch.Tensor) -> dict[str | float, torch.Tensor]:
        y_hat, _ = self.model(x)
        pred = {'model': y_hat}  # predict the values using the underlying model
        for a in self.alpha:
            lower = y_hat - self.q[a]
            upper = y_hat + self.q[a]
            pred[a] = torch.stack([lower, upper], -1)  # generate the prediction intervals
        return pred

    def nonconformity(self, prediction, calibration_example):
        """
        Measures the nonconformity between output and target time series.
        :param prediction: the prediction given by the auxiliary forecasting model
        :param calibration_example: the tuple consisting of calibration example's input sequence and ground truth
                                    forecast
        :return: Average MAE loss for every step in the sequence.
        """
        label = calibration_example[1]
        return torch.nn.functional.l1_loss(prediction, label, reduction="none")

    def fit(self, split_know_examples: bool = True) -> None:
        """
        Fits the underlying model to the proper training set and generates predictions for the calibration set.
        :param split_know_examples: if True, the known examples are split again.
                                    If false, the last split and thereby the last proper training set is used.
        """

        # split the known examples
        if split_know_examples:
            self.proper_train_set, self.cal_set = self._split_dataset()

        # fit the model
        model = self.model_class(**self.model_params)
        start = time.time()
        model.fit(train_dataset=self.proper_train_set, batch_size=self.model_params['batch_size'],
                  epochs=self.model_params['epochs'], lr=self.model_params['lr'])
        logging.info(f'Model fitting duration: {time.time() - start} s')
        self.model = model

        # generate predictions for the calibration set
        self.cal_y_hat = self._predict_cal_y_hat()

    def cal(self) -> None:
        """
        Calibrates the split conformal predictor using all available elements in the calibration set and their
        associated predictions made by the underlying model.
        :return: None
        """
        assert self.cal_set is not None, 'The ICP cannot be calibrated because the calibration set is empty. ' \
                                         'Call the fit(...) method or add elements manually to the calibration set ' \
                                         'using the add_example(...) method.'
        assert self.cal_y_hat is not None, 'The ICP cannot be calibrated because there are no predictions available ' \
                                           'for the calibration set. ' \
                                           'Call the fit(...) method or add examples and their associated predictions' \
                                           ' manually to the calibration set using the add_example(...) method. ' \
                                           'In very rare instances you may want to call predict_cal_y_hat().'
        assert len(self.cal_set) == len(self.cal_y_hat), 'cal_set and cal_y_hat need to have the same length.'
        # generate the nonconformity scores for the predictions made for the calibration set
        cal_loader = torch.utils.data.DataLoader(self.cal_set, batch_size=1, shuffle=False)
        for i, cal_example in enumerate(cal_loader):
            y_hat = self.cal_y_hat[i]
            score = self.nonconformity(y_hat, cal_example)  # [batch_size, horizon, output_size]
            self.ncs.append(score)
        # use these nonconformity scores to compute the quantiles
        self._comp_quantiles()

    def add_cal_example(self, z, y_hat) -> None:
        """
        Computes the residuals for a given batch of examples
        :param z: calibration example (x, y): ([batch_size, wt, wf], [batch_size,ht, hf])
        :param y_hat: associated prediction: [batch_size, ht, hf]
        :return:
        """
        assert len(z[0].shape) == 3, 'Examples must have the format ([batch_size, wt, wf], [batch_size,ht, hf])'
        assert len(z[1].shape) == 3, 'Examples must have the format ([batch_size, wt, wf], [batch_size,ht, hf])'
        assert len(y_hat.shape) == 3, 'y_hat must have the format ([batch_size, wt, wf], [batch_size,ht, hf])'
        assert z[0].shape[0] == y_hat.shape[0], 'The same number of examples need to be provided for z and y_hat.'
        assert z[1].shape[0] == y_hat.shape[0], 'The same number of examples need to be provided for z and y_hat.'
        for i in range(z[0].shape[0]):
            self.known_examples.append(z[0][i], z[1][i])
            self.cal_y_hat.append(y_hat[[i], :, :])

        # compute residual with true label and add to existing ones
        score = self.nonconformity(y_hat, z)  # [batch_size, horizon, output_size]
        self.ncs.append(score)
        logging.debug(f'Added NCS shape: {score.shape}')

        # re-compute the quantile widths based on the new nonconformity score
        self._comp_quantiles()

    def add_example(self, z) -> None:
        """
        Adds a new known example to the data set that can be used in the on-line setting. No calibration is done.
        :param z:
        :return:
        """
        assert len(z[0].shape) == 3, 'Examples must have the format ([batch_size, wt, wf], [batch_size,ht, hf])'
        assert len(z[1].shape) == 3, 'Examples must have the format ([batch_size, wt, wf], [batch_size,ht, hf])'
        assert z[0].shape[0] == z[1].shape[0], 'The same number of objects (x) and labels (y) need to be provided'
        for i in range(z[0].shape[0]):
            self.known_examples.append(z[0][i], z[1][i])

    def _split_dataset(self):
        # splitting the training set
        if isinstance(self.proper_training_set_size, float):
            n_proper_train = int(self.proper_training_set_size * len(self.known_examples))
        elif isinstance(self.proper_training_set_size, int):
            n_proper_train = self.proper_training_set_size
        else:
            raise TypeError('proper_training_set_size should be float or int.')
        n_cal = len(self.known_examples) - n_proper_train
        logging.debug('Dataset size, proper training set, calibration set: '
                      + f'{len(self.known_examples)}, {n_proper_train}, {n_cal}')

        if self.random_split:
            proper_train_set, cal_set = torch.utils.data.random_split(
                self.known_examples, (n_proper_train, n_cal))
        else:
            proper_train_set = torch.utils.data.Subset(self.known_examples, range(n_proper_train))
            cal_set = torch.utils.data.Subset(self.known_examples, range(n_proper_train, n_proper_train + n_cal))

        return proper_train_set, cal_set

    def _predict_cal_y_hat(self) -> list[torch.Tensor]:
        """
        Generates predictions for the calibration set using the underlying model.
        :return:
        """
        # generate predictions for the calibration set
        cal_loader = torch.utils.data.DataLoader(self.cal_set, batch_size=1, shuffle=False)
        with torch.set_grad_enabled(False):
            self.model.eval()  # sets model into evaluation mode
            cal_y_hat = []
            for cal_example in cal_loader:
                sequences, targets = cal_example  # sequences = windows if windowed
                out, _ = self.model(sequences)
                cal_y_hat.append(out)
        return cal_y_hat

    def _comp_quantiles(self) -> None:
        """
        Computes the width of the intervals for every alpha and every time step and feature in the prediction horizon.
        """
        assert len(self.ncs) > 0, 'No nonconformity scores are available to compute the quantiles.'

        residuals = torch.vstack(self.ncs)  # [n_samples, ht, hf]
        n_cal = residuals.shape[0]
        n_hf = residuals.shape[1] * residuals.shape[2]  # number of label time steps and features
        logging.debug(f'n_cal ={n_cal}, n_hf ={n_hf}')

        # The epsilons for the quantile function are computed according to the chosen correction
        if self.correction == 'no':
            epsilon = 1 - self.alpha
        elif self.correction == 'bonferroni':
            epsilon = 1 - (self.alpha / n_hf)
        elif self.correction == 'independent':
            epsilon = np.power(1 - self.alpha, 1 / n_hf)
        else:
            raise ValueError(f'Correction method {self.correction} not supported.')
        logging.debug(f'epsilon: {epsilon}')

        # todo: make this entire mess more beautiful and efficient
        if self.weight_f is None:
            # this block is left as a comparison to the standard quantile function which should not be used in the
            # nonexchangeable setting.
            logging.warning('Only use the unweighted quantile function if the examples are exchangeable.')
            for i, a in enumerate(self.alpha):
                q_i = epsilon[i]
                self.q[a] = torch.tensor([
                    [torch.quantile(residuals[:, t, f], q=q_i, interpolation='higher')
                     for f in range(residuals.shape[2])]
                    for t in range(residuals.shape[1])
                ]).to(self.device)
                logging.debug(f'weight_f_none: self.q[a].shape = {self.q[a].shape}')
        else:
            logging.debug('Weight f used')
            weights = np.array(self.weight_f(n_cal))
            wq = []
            for f in range(residuals.shape[2]):
                wq_f = []  # quantiles for the current feature
                for t in range(residuals.shape[1]):  # quantiles for current feature and time step
                    values = residuals[:, t, f].detach().cpu().numpy()
                    logging.debug(f'values.shape = {values.shape}')
                    wq_f.append(ICP.weighted_quantile(values, epsilon, weights))
                wq.append(np.array(wq_f))  # todo: comment on dimensions
            wq = np.array(wq).T  # todo: comment on dimensions
            logging.debug(f'wq.shape = {wq.shape}')

            for i, a in enumerate(self.alpha):
                self.q[a] = torch.tensor(wq[i, :]).to(self.device)
                logging.debug(f'weight_f_named: self.q[a].shape = {self.q[a].shape}')

    # todo: add possibility to save and restore?
