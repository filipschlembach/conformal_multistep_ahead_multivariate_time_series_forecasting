import sys
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import logging
import os.path
import time

import numpy as np
import pandas as pd
import torch.utils.data

from conformal_predictors.icp import ICP
from data_ingest.datasets import DatasetFactory, Dataset
import metrics.coverage_rate
import metrics.interval_width
import metrics.mae
from models.utils import select_device, KnownModels
import torch
from utils.params import Params


def _icp_off_line(ds: Dataset, icp_p: Params, icp: ICP) -> (dict, torch.Tensor, torch.Tensor):
    # todo: documentation
    # generating the predictions
    test_ds = torch.utils.data.Subset(ds, range(icp_p.start_offset, len(ds)))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=len(test_ds), shuffle=icp_p.shuffle_test_set)
    test_batch = next(iter(test_loader))
    test_x, test_y = test_batch

    # running ICP off-line
    predictions = icp.forward(test_x)
    for k in predictions.keys():
        predictions[k] = predictions[k].cpu().detach()

    return predictions, test_x.cpu().detach(), test_y.cpu().detach()


def _icp_semi_off_line(ds: Dataset, icp_p: Params, icp: ICP) -> (dict, torch.Tensor, torch.Tensor):
    # todo: documentation
    # dict for storing the results
    predictions = {}
    test_x = []
    test_y = []

    # generating the predictions
    for i, t in enumerate(range(icp_p.start_offset, len(ds))):
        if i % 5 == 0:
            logging.info(f'Iteration {i:5}, Time Step {t:5}, '
                         f'|dataset| = |training set| + |test set|: {len(ds)} = {t} + 1 + {len(ds) - t - 1}')
        # getting the current example as a batch with a single element
        t_loader = torch.utils.data.DataLoader(  # data loader containing a set with only one example
            torch.utils.data.Subset(ds, range(t, t + 1)),  # get the current example as data set
            batch_size=1,
            shuffle=False)
        t_batch = next(iter(t_loader))  # batch containing only one example
        t_x, t_y = t_batch

        # running ICP semi-off-line
        t_p = icp.forward(t_x)
        for k in t_p.keys():
            if k in predictions:
                predictions[k].append(t_p[k].cpu().detach())
            else:
                predictions[k] = [t_p[k].cpu().detach()]
        test_x.append(t_x)
        test_y.append(t_y)
        icp.add_cal_example(t_batch, t_p['model'])

    # stacking the individual predictions: putting all elements from the test set into a single batch
    test_ds = torch.utils.data.Subset(ds, range(icp_p.start_offset, len(ds)))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    test_batch = next(iter(test_loader))
    test_x, test_y = test_batch
    test_x = test_x.cpu().detach()
    test_y = test_y.cpu().detach()

    for k in predictions:
        predictions[k] = torch.vstack(predictions[k])

    return predictions, test_x, test_y


def _icp_on_line(ds: Dataset, icp_p: Params, icp: ICP) -> (dict, torch.Tensor, torch.Tensor):
    # todo: documentation
    # todo: implement
    raise RuntimeError(' not yet implemented')


def icp_trial(i: int, experiment_name: str, trial_names: list, ds_p: Params, model_p: Params, icp_p: Params,
              root_target_dir: str, metric_selection: list, metric_results: dict):
    # todo: documentation
    trial_name = f'{experiment_name}/trial{i:02}'  # this should be unique
    logger = logging.getLogger(trial_name)

    logger.info(f'Starting trial {i}')
    trial_start_time = time.time()

    trial_names.append(trial_name)
    ds_p.experiment = trial_name
    model_p.experiment = trial_name
    icp_p.experiment = trial_name

    experiment_dir = os.path.join(root_target_dir, trial_name)
    os.mkdir(experiment_dir)

    # loading the data set
    logger.info(f'Dataset parameters:\n{ds_p}')
    ds = DatasetFactory.create(**ds_p)

    # preparing the underlying model
    model_p.window_l = ds.wl  # not required for rnn but for lin_reg
    model_p.horizon_l = ds.hl
    model_p.window_f = ds.wf
    model_p.horizon_f = ds.hf
    model_p.path = os.path.join(experiment_dir, model_p.name)
    model_p.device = select_device()
    logger.info(f'Model parameters:\n{model_p}')

    ds.to(model_p.device)

    # initializing the conformal predictor
    icp_p.start_offset = int(icp_p.start_offset_p * len(ds))
    icp_p.n_samples_total = len(ds)  # just for documentation
    train_ds = ds.subset(slice(0, icp_p.start_offset))  # splitting the data set into training and test set
    icp = ICP(known_examples=train_ds,
              model_class=KnownModels.get_model_class(model_p.type),
              model_params=model_p,
              alpha=icp_p.alpha,
              proper_training_set_size=icp_p.proper_training_set_size,
              random_split=icp_p.random_split,
              correction=icp_p.correction,
              device=model_p.device,
              weight_f=ICP.WeightFunctions.get(
                  icp_p.weight_function,
                  fct_params=icp_p.weight_function_params if 'weight_function_params' in icp_p else None
              ))
    icp.fit()  # fitting the underlying model to the proper training set
    icp.cal()  # calibrating the cp on the calibration set
    logger.info(f'ICP parameters:\n{icp_p}')

    if icp_p.setting == 'off_line':
        predictions, test_x, test_y = _icp_off_line(ds, icp_p, icp)
    elif icp_p.setting == 'semi_off_line':
        predictions, test_x, test_y = _icp_semi_off_line(ds, icp_p, icp)
    elif icp_p.setting == 'on_line':
        predictions, test_x, test_y = _icp_on_line(ds, icp_p, icp)
    else:
        raise ValueError(f'Setting "{icp_p.setting}" unknown.')
    icp_p.n_samples_test_set = len(test_x)  # just for documentation

    # saving training parameters and predictions
    ds_p.save()
    model_p.save()
    icp_p.save()
    for k in predictions.keys():
        with open(os.path.join(experiment_dir, f'test_{k}.npy'), 'wb') as f:
            np.save(f, np.array(predictions[k]))
    with open(os.path.join(experiment_dir, f'test_x.npy'), 'wb') as f:
        np.save(f, np.array(test_x))
    with open(os.path.join(experiment_dir, f'test_y.npy'), 'wb') as f:
        np.save(f, np.array(test_y))

    # evaluation
    for mtrc in metric_selection:
        logger.info(f'Evaluating {mtrc.name()}.')
        try:
            score = mtrc.eval(test_y, predictions, rolling_window_length=icp_p.rolling_window_length)
            metric_results[mtrc.snake_name()].append(score)
            logger.debug(f'Saving {mtrc.name()} evaluation result.')
            mtrc.save(score, os.path.join(experiment_dir, mtrc.snake_name()), logger=logger)
            # TODO: The plots (matplotlib) seem to be causing a lot of issues. Therefore the creation of plots for every
            #  trial and metric is commented out until a permanent solution is found.
            # logger.info(f'Plotting {mtrc.name()} evaluation result.')
            # mtrc.simple_plot(score, f'{mtrc.name()} {experiment_name} t{i}',
            #                  os.path.join(experiment_dir, mtrc.snake_name() + '.png'),
            #                  display=False)
            logger.debug(f'Done evaluating {mtrc.name()}.')
        except ValueError as e:
            logger.error(e)
        except IndexError as e:
            logger.error(e)
        except TypeError as e:
            logger.error(e)
        except:
            logger.error(f'Unexpected error: {sys.exc_info()[0]} in {mtrc.name()}.')
            raise
        finally:
            logger.error(f'Cold not complete evaluation of {mtrc.name()}.') # todo: fix this, always produces an error mesage in the log

    summary_df = pd.DataFrame({
        'name': [trial_name],
        'runtime': [int(time.time() - trial_start_time)],
        'mode': ['semi-off-line'],
    })
    summary_df.T.to_csv(os.path.join(experiment_dir, 'summary.csv'))

    logger.info(f'Trial {i} completed in {int(time.time() - trial_start_time)} s.')


def icp(ds_params_path: str, model_params_path: str, icp_params_path: str, n_trials: int = 5,
        root_target_dir: str = None, experiment_name: str = None, n_threads: int = 6):
    """
    Runs icp as defined by the parameter objects.
    :param ds_params_path: dataset parameters
    :param model_params_path: parameters of the underlying model
    :param icp_params_path: parameters of the ICP
    :param n_trials: how often the experiment is repeated
    :param root_target_dir: location where the results will be stored
    :param experiment_name: subdir for the actual experiment
    :param n_threads: number of threads for parallel execution
    :return:
    """
    start_time = time.time()

    # loading the parameter objects
    ds_p = Params.load(ds_params_path)
    model_p = Params.load(model_params_path)
    icp_p = Params.load(icp_params_path)
    if 'rolling_window_length' not in icp_p:
        icp_p.rolling_window_length = 300

    # setting output paths
    ds_p.root_dir = root_target_dir
    model_p.root_dir = root_target_dir
    icp_p.root_dir = root_target_dir

    trial_names = []

    # metrics for the evaluation of the results
    metric_selection = [
        metrics.mae.MAE,
        metrics.mae.RollingMAE,
        metrics.interval_width.MeanIntervalWidth,
        metrics.interval_width.MeanIntervalWidthByDimension,
        metrics.interval_width.MeanIntervalWidthByFeature,
        metrics.interval_width.MeanIntervalWidthByTimeStep,
        metrics.interval_width.RollingMeanIntervalWidth,
        metrics.interval_width.RollingMeanIntervalWidthByDimension,
        metrics.interval_width.RollingMeanIntervalWidthByFeature,
        metrics.interval_width.RollingMeanIntervalWidthByTimeStep,
        metrics.coverage_rate.CoverageRate,
        metrics.coverage_rate.CoverageRateByDimension,
        metrics.coverage_rate.CoverageRateByFeature,
        metrics.coverage_rate.CoverageRateByTimeStep,
        metrics.coverage_rate.RollingCoverageRate,
        metrics.coverage_rate.RollingCoverageRateByDimension,
        metrics.coverage_rate.RollingCoverageRateByFeature,
        metrics.coverage_rate.RollingCoverageRateByTimeStep
    ]
    metric_results = {mtrc.snake_name(): [] for mtrc in metric_selection}

    os.mkdir(os.path.join(root_target_dir, experiment_name))

    # run the trials
    icp_trial_params = [(i, experiment_name, trial_names, ds_p.cp(), model_p.cp(), icp_p.cp(), root_target_dir,
                         metric_selection, metric_results) for i in range(n_trials)]
    executor = ThreadPoolExecutor(max_workers=n_threads)
    futures = [executor.submit(icp_trial, *args) for args in icp_trial_params]
    wait(futures, timeout=None, return_when=ALL_COMPLETED)
    logging.info('Completed execution of all trials for this experiment. Starting to compute summary.')

    # aggregate the results over multiple trials
    aggregate_dir = os.path.join(os.path.join(ds_p.root_dir, experiment_name), 'summary')
    os.mkdir(aggregate_dir)
    for mtrc in metric_selection:
        try:
            scores_all_trials = metric_results[mtrc.snake_name()]  # list of scores
            score_aggregate = mtrc.aggregate(scores_all_trials)
            mtrc.save(score_aggregate, os.path.join(aggregate_dir, mtrc.snake_name()))
            mtrc.simple_plot(score_aggregate, f'{mtrc.name()} {experiment_name}',
                             os.path.join(aggregate_dir, mtrc.snake_name() + '.png'),
                             display=False)
        except ValueError as e:
            logging.error(e)
        except IndexError as e:
            logging.error(e)
        except TypeError as e:
            logging.error(e)
        except:
            logging.error(f'Unexpected error: {sys.exc_info()[0]} in {mtrc.name()}.')
            raise
        finally:
            logging.error(f'Cold not complete evaluation of {mtrc.name()}.')

    logging.info(f'Done. Total wall time {int(time.time() - start_time)} s.')
