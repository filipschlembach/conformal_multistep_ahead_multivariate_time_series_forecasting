"""
Make sure the working directory is the repository root.
"""
import logging
import os.path

from experiments.run_trials import icp_off_line

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(filename)s:%(lineno)s %(funcName)s() %(message)s')

experiment_group_out_dir = 'assets/experimental_results/exp_01_stankeviciute_correction'
experiment_dirs = ['bonferroni', 'independent', 'no']

# constant for the entire experiment group, as only the correction method in the ICP changes
ds_params_path = 'assets/experimental_configurations/exp_01_stankeviciute_correction/dataset_stankeviciute.json'
model_params_path = 'assets/experimental_configurations/exp_01_stankeviciute_correction/model_rnn.json'

for experiment_dir in experiment_dirs:
    experiment_param_dir = os.path.join(experiment_group_out_dir, experiment_dir)
    icp_params_path = f'assets/experimental_configurations/exp_01_stankeviciute_correction' \
                      f'/off_line_icp_{experiment_dir}.json'
    n_trials = 100
    icp_off_line(ds_params_path, model_params_path, icp_params_path, n_trials,
                 root_target_dir=experiment_group_out_dir, experiment_name=experiment_dir)
