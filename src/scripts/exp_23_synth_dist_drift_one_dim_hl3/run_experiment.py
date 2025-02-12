"""
Make sure the working directory is the repository root.
"""
import logging
import os.path
import sys

from experiments.run_trials import icp

logging.basicConfig(level=logging.INFO,
                    format='%(name)s[%(levelname)s] %(filename)s:%(lineno)s %(funcName)s() %(message)s')

experiment_dir_name = 'exp_23_synth_dist_drift_one_dim_hl3'
group_out_dir = f'assets/experimental_results/{experiment_dir_name}'
group_params_dir = f'assets/experimental_configurations/{experiment_dir_name}'

# in this experiment I want to compare and calibrate different weight functions
weight_functions_and_corrections = {
    'ol_icp_bf_cst': 'ol_icp_bf_constant.json',
    'sol_icp_bf_cst': 'sol_icp_bf_constant.json',
    'sol_icp_bf_exp_b0.007': 'sol_icp_bf_exponential_b0.007.json',
    'sol_icp_bf_lin': 'sol_icp_bf_linear.json',
    'sol_icp_bf_sc_c200_s50': 'sol_icp_bf_soft_cutoff_c200_s50.json'
}
univariate_and_multivariate_ds = {
    'multv': 'ds_synth_dist_drift.json',
}

n_trials = 20

model_params_path = f'assets/experimental_configurations/{experiment_dir_name}/model_lin_reg.json'
try:
    assert os.path.isfile(model_params_path)
except AssertionError:
    print(f'{model_params_path} is not a file.')
    sys.exit(-1)

# check if paths exist
for ds_setting, ds_params_file in univariate_and_multivariate_ds.items():
    for icp_setting, icp_params_file in weight_functions_and_corrections.items():
        ds_params_path = os.path.join(group_params_dir, ds_params_file)
        icp_params_path = os.path.join(group_params_dir, icp_params_file)
        assert os.path.isfile(ds_params_path)
        assert os.path.isfile(icp_params_path)

for ds_setting, ds_params_file in univariate_and_multivariate_ds.items():
    for icp_setting, icp_params_file in weight_functions_and_corrections.items():
        ds_params_path = os.path.join(group_params_dir, ds_params_file)
        icp_params_path = os.path.join(group_params_dir, icp_params_file)
        experiment_name = f'{ds_setting}_{icp_setting}'
        icp(ds_params_path, model_params_path, icp_params_path, n_trials,
            root_target_dir=group_out_dir, experiment_name=experiment_name,
            n_threads=10)
