"""
Make sure the working directory is the repository root.
"""
import logging
import os.path

from experiments.run_trials import icp

logging.basicConfig(level=logging.INFO,
                    format='%(name)s[%(levelname)s] %(filename)s:%(lineno)s %(funcName)s() %(message)s')

group_out_dir = 'assets/experimental_results/exp_07_longelec2_lin_correction'
group_params_dir = 'assets/experimental_configurations/exp_07_longelec2_lin_correction'

# in this experiment I want to compare and calibrate different weight functions
weight_functions_and_corrections = {
    'bf_soft_cutoff': 'sol_icp_bf_soft_cutoff_c200_s50.json',
    'independent_soft_cutoff': 'sol_icp_independent_soft_cutoff_c200_s50.json',
    'no_soft_cutoff': 'sol_icp_no_soft_cutoff_c200_s50.json',
}
univariate_and_multivariate_ds = {
    'multv': 'elec2_ds_multivariate.json',
}

n_trials = 20

model_params_path = 'assets/experimental_configurations/exp_07_longelec2_lin_correction/model_rnn.json'
assert os.path.isfile(model_params_path)

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
            root_target_dir=group_out_dir, experiment_name=experiment_name)
