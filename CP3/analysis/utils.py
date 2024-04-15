import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def get_dataset_settings(home_dir, data_name):
    # Initialize settings dictionary
    settings = {}
    
    # Define settings for each dataset
    if data_name in ["intersection_flow", "intersection_speed"]:
        settings = {
            'lower_bound': -5,
            'unguided': -10,
            'upper_bound': 0,
            'delta_min': 1,
            'delta_max': 20,
            'slope': 0.005
        }
    else:
        settings = {'error': "data not recognized"}
    
    return settings

def import_data(home_dir, data_name, random=False):
    settings = get_dataset_settings(home_dir, data_name)
    delta_min = settings['delta_min']
    delta_max = settings['delta_max']
    slope = settings['slope']
    lower_bound = settings['lower_bound']
    upper_bound = settings['upper_bound']
    unguided = settings['unguided']
    
    if os.path.exists(home_dir+'/data/'+data_name+'_transfer_result.csv'):
        data_transfer = pd.read_csv(home_dir+'/data/'+data_name+'_transfer_result.csv', header=None)
        # data_transfer = data_transfer.reset_index(drop=True).T.reset_index(drop=True).T
    else:
        print("No data found for", data_name)
    if os.path.exists(home_dir+'/data/'+data_name+'_transfer_result_stdev.csv'):
        data_transfer_std = pd.read_csv(home_dir+'/data/'+data_name+'_transfer_result_stdev.csv', index_col=0)
    else:
        data_transfer_std = pd.DataFrame(np.zeros(data_transfer.shape))
    if random:
        data_transfer_random = data_transfer + data_transfer_std*np.random.normal(0, 0.5, data_transfer.shape)
        data_transfer_random[data_transfer_random < lower_bound] = lower_bound
        data_transfer_random[data_transfer_random > upper_bound] = upper_bound
        data_transfer = data_transfer_random
    deltas = data_transfer.columns.values.astype(float)

    return data_transfer, data_transfer_std, deltas, delta_min, delta_max, slope, lower_bound, upper_bound, unguided

def collect_J_matrix(data_transfer, source_tasks, deltas, num_transfer_steps=15):
    J_tmp = np.zeros((len(deltas), num_transfer_steps))
    for k in range(num_transfer_steps):
        for i in range(len(deltas)):
            if k==0:
                J_tmp[i, k] = data_transfer.iloc[np.where(deltas == source_tasks[k])[0][0]][i]
            else:
                J_tmp[i, k] = max(data_transfer.iloc[np.where(deltas == source_tasks[k])[0][0]][i], J_tmp[i, k-1])
    return J_tmp

def get_baseline_performance(data_transfer, num_transfer_steps):
    deltas = data_transfer.columns.values.astype(float)
    
    # Oracle transfer
    oracle_transfer = [data_transfer.max(axis=0).mean()] * num_transfer_steps
    
    # Exhaustive training
    data_transfer_diagonal = np.zeros(len(deltas))
    for i in range(len(deltas)):
        data_transfer_diagonal[i] = data_transfer.iloc[i][i]
    
    exhaustive_training = [data_transfer_diagonal.mean()] * num_transfer_steps
    
    # Sequential oracle training
    sequential_oracle_training = []
    sot_deltas = []

    # 1st step
    sot_deltas.append(data_transfer.mean(axis=1).argmax())
    sequential_oracle_training.append(data_transfer.iloc[data_transfer.mean(axis=1).argmax(),:].mean())
    for _ in range(num_transfer_steps-1):
        candidate_indices = [x for x in range(len(deltas)) if x not in sot_deltas]
        index_tmp = [data_transfer.T[sot_deltas+[i]].max(axis=1).mean() for i in candidate_indices].index(max([data_transfer.T[sot_deltas+[i]].max(axis=1).mean() for i in candidate_indices]))
        sot_deltas.append(candidate_indices[index_tmp])
        sequential_oracle_training.append(data_transfer.T[sot_deltas].max(axis=1).mean())
    
    return oracle_transfer, exhaustive_training, sequential_oracle_training

def evaluate_on_task(data_transfer, source_tasks, deltas, num_transfer_steps):
    assert len(source_tasks) == num_transfer_steps
    return collect_J_matrix(data_transfer, source_tasks, deltas, num_transfer_steps).mean(axis=0)