import sys, math
import numpy as np
from functools import partial
import nengo
from dmosopt import dmosopt
from nengo_extras.neurons import (rates_kernel, rates_isi )
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def obj_modulation_depth(rates):
    mod_depths = []
    for i in range(rates.shape[1]):
        rates_i = rates[:, i]
        peak_pctile = np.percentile(rates_i, 80)
        med_pctile = np.percentile(rates_i, 50)
        peak_idxs = np.argwhere(rates_i >= peak_pctile)
        med_idxs = np.argwhere(rates_i <= med_pctile)
        mean_peak = np.mean(rates_i[peak_idxs])
        mean_med = np.mean(rates_i[med_idxs])
        mod_depth = (mean_peak - mean_med) ** 2.
        mod_depths.append(mod_depth)
        #logger.info(f"modulation_depth {i}: peak_pctile: {peak_pctile} med_pctile: {med_pctile} mod_depth: {mod_depth}")
    res = -np.mean(mod_depths)
    return res


def obj_fraction_active(rates, target=0.1):
    n = rates.shape[1]
    bin_fraction_active = []
    for i in range(rates.shape[0]):
        rates_i = rates[i, :]
        a = len(np.argwhere(rates_i >= 1.0))
        bin_fraction_active.append(float(a) / float(n))
        #logger.info(f"fraction_active {i}: a: {a} n: {n} fraction: {float(a) / float(n)}")
    res = (np.mean(bin_fraction_active) - target)**2
    return res

        


            
def eval_srf_autoenc(train_data, params, coords_dict, seed, dt=0.01):

    model_dict = build_network(params, inputs=train_data,
                               coords=coords_dict,
                               seed=seed, dt=dt)
    t_end = float(train_data.shape[0]) * dt
    output_dict = run(model_dict, t_end, dt=dt)
                
    srf_autoenc_output_rates = output_dict['srf_autoenc_output_rates']
    srf_autoenc_decoder_rates = output_dict['srf_autoenc_decoder_rates']

    return np.asarray([obj_modulation_depth(srf_autoenc_output_rates), 
                       obj_fraction_active(srf_autoenc_output_rates),
                       obj_mse(train_data, srf_autoenc_decoder_rates)])



def init_obj_fun(worker):
    """ Returns objective function to be minimized. """

    input_seed = 29

    n_steps = 100
    n_outputs = 50
    n_inh = 100
    
    input_matrix = None
    n_x = None
    n_y = None
    if worker.worker_id == 1:
        train_image_array, test_labels = generate_inputs(plot=False, dataset='training', seed=input_seed)

        n_x = train_image_array.shape[1]
        n_y = train_image_array.shape[2]

        normed_train_image_array = train_image_array / np.max(train_image_array)
        imput_matrix = np.repeat(normed_train_image_array[:30], n_steps, axis=0) * 10.

    input_matrix = worker.group_comm.bcast(input_matrix, root=0)
    n_x, n_y = worker.group_comm.bcast((n_x, n_y), root=0)
    n_exc=n_x*n_y

    srf_exc_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_y), indexing='ij')])
    srf_inh_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_inh), np.linspace(0, 1, n_inh), indexing='ij')])
    srf_output_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_outputs), np.linspace(0, 1, n_outputs), indexing='ij')])

    decoder_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_y), indexing='ij')])
    decoder_inh_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_inh), np.linspace(0, 1, n_inh), indexing='ij')])

    coords_dict = { 'srf_output': srf_output_coords,
                    'srf_exc': srf_exc_coords,
                    'srf_inh': srf_inh_coords,
                    'decoder': decoder_coords,
                    'decoder_inh': decoder_inh_coords }

    def obj_fun(pp, pid=None):
        res = eval_srf_autoenc(input_matrix, pp, coords_dict, pp['seed'])
        logger.info(f"Iter: {pid}\t pp:{pp}, result:{res}")
        return res

    return obj_fun
    
if __name__ == '__main__':

    seed = 21
    
    problem_parameters = {'seed': seed,
                          'tau_input': 0.1,
                          'isp_target_rate': 1.0,
                         }
        
    space = {'w_initial_E': [1e-5, 1e-1],
             'w_initial_EI': [1e-5, 1e-1],
             'w_initial_I': [-1e-1, -1e-5],
             'w_EI_Ext': [1e-5, 1e-1],
             'w_PV_E': [1e-5, 1e-1],
             'w_SV_E': [1e-5, 1e-1],
             'w_PV_I': [1e-5, 1e-1],
             'p_E_srf': [0.001, 0.25],
             'p_E_place': [0.001, 0.25],
             'p_EE': [0.01, 0.25],
             'p_EI': [0.01, 0.25],
             'p_EI_Ext': [0.001, 0.1],
             'p_PV_E': [0.005, 0.1],
             'p_SV_E': [0.001, 0.1],
             'tau_E': [0.005, 0.05],
             'tau_I': [0.01, 0.05],
             'learning_rate_I': [1e-4, 1e-1],
             'learning_rate_E': [1e-4, 1e-1],
            }

    objective_names = ['srf_modulation_depth', 
                       'place_modulation_depth', 
                       'decoder_mse']

    dmosopt_params = {'opt_id': 'dmosopt_srf_autoenc_mnist',
                      'obj_fun_init_name': 'init_obj_fun',
                      'obj_fun_init_args': {},
                      'obj_fun_init_module': 'optimize_srf_autoenc_mnist',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'n_iter': 10,
                      'objective_names': objective_names,
                      'n_initial': 50,
                      'resample_fraction': 0.9,
                      'mutation_rate': 0.5,
                      'save': True,
                  }
    
    best = dmosopt.run(dmosopt_params, spawn_workers=False, verbose=True)
            
