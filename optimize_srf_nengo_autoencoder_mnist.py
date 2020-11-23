import sys, math
import numpy as np
from functools import partial
import nengo
from nengo_extras.neurons import (rates_kernel, rates_isi )
from prf_net import PRF
from tqdm import tqdm
import distgfs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

input_matrix = np.load("srf_nengo_autoencoder_input_matrix.npy")

def modulation_index(rates):
    mod_idxs = []
    for i in range(rates.shape[1]):
        rates_i = rates[:, i]
        peak_pctile = np.percentile(rates_i, 90)
        med_pctile = np.percentile(rates_i, 50)
        peak_idxs = np.argwhere(rates_i >= peak_pctile)
        med_idxs = np.argwhere(rates_i <= med_pctile)
        mod_index = np.sum(rates_i[peak_idxs]) / np.max(np.sum(rates_i[med_idxs]), 1e-4)
        mod_idxs.append(mod_index)
        logger.info(f"modulation_index {i}: peak_pctile: {peak_pctile} med_pctile: {med_pctile} mod_index: {mod_index}")
    res = np.mean(mod_idxs)
    return res
        


def array_input(input_matrix, dt, t, *args):
    i = int(t/dt)
    if i >= input_matrix.shape[0]:
        i = -1
    return input_matrix[i,:]

            
def eval_srf_net(input_matrix, params):

    srf_network = PRF(exc_input_func = partial(array_input, input_matrix, params['dt']),
                          connect_exc_inh_input = True,
                          n_excitatory = params['N_Exc'],
                          n_inhibitory = params['N_Inh'],
                          n_outputs = params['N_Outputs'],
                          w_initial_E = params['w_initial_E'],
                          w_initial_I = params['w_initial_I'],
                          w_initial_EI = params['w_initial_EI'],
                          w_EI_Ext = params['w_EI_Ext'],
                          p_EI_Ext = params['p_EI_Ext'],
                          p_E = params['p_E'],
                          p_EE = params['p_EE'],
                          tau_E = params['tau_E'],
                          tau_I = params['tau_I'],
                          tau_input = params['tau_input'],
                          isp_target_rate = params['isp_target_rate'],
                          learning_rate_I=params['learning_rate_I'],
                          learning_rate_E=params['learning_rate_E'],
                          label="Spatial receptive field network",
                          seed=params['seed'])
    
    with srf_network:
        p_output_spikes = nengo.Probe(srf_network.output.neurons, 'spikes', synapse=None)
        
        with nengo.Simulator(srf_network, optimize=True) as sim:
                sim.run(np.max(params['t_end']))
                
        output_spikes = sim.data[p_output_spikes]
        output_rates = rates_kernel(sim.trange(), output_spikes, tau=0.25)
        return modulation_index(output_rates)


def obj_fun(pp, pid):
    """ Objective function to be _maximized_ by GFS. """
    res = eval_srf_net(input_matrix, pp)
    logger.info(f"Iter: {pid}\t pp:{pp}, result:{res}")
    # Since Dlib maximizes, but we want to find the minimum,
    # we negate the result before passing it to the Dlib optimizer.
    return -res
    
if __name__ == '__main__':

    seed = 0
    dt = 0.001
    t_end = input_matrix.shape[0]*dt
    
    N_Exc = input_matrix.shape[1]
    N_Inh = int(N_Exc/4)
    N_Outputs = 100
    problem_parameters = {'seed': seed,
                          'dt': dt,
                          't_end': t_end,
                          'N_Exc': N_Exc,
                          'N_Inh': N_Inh,
                          'N_Outputs': N_Outputs,
                          'p_EI_Ext': 0.25,
                          'tau_input': 0.1,
                          'isp_target_rate': 1.0,
                          'learning_rate_I': 1e-4,
                          'learning_rate_E': 2e-5,
                         }
        
    space = {'w_initial_E': [1e-5, 1e-1],
             'w_initial_EI': [1e-5, 1e-1],
             'w_EI_Ext': [1e-5, 1e-1],
             'w_initial_I': [-1e-1, -1e-5],
             'p_E': [0.01, 0.5],
             'p_EE': [0.01, 0.5],
             'tau_E': [0.01, 0.1],
             'tau_I': [0.01, 0.1],
            }

    # Create an optimizer
    distgfs_params = {'opt_id': 'distgfs_srf_nengo_autoencoder_mnist',
                      'obj_fun_name': 'obj_fun',
                      'obj_fun_module': 'optimize_srf_nengo_autoencoder_mnist',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'n_iter': 2000}
    
    distgfs.run(distgfs_params, spawn_workers=False, verbose=True)
            
