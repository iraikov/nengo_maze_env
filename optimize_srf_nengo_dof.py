import sys, math
import numpy as np
from functools import partial
import nengo
from nengo_extras.neurons import (rates_kernel, rates_isi )
from prf_net import PRF
from ei_net import EI
from hsp import HSP
from isp import ISP
from dmosopt import dmosopt
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

def obj_fraction_active(rates, target=0.05):
    n = rates.shape[1]
    bin_fraction_active = []
    for i in range(rates.shape[0]):
        rates_i = rates[i, :]
        a = len(np.argwhere(rates_i >= 1.0))
        bin_fraction_active.append(float(a) / float(n))
        #logger.info(f"fraction_active {i}: a: {a} n: {n} fraction: {float(a) / float(n)}")
    res = (np.mean(bin_fraction_active) - target)**2
    return res

        


def array_input(input_matrix, dt, t, *args):
    i = int(t/dt)
    if i >= input_matrix.shape[0]:
        i = -1
    return input_matrix[:,i]

            
def eval_srf_net(input_matrix, params):

    N_inputs = input_matrix.shape[0]

    N_outputs_srf = params['N_outputs_srf']
    N_exc_srf = N_inputs
    N_inh_srf = params['N_inh_srf']

    N_exc_place = params['N_exc_place']
    N_inh_place = params['N_inh_place']

    dt = 0.001
    t_end = input_matrix.shape[1] * dt

    srf_place_network = nengo.Network(label="Learning with spatial receptive fields", seed=params['seed'])
    rng = np.random.RandomState(seed=params['seed'])

    with srf_place_network as model:
        
        srf_network = PRF(exc_input_func = partial(array_input, input_matrix, params['dt']),
                          connect_exc_inh_input = True,
                          n_excitatory = N_exc_srf,
                          n_inhibitory = params['N_inh_srf'],
                          n_outputs = params['N_outputs_srf'],
                          w_initial_E = params['w_initial_E'],
                          w_initial_I = params['w_initial_I'],
                          w_initial_EI = params['w_initial_EI'],
                          w_EI_Ext = params['w_EI_Ext'],
                          p_E = params['p_E'],
                          p_EE = params['p_EE'],
                          p_EI_Ext = params['p_EI_Ext'],
                          p_EI = params['p_EI'],
                          tau_E = params['tau_E'],
                          tau_I = params['tau_I'],
                          tau_input = params['tau_input'],
                          isp_target_rate = params['isp_target_rate'],
                          learning_rate_I=params['learning_rate_I'],
                          learning_rate_E=params['learning_rate_E'],
                          label="Spatial receptive field network",
                          seed=params['seed'])

        place_network = EI(n_excitatory = params['N_exc_place'],
                           n_inhibitory = params['N_inh_place'],
                           isp_target_rate = params['isp_target_rate'],
                           learning_rate_I=params['learning_rate_I'],
                           learning_rate_E=params['learning_rate_E'],
                           p_EE = params['p_EE'],
                           p_EI = params['p_EI'],
                           tau_E = params['tau_E'], 
                           tau_I = params['tau_I'],
                           tau_input = params['tau_input'],
                           connect_exc_inh = True,
                           label="Place network",
                           seed=params['seed'])

        w_PV_E = params['w_PV_E']
        p_PV_E = params['p_PV_E']
        weights_dist_PV_E = rng.normal(size=N_outputs_srf*N_exc_place).reshape((N_exc_place, N_outputs_srf))
        weights_initial_PV_E = (weights_dist_PV_E - weights_dist_PV_E.min()) / (weights_dist_PV_E.max() - weights_dist_PV_E.min()) * w_PV_E
        for i in range(N_exc_place):
            sources = np.asarray(rng.choice(N_outputs_srf, round(p_PV_E * N_outputs_srf), replace=False),
                                 dtype=np.int32)
            weights_initial_PV_E[i, np.logical_not(np.in1d(range(N_outputs_srf), sources))] = 0.
                
        conn_PV_E = nengo.Connection(srf_network.output.neurons,
                                     place_network.exc.neurons,
                                     transform=weights_initial_PV_E,
                                     synapse=nengo.Alpha(params['tau_E']),
                                     learning_rule_type=HSP(learning_rate=params['learning_rate_E']))
                
        w_SV_E = params['w_PV_E']
        p_SV_E = params['p_SV_E']
        weights_dist_SV_E = rng.normal(size=N_exc_srf*N_exc_place).reshape((N_exc_place, N_exc_srf))
        weights_initial_SV_E = (weights_dist_SV_E - weights_dist_SV_E.min()) / (weights_dist_SV_E.max() - weights_dist_SV_E.min()) * w_SV_E
        for i in range(N_exc_place):
                sources = np.asarray(rng.choice(N_exc_srf, round(p_SV_E * N_exc_srf), replace=False), dtype=np.int32)
                weights_initial_SV_E[i, np.logical_not(np.in1d(range(N_exc_srf), sources))] = 0.

        conn_SV_E = nengo.Connection(srf_network.exc.neurons,
                                     place_network.exc.neurons,
                                     transform=weights_initial_SV_E,
                                     synapse=nengo.Alpha(params['tau_E']),
                                     learning_rule_type=HSP(learning_rate=params['learning_rate_E']))
   
        w_PV_I = params['w_PV_I']
        weights_initial_PV_I = rng.uniform(size=N_inh_place*N_outputs_srf).reshape((N_inh_place, N_outputs_srf)) * w_PV_I

        conn_PV_I = nengo.Connection(srf_network.output.neurons,
                                     place_network.inh.neurons,
                                     transform=weights_initial_PV_I,
                                     synapse=nengo.Alpha(params['tau_I']))

        with place_network:
            p_place_output_spikes = nengo.Probe(place_network.exc.neurons, synapse=None)
        with srf_network:
            p_output_spikes = nengo.Probe(srf_network.output.neurons, synapse=None)
        
    with nengo.Simulator(srf_place_network, optimize=True) as sim:
        sim.run(t_end)
                
    srf_output_spikes = sim.data[p_output_spikes]
    srf_output_rates = rates_kernel(sim.trange(), srf_output_spikes, tau=0.1)
    place_output_spikes = sim.data[p_place_output_spikes]
    place_output_rates = rates_kernel(sim.trange(), place_output_spikes, tau=0.1)

    return np.asarray([obj_modulation_depth(srf_output_rates), 
                       obj_modulation_depth(place_output_rates), 
                       obj_fraction_active(srf_output_rates),
                       obj_fraction_active(place_output_rates)])



def init_obj_fun(worker):
    """ Returns objective function to be minimized. """

    input_matrix = None
    if worker.worker_id == 1:
        input_matrix = np.load("/scratch1/03320/iraikov/srf_nengo_dof_input_matrix.npy")
    input_matrix = worker.group_comm.bcast(input_matrix, root=0)

    def obj_fun(pp, pid=None):
        res = eval_srf_net(input_matrix, pp)
        logger.info(f"Iter: {pid}\t pp:{pp}, result:{res}")
        return res

    return obj_fun
    
if __name__ == '__main__':

    seed = 0
    dt = 0.001
    
    N_outputs_srf = 2000
    N_inh_srf = int(N_outputs_srf/2)

    N_exc_place = 2000
    N_inh_place = int(N_exc_place/2)
    
    problem_parameters = {'seed': seed,
                          'dt': dt,
                          'N_inh_srf': N_inh_srf,
                          'N_outputs_srf': N_outputs_srf,
                          'N_exc_place': N_exc_place,
                          'N_inh_place': N_inh_place,
                          'tau_input': 0.1,
                          'isp_target_rate': 1.0,
                         }
        
    space = {'w_initial_E': [1e-5, 1e-1],
             'w_initial_EI': [1e-5, 1e-1],
             'w_initial_I': [-1e-1, -1e-5],
             'w_EI_Ext': [1e-5, 1e-1],
             'w_PV_E': [1e-5, 1e-1],
             'w_SV_E': [1e-5, 1e-1],
             'w_PV_I': [-1e-1, -1e-5],
             'p_E': [0.01, 0.5],
             'p_EE': [0.01, 0.5],
             'p_EI': [0.01, 0.5],
             'p_EI_Ext': [0.01, 0.5],
             'p_PV_E': [0.001, 0.1],
             'p_SV_E': [0.001, 0.1],
             'tau_E': [0.01, 0.1],
             'tau_I': [0.01, 0.1],
             'learning_rate_I': [1e-4, 1e-1],
             'learning_rate_E': [1e-4, 1e-1],
            }

    objective_names = ['srf_modulation_depth', 'place_modulation_depth', 
                       'srf_fraction_active', 'place_fraction_active']

    dmosopt_params = {'opt_id': 'dmosopt_srf_nengo_dof',
                      'obj_fun_init_name': 'init_obj_fun',
                      'obj_fun_init_args': {},
                      'obj_fun_init_module': 'optimize_srf_nengo_dof',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'n_iter': 10,
                      'objective_names': objective_names,
                      'n_initial': 50,
                      'resample_fraction': 0.8,
                      'mutation_rate': 0.5,
                      'file_path': '/scratch1/03320/iraikov/dmosopt.srf_nengo_dof.h5',
                      'save': True,
                  }
    
    best = dmosopt.run(dmosopt_params, spawn_workers=False, verbose=True)
            
