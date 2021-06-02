
import sys, gc
sys.path.append('/home/igr/src/model/nengo_maze_env')
from functools import partial
import nengo
import numpy as np
#import matplotlib.pyplot as plt
from prf_net import PRF
from cdisp import CDISP
from hsp import HSP
from isp import ISP
from tqdm import tqdm
import scipy.interpolate
from scipy.interpolate import Rbf, PchipInterpolator, Akima1DInterpolator
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo.utils.progress import TerminalProgressBar

def distance_probs(dist, sigma):                                                                                                   
    weights = np.exp(-dist/sigma**2)                                                                                               
    prob = weights / weights.sum(axis=0)                                                                                           
    return prob                                                                                                                    


def contiguous_ranges(input, return_indices=False):
    """Finds contiguous regions of the array "input". Returns
    a list of ranges with the start and end index of each region. Code based on:
    https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array/4495197
    """

    # Find the indices of changes in "condition"
    d = np.diff(input)
    nz, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the ranges by 1 to the right.
    nz += 1
    nz = np.concatenate([nz, [input.size]])
    
    ranges = np.vstack([ [nz[ri], nz[ri+1]] for ri in range(nz.size-1) ])
    ranges = np.vstack([[0, nz[0]], ranges])

    # Reshape the result into two columns
    ranges.shape = (-1,2)

    if return_indices:
        result = ( np.arange(*r) for r in ranges )
    else:
        result = ranges

    return result

dof_input_matrix = np.asarray(np.load("coral_env_dof_array.npy"), dtype=np.float32)
print(f'dof_input_matrix shape: {dof_input_matrix.shape}')
cnt_input_matrix = np.asarray(np.load("coral_env_cnt_array.npy"), dtype=np.float32)
print(f'cnt_input_matrix shape: {cnt_input_matrix.shape}')

n_samples = dof_input_matrix.shape[-1]

dof_input_flat = dof_input_matrix.reshape((-1, n_samples))
cnt_input_flat = cnt_input_matrix.reshape((-1, n_samples))

n_steps = 100
n_trials = 3
input_matrix = np.vstack((dof_input_flat, cnt_input_flat))
print(f'input_matrix shape: {input_matrix.shape}')
normed_input_matrix = input_matrix / np.max(input_matrix)
train_data = np.tile(np.repeat(normed_input_matrix[:,:60], n_steps, axis=1), (1, n_trials))
print(f'train_data shape: {train_data.shape}')
print(np.max(train_data))
np.save("srf_nengo_dof_input_matrix.npy", train_data)


def array_input(input_matrix, dt, t, *args):
    i = int(t/dt)
    if i >= input_matrix.shape[-1]:
        i = -1
    return input_matrix[:,i]


N_inputs = train_data.shape[0]

N_outputs_srf = 1000
N_exc_srf = N_inputs
N_inh_srf = int(N_outputs_srf/4)


N_inh_decoder = int(N_inh_srf)

dt = 0.001
srf_seed = 19
t_end = train_data.shape[1] * dt
print(f't_end = {t_end}')


srf_place_network = nengo.Network(label="Learning with spatial receptive fields", seed=srf_seed)
rng = np.random.RandomState(seed=srf_seed)



params = {'w_initial_E': 0.0817, 
          'w_initial_EI': 0.00146, 
          'w_initial_I': -0.0355, 
          'w_EI_Ext': 0.0058, 
          'w_PV_E': 0.015, 
          'w_PV_I': 0.044, 
          'p_E_srf': 0.2, 
          'p_EE': 0.05, 
          'p_EI': 0.1,
          'p_EI_Ext': 0.007, 
          'p_PV': 0.01, 
          'tau_E': 0.005, 
          'tau_I': 0.020, 
          'learning_rate_I': 0.01, 
          'learning_rate_E': 0.05}

params = {'w_initial_E': 0.01, 
          'w_initial_EI': 0.012681074, 
          'w_initial_I': -0.028862255, 
          'w_EI_Ext': 0.02956312, 
          'w_PV_E': 0.005, 
          'w_PV_I': 0.001, 
          'p_E_srf': 0.2, 
          'p_EE': 0.01, 
          'p_EI': 0.1,
          'p_EI_Ext': 0.007, 
          'p_PV': 0.2, 
          'tau_E': 0.005, 
          'tau_I': 0.020, 
          'tau_input': 0.1,
          'learning_rate_I': 0.01, 
          'learning_rate_E': 0.04596530}



with srf_place_network as model:
    
    
    srf_network = PRF(exc_input_func=partial(array_input, train_data, dt),
                      connect_exc_inh_input = True,
                      n_excitatory = N_exc_srf,
                      n_inhibitory = N_inh_srf,
                      n_outputs = N_outputs_srf,
                      isp_target_rate = 1.0,
                      tau_input = params['tau_input'],

                      w_initial_E = params['w_initial_E'],
                      w_initial_I = params['w_initial_I'],
                      w_initial_EI = params['w_initial_EI'],
                      w_EI_Ext = params['w_EI_Ext'],
                      p_E = params['p_E_srf'],
                      p_EE = params['p_EE'],
                      p_EI_Ext = params['p_EI_Ext'],
                      p_EI = params['p_EI'],
                      tau_E = params['tau_E'],
                      tau_I = params['tau_I'],
                      learning_rate_I=params['learning_rate_I'],
                      learning_rate_E=params['learning_rate_E'],
               
                      label="Spatial receptive field network",
                      seed=srf_seed)
    
    decoder = nengo.Ensemble(N_exc_srf, dimensions=1,
                             neuron_type = nengo.SpikingRectifiedLinear(),                                                                  
                             radius = 1,                                                                                             
                             intercepts=nengo.dists.Choice([0.1]),                                                 
                             max_rates=nengo.dists.Choice([40]))
    decoder_inh =  nengo.Ensemble(N_inh_decoder, dimensions=1,
                             neuron_type = nengo.RectifiedLinear(),                                                                  
                             radius = 1,                                                                                             
                             intercepts=nengo.dists.Choice([0.1]),                                                 
                             max_rates=nengo.dists.Choice([100]))

 
    w_PV_E = params['w_PV_E']
    p_PV = params['p_PV']
    #weights_dist_PV_E = rng.uniform(size=N_outputs_srf*N_exc_srf).reshape((N_exc_srf, N_outputs_srf))
    weights_initial_PV_E = rng.uniform(size=N_outputs_srf*N_exc_srf).reshape((N_exc_srf, N_outputs_srf)) * w_PV_E
    r = float(N_exc_srf) / float(N_outputs_srf)
    for i in range(N_exc_srf):
        dist = np.abs(i - (r * np.asarray(range(N_outputs_srf))))                                                                    
        sigma = p_PV * N_outputs_srf / 10.0                                                                                       
        prob = distance_probs(dist, sigma)    
        sources = np.asarray(rng.choice(N_outputs_srf, round(p_PV * N_outputs_srf), replace=False, p=prob), dtype=np.int32)
        weights_initial_PV_E[i, np.logical_not(np.in1d(range(N_outputs_srf), sources))] = 0.

    conn_PV_E = nengo.Connection(srf_network.output.neurons,
                             decoder.neurons,
                             transform=weights_initial_PV_E,
                             synapse=nengo.Alpha(params['tau_E']))
                             #learning_rule_type=HSP(learning_rate=1e-2))


    w_PV_I = params['w_PV_I']
    weights_initial_PV_I = rng.uniform(size=N_inh_decoder*N_outputs_srf).reshape((N_inh_decoder, N_outputs_srf)) * w_PV_I
    r = float(N_outputs_srf) / float(N_inh_decoder)
    for i in range(N_inh_decoder):
        dist = np.abs(r*i - np.asarray(range(N_outputs_srf)))                                                            
        sigma = p_PV * N_outputs_srf / 10.0                                                                                       
        prob = distance_probs(dist, sigma)    
        sources = np.asarray(rng.choice(N_outputs_srf, round(p_PV * N_outputs_srf), replace=False, p=prob), dtype=np.int32)
        weights_initial_PV_I[i, np.logical_not(np.in1d(range(N_outputs_srf), sources))] = 0.
    conn_PV_I = nengo.Connection(srf_network.output.neurons,
                             decoder_inh.neurons,
                             transform=weights_initial_PV_I,
                             synapse=nengo.Alpha(params['tau_E']))
    
    w_decoder_I = params['w_initial_I']
    weights_initial_decoder_I = rng.uniform(size=N_inh_decoder*N_exc_srf).reshape((N_exc_srf, N_inh_decoder)) * w_decoder_I

    conn_decoder_I = nengo.Connection(decoder_inh.neurons,
                             decoder.neurons,
                             transform=weights_initial_decoder_I,
                             synapse=nengo.Alpha(params['tau_I']),
                             learning_rule_type=CDISP(learning_rate=0.025))
 
    coincidence_detection = nengo.Node(size_in=N_inputs*2, size_out=N_inputs,
                                      output=lambda t,x: np.subtract(x[:N_inputs], x[N_inputs:]) * 0.1)
    nengo.Connection(coincidence_detection, conn_decoder_I.learning_rule)
    nengo.Connection(srf_network.exc.neurons, coincidence_detection[:N_inputs])
    nengo.Connection(decoder.neurons, coincidence_detection[N_inputs:])
                     
        
    with srf_network:
        p_output_spikes = nengo.Probe(srf_network.output.neurons, synapse=None)
        p_exc_rates = nengo.Probe(srf_network.exc.neurons)
        p_inh_rates = nengo.Probe(srf_network.inh.neurons)
        p_inh_weights = nengo.Probe(srf_network.conn_I, 'weights', sample_every=1.0)
        p_exc_weights = nengo.Probe(srf_network.conn_E, 'weights', sample_every=1.0)
        if srf_network.conn_EE is not None:
            p_rec_weights = nengo.Probe(srf_network.conn_EE, 'weights', sample_every=1.0)

    p_weights_PV_E = nengo.Probe(conn_PV_E, 'weights', sample_every=1.0)
    p_weights_PV_I = nengo.Probe(conn_PV_I, 'weights', sample_every=1.0)
    p_weights_decoder_I = nengo.Probe(conn_decoder_I, 'weights', sample_every=1.0)
    p_decoder_spikes = nengo.Probe(decoder.neurons, synapse=None)
    p_decoder_inh_rates = nengo.Probe(decoder_inh.neurons, synapse=None)
    p_cd = nengo.Probe(coincidence_detection, synapse=None, sample_every=0.1)

    
with nengo.Simulator(model, optimize=True, progress_bar=TerminalProgressBar()) as sim:
    sim.run(np.max(t_end))


output_spikes = sim.data[p_output_spikes]
exc_rates = sim.data[p_exc_rates]
inh_rates = sim.data[p_inh_rates] 
decoder_spikes = sim.data[p_decoder_spikes]
decoder_inh_rates = sim.data[p_decoder_inh_rates]
output_rates = rates_kernel(sim.trange(), output_spikes, tau=0.1)

exc_weights = sim.data[p_exc_weights]
#inh_weights = sim.data[p_inh_weights] 
np.save("output_spikes_dof.npy", output_spikes)
np.save("output_rates_dof.npy", output_rates)
np.save("decoder_spikes_dof.npy", decoder_spikes)
np.save("exc_rates_dof.npy", exc_rates)
np.save("inh_rates_dof.npy", inh_rates)

