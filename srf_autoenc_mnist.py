
import sys, gc
from functools import partial
import nengo
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from prf_net import PRF
from cdisp import CDISP
from hsp import HSP
from isp import ISP
from mnist_data import generate_inputs
import scipy.interpolate
from scipy.interpolate import Rbf, PchipInterpolator, Akima1DInterpolator
from scipy.spatial.distance import cdist
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo.utils.progress import TerminalProgressBar
from mnist_data import generate_inputs

def distance_probs(dist, sigma):                                                                                                   
    weights = np.exp(-dist/sigma**2)                                                                                               
    prob = weights / weights.sum(axis=0)                                                                                           
    return prob                                                                                                                    

def array_input(input_matrix, dt, t, *args):
    i = int(t/dt)
    if i >= input_matrix.shape[-1]:
        i = -1
    return input_matrix[i].ravel()


def build_network(params, inputs, coords, seed=None, dt=0.001):
    
    if seed is None:
        seed = 19

    local_random = np.random.RandomState(seed)
        
    srf_output_coords = coords['srf_output']
    srf_exc_coords = coords['srf_exc']
    srf_inh_coords = coords['srf_inh']

    decoder_coords = coords['decoder']
    decoder_inh_coords = coords['decoder_inh']

    n_inputs = np.product(inputs.shape[1:])
    n_outputs = len(srf_output_coords)
    n_exc = len(srf_exc_coords)
    n_inh = len(srf_inh_coords)
    n_inh_decoder = len(decoder_inh_coords)
    
    autoencoder_network = nengo.Network(label="Learning with spatial receptive fields", seed=seed)

    with autoencoder_network as model:

        srf_network = PRF(exc_input_func = partial(array_input, inputs, dt),
                          connect_exc_inh_input = True,
                          connect_out_out = False,
                          n_excitatory = n_exc,
                          n_inhibitory = n_inh,
                          n_outputs = n_outputs,

                          output_coordinates = srf_output_coords,
                          exc_coordinates = srf_exc_coords,
                          inh_coordinates = srf_inh_coords,
                          
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
                          tau_input = params['tau_input'],
                          learning_rate_I=params['learning_rate_I'],
                          learning_rate_E=params['learning_rate_E'],

                          isp_target_rate = 2.0,
                          label="Spatial receptive field network",
                          seed=seed)
        
        decoder = nengo.Ensemble(n_exc, dimensions=1,
                                 neuron_type = nengo.SpikingRectifiedLinear(),
                                 radius = 1,
                                 intercepts=nengo.dists.Choice([0.1]),                                                 
                                 max_rates=nengo.dists.Choice([40]))

        decoder_inh =  nengo.Ensemble(n_inh, dimensions=1,
                                      neuron_type = nengo.RectifiedLinear(),
                                      radius = 1,
                                      intercepts=nengo.dists.Choice([0.1]),                                                 
                                      max_rates=nengo.dists.Choice([100]))

        w_PV_E = params['w_PV_E']
        p_PV = params['p_PV']
        weights_initial_PV_E = local_random.uniform(size=n_outputs*n_exc).reshape((n_exc, n_outputs)) * w_PV_E
        for i in range(n_exc):
            dist = cdist(decoder_coords[i,:].reshape((1,-1)), srf_output_coords).flatten()
            sigma = 1
            prob = distance_probs(dist, sigma)    
            sources = np.asarray(local_random.choice(n_outputs, round(p_PV * n_outputs), replace=False, p=prob), dtype=np.int32)
            weights_initial_PV_E[i, np.logical_not(np.in1d(range(n_outputs), sources))] = 0.

        conn_PV_E = nengo.Connection(srf_network.output.neurons,
                                     decoder.neurons,
                                     transform=weights_initial_PV_E,
                                     synapse=nengo.Alpha(params['tau_E']))

        w_PV_I = params['w_PV_I']
        weights_initial_PV_I = local_random.uniform(size=n_inh*n_outputs).reshape((n_inh, n_outputs)) * w_PV_I
        for i in range(n_inh_decoder):
            dist = cdist(decoder_inh_coords[i,:].reshape((1,-1)), srf_output_coords).flatten()
            sigma = 1
            prob = distance_probs(dist, sigma)    
            sources = np.asarray(local_random.choice(n_outputs, round(p_PV * n_outputs), replace=False, p=prob), dtype=np.int32)
            weights_initial_PV_I[i, np.logical_not(np.in1d(range(n_outputs), sources))] = 0.
        conn_PV_I = nengo.Connection(srf_network.output.neurons,
                                     decoder_inh.neurons,
                                     transform=weights_initial_PV_I,
                                     synapse=nengo.Alpha(params['tau_E']))
    
        w_decoder_I = params['w_initial_I']
        weights_initial_decoder_I = local_random.uniform(size=n_inh*n_exc).reshape((n_exc, n_inh)) * w_decoder_I

        conn_decoder_I = nengo.Connection(decoder_inh.neurons,
                                          decoder.neurons,
                                          transform=weights_initial_decoder_I,
                                          synapse=nengo.Alpha(params['tau_I']),
                                          learning_rule_type=CDISP(learning_rate=0.025))
    
        coincidence_detection = nengo.Node(size_in=2*n_inputs, size_out=n_inputs,
                                           output=lambda t,x: np.subtract(x[:n_inputs], x[n_inputs:]) * 0.1)
        nengo.Connection(coincidence_detection, conn_decoder_I.learning_rule)
        nengo.Connection(srf_network.exc.neurons, coincidence_detection[:n_inputs])
        nengo.Connection(decoder.neurons, coincidence_detection[n_inputs:])

        p_srf_rec_weights = None
        with srf_network:
            p_srf_output_spikes = nengo.Probe(srf_network.output.neurons, 'output', synapse=None)
            p_srf_exc_rates = nengo.Probe(srf_network.exc.neurons, 'output')
            p_srf_inh_rates = nengo.Probe(srf_network.inh.neurons, 'output')
            p_srf_inh_weights = nengo.Probe(srf_network.conn_I, 'weights')
            p_srf_exc_weights = nengo.Probe(srf_network.conn_E, 'weights')
            if srf_network.conn_EE is not None:
                p_srf_rec_weights = nengo.Probe(srf_network.conn_EE, 'weights')
                
        p_decoder_spikes = nengo.Probe(decoder.neurons, synapse=None)
        p_decoder_inh_rates = nengo.Probe(decoder_inh.neurons, synapse=None)

    return { 'network': autoencoder_network,
             'neuron_probes': {'srf_output_spikes': p_srf_output_spikes,
                               'srf_exc_rates': p_srf_exc_rates,
                               'srf_inh_rates': p_srf_inh_rates,
                               'decoder_spikes': p_decoder_spikes,
                               'decoder_inh_rates': p_decoder_inh_rates},
             'weight_probes': {'srf_inh_weights': p_srf_inh_weights,
                               'srf_exc_weights': p_srf_exc_weights,
                               'srf_rec_weights': p_srf_rec_weights}
    }


def run(model_dict, t_end, dt=0.001):
        
    with nengo.Simulator(model_dict['network'], optimize=True, dt=dt) as sim:
        sim.run(np.max(t_end))

    p_srf_output_spikes = model_dict['neuron_probes']['output_spikes']
    p_exc_rates = model_dict['neuron_probes']['srf_exc_rates']
    p_inh_rates = model_dict['neuron_probes']['srf_inh_rates']
    p_decoder_spikes = model_dict['neuron_probes']['decoder_spikes']
    p_decoder_inh_rates = model_dict['neuron_probes']['decoder_inh_rates']
    srf_output_spikes = sim.data[p_srf_output_spikes]
    srf_exc_rates = sim.data[p_srf_exc_rates]
    srf_inh_rates = sim.data[p_srf_inh_rates]
    decoder_spikes = sim.data[p_decoder_spikes]
    decoder_inh_rates = sim.data[p_decoder_inh_rates]
    np.save("srf_inh_rates", np.asarray(srf_inh_rates, dtype=np.float32))
    np.save("srf_output_spikes", np.asarray(srf_output_spikes, dtype=np.float32))
    np.save("srf_time_range", np.asarray(sim.trange(), dtype=np.float32))
    np.save("decoder_spikes", np.asarray(decoder_spikes, dtype=np.float32))
    srf_output_rates = rates_kernel(sim.trange(), srf_output_spikes, tau=0.1)
    #sorted_idxs = np.argsort(-np.argmax(output_rates[53144:].T, axis=1))

    return {'srf_output_rates': srf_output_rates,
            'srf_exc_rates': srf_exc_rates,
            'srf_inh_rates': srf_inh_rates }
    
#output_rates = sim.data[p_output_rates]
#plot_spikes(sim.trange(), sim.data[p_inh_rates][0,:])

seed=23

train_image_array, test_labels = generate_inputs(plot=False, dataset='training', seed=seed)

n_x = train_image_array.shape[1]
n_y = train_image_array.shape[2]

n_steps = 100

normed_train_image_array = train_image_array / np.max(train_image_array)
train_data = np.repeat(normed_train_image_array[:30], n_steps, axis=0)

print(f'train_data shape: {train_data.shape}')
n_outputs=50
n_exc=n_x*n_y
n_inh=100

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


model_dict = build_network(params, inputs=train_data,
                           coords=coords_dict,
                           seed=seed)
dt = 0.001
t_end = float(train_data.shape[0]) * dt
output_dict = run(model_dict, t_end, dt=dt)
