
import sys, gc
from functools import partial
import nengo
import numpy as np
from prf_net import PRF
from cdisp import CDISP
from hsp import HSP
from isp import ISP
import scipy.interpolate
from scipy.interpolate import Rbf, PchipInterpolator, Akima1DInterpolator
from scipy.spatial.distance import cdist
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo.utils.progress import TerminalProgressBar

def distance_probs(dist, sigma):                                                                                                   
    weights = np.exp(-dist/sigma**2)                                                                                               
    prob = weights / weights.sum(axis=0)                                                                                           
    return prob                                                                                                                    

def array_input(input_matrix, dt, t, *args):
    i = int(t/dt)
    if i >= input_matrix.shape[-1]:
        i = -1
    return input_matrix[i].ravel()


def build_network(params, inputs, coords=None, n_outputs=None, n_exc=None, n_inh=None, n_inh_decoder=None, seed=0, dt=0.001):
    
    local_random = np.random.RandomState(seed)

    if coords is None:
        coords = {}
    
    srf_output_coords = coords.get('srf_output', None)
    srf_exc_coords = coords.get('srf_exc', None)
    srf_inh_coords = coords.get('srf_inh', None)

    decoder_coords = coords.get('decoder', None)
    decoder_inh_coords = coords.get('decoder_inh', None)

    n_inputs = np.product(inputs.shape[1:])
    if srf_output_coords is not None:
        n_outputs = srf_output_coords.shape[0]
    if srf_exc_coords is not None:
        n_exc = srf_exc_coords.shape[0]
    if srf_inh_coords is not None:
        n_inh = srf_inh_coords.shape[0]
    if decoder_inh_coords is not None:
        n_inh_decoder = decoder_inh_coords.shape[0]

    if n_outputs is None:
        raise RuntimeError("n_outputs is not provided and srf_output coordinates are not provided")
    if n_exc is None:
        raise RuntimeError("n_exc is not provided and srf_exc coordinates are not provided")
    if n_inh is None:
        raise RuntimeError("n_exc is not provided and srf_inh coordinates are not provided")
    if n_inh_decoder is None:
        raise RuntimeError("n_inh_decoder is not provided and decoder_inh coordinates are not provided")
    
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
                                           output=lambda t,x: np.subtract(x[:n_inputs], x[n_inputs:]))
        nengo.Connection(coincidence_detection, conn_decoder_I.learning_rule)
        nengo.Connection(srf_network.exc.neurons, coincidence_detection[:n_inputs])
        nengo.Connection(decoder.neurons, coincidence_detection[n_inputs:])

        p_srf_rec_weights = None
        with srf_network:
            p_srf_output_spikes = nengo.Probe(srf_network.output.neurons, 'output', synapse=None)
            p_srf_exc_rates = nengo.Probe(srf_network.exc.neurons, 'output')
            p_srf_inh_rates = nengo.Probe(srf_network.inh.neurons, 'output')
            #p_srf_inh_weights = nengo.Probe(srf_network.conn_I, 'weights')
            #p_srf_exc_weights = nengo.Probe(srf_network.conn_E, 'weights')
            #if srf_network.conn_EE is not None:
            #    p_srf_rec_weights = nengo.Probe(srf_network.conn_EE, 'weights')
                
        p_decoder_spikes = nengo.Probe(decoder.neurons, synapse=None)
        p_decoder_inh_rates = nengo.Probe(decoder_inh.neurons, synapse=None)

    return { 'network': autoencoder_network,
             'neuron_probes': {'srf_output_spikes': p_srf_output_spikes,
                               'srf_exc_rates': p_srf_exc_rates,
                               'srf_inh_rates': p_srf_inh_rates,
                               'decoder_spikes': p_decoder_spikes,
                               'decoder_inh_rates': p_decoder_inh_rates},
#             'weight_probes': {'srf_inh_weights': p_srf_inh_weights,
#                               'srf_exc_weights': p_srf_exc_weights,
#                               'srf_rec_weights': p_srf_rec_weights}
    }


def run(model_dict, t_end, dt=0.001, save_results=False):
        
    with nengo.Simulator(model_dict['network'], optimize=True, dt=dt, progress_bar=TerminalProgressBar()) as sim:
        sim.run(np.max(t_end))

    p_srf_output_spikes = model_dict['neuron_probes']['srf_output_spikes']
    p_srf_exc_rates = model_dict['neuron_probes']['srf_exc_rates']
    p_srf_inh_rates = model_dict['neuron_probes']['srf_inh_rates']
    p_decoder_spikes = model_dict['neuron_probes']['decoder_spikes']
    p_decoder_inh_rates = model_dict['neuron_probes']['decoder_inh_rates']
    srf_output_spikes = sim.data[p_srf_output_spikes]
    srf_exc_rates = sim.data[p_srf_exc_rates]
    srf_inh_rates = sim.data[p_srf_inh_rates]
    decoder_spikes = sim.data[p_decoder_spikes]
    decoder_inh_rates = sim.data[p_decoder_inh_rates]
    srf_output_rates = rates_kernel(sim.trange(), srf_output_spikes, tau=0.1)
    srf_decoder_rates = rates_kernel(sim.trange(), decoder_spikes, tau=0.1)
    if save_results:
        np.save("srf_autoenc_exc_rates", np.asarray(srf_exc_rates, dtype=np.float32))
        np.save("srf_autoenc_inh_rates", np.asarray(srf_inh_rates, dtype=np.float32))
        np.save("srf_autoenc_output_spikes", np.asarray(srf_output_spikes, dtype=np.float32))
        np.save("srf_autoenc_decoder_spikes", np.asarray(decoder_spikes, dtype=np.float32))
        np.save("srf_autoenc_output_rates", np.asarray(srf_output_rates, dtype=np.float32))
        np.save("srf_autoenc_decoder_rates", np.asarray(decoder_rates, dtype=np.float32))
        np.save("srf_autoenc_time_range", np.asarray(sim.trange(), dtype=np.float32))

    return {'srf_autoenc_output_rates': srf_output_rates,
            'srf_autoenc_decoder_rates': decoder_rates,
            'srf_autoenc_exc_rates': srf_exc_rates,
            'srf_autoenc_inh_rates': srf_inh_rates,
            'srf_autoenc_output_spikes': srf_output_spikes,
            'srf_autoenc_decoder_spikes': decoder_spikes,

            
    }
