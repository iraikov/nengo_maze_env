
import sys, gc
from functools import partial
import nengo
import numpy as np
from prf_net import PRF
from cdisp import CDISP
from hsp import HSP
from isp import ISP
from gsp import GSP
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

class ArrayInput:

    def __init__(self, input_matrix, oob_value, dt):
        self.input_matrix = input_matrix
        self.oob_value = oob_value
        self.dt = dt

    def array_input(self, t):
        i = int(t/self.dt)
        if i >= self.input_matrix.shape[0]:
            i = -1
        if i == -1:
            if self.oob_value is None:
                res = self.input_matrix[i].flatten()
            else:
                res = np.ones(self.input_matrix[i].shape).flatten()*self.oob_value
        else:
            res = self.input_matrix[i].flatten()
            
        return res


def callable_input(inputs, oob_value, t, centered=False):
    if centered:
        result = np.asarray([ 2.*y(t) - 1. for y in inputs ])
    else:
        result = np.asarray([ y(t) for y in inputs ])
    if len(np.isnan(result)) > 0:
        result[np.isnan(result)] = oob_value
    return result


def build_network(params, inputs, dimensions, input_encoders=None, direct_input=True, oob_value=None, coords=None, n_outputs=None, n_exc=None, n_inh=None, n_inh_decoder=None, seed=0, dt=None):
    
    local_random = np.random.RandomState(seed)

    if coords is None:
        coords = {}
    
    srf_output_coords = coords.get('srf_output', None)
    srf_exc_coords = coords.get('srf_exc', None)
    srf_inh_coords = coords.get('srf_inh', None)

    decoder_coords = coords.get('decoder', None)
    decoder_inh_coords = coords.get('decoder_inh', None)
    
    if type(inputs) == np.ndarray:
        n_inputs = np.product(inputs.shape[1:])
    else:
        n_inputs = len(inputs)
    print(f'n_inputs = {n_inputs}')
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
        n_inh_decoder = n_inh

    if srf_exc_coords is None:
        srf_exc_coords = np.asarray(range(n_exc)).reshape((n_exc,1)) / n_exc
    if srf_output_coords is None:
        srf_output_coords = np.asarray(range(n_outputs)).reshape((n_outputs,1)) / n_outputs
    if srf_inh_coords is None:
        srf_inh_coords = np.asarray(range(n_inh)).reshape((n_inh,1)) / n_inh
    if decoder_coords is None:
        decoder_coords = np.asarray(range(n_exc)).reshape((n_exc,1)) / n_exc
    if decoder_inh_coords is None:
        decoder_inh_coords = np.asarray(range(n_inh_decoder)).reshape((n_inh_decoder,1)) / n_inh_decoder

    autoencoder_network = nengo.Network(label="Learning with spatial receptive fields", seed=seed)

    exc_input_func = None

    if type(inputs) == np.ndarray:
        if dt is None:
            raise RuntimeError("dt is not provided when array input is provided")
        array_input_obj = ArrayInput(inputs, oob_value, dt)
        exc_input_func = array_input_obj.array_input
    else:
        exc_input_func = partial(callable_input, inputs, oob_value)
        
    with autoencoder_network as model:

        
        srf_network = PRF(dimensions = dimensions,
                          exc_input_func = exc_input_func,
                          connect_exc_inh_input = True,
                          connect_out_out = True,
                          n_excitatory = n_exc,
                          n_inhibitory = n_inh,
                          n_outputs = n_outputs,

                          output_coordinates = srf_output_coords,
                          exc_coordinates = srf_exc_coords,
                          inh_coordinates = srf_inh_coords,
                          
                          w_initial_E = params['w_initial_E'],
                          w_initial_I = params['w_initial_I'],
                          w_initial_EI = params['w_initial_EI'],
                          w_initial_EE = params['w_initial_EE'],
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
                          learning_rate_EE=params.get('learning_rate_EE', 1e-5),
                          sigma_scale_E = 0.005,
                          isp_target_rate = 2.0,
                          direct_input = direct_input,
                          dt=dt,
                          label="Spatial receptive field network",
                          seed=seed)


        if input_encoders is not None:
            srf_network.exc.encoders = input_encoders
        
        
        decoder = nengo.Ensemble(n_exc, dimensions=dimensions,
                                 neuron_type = nengo.LIF(),
                                 radius = 1,
                                 intercepts=nengo.dists.Choice([0.1]),
                                 max_rates=nengo.dists.Choice([40]))

        decoder_inh =  nengo.Ensemble(n_inh_decoder, dimensions=dimensions,
                                      neuron_type = nengo.LIF(tau_rc=0.005),
                                      radius = 1,
                                      intercepts=nengo.dists.Choice([0.1]),                                                 
                                      max_rates=nengo.dists.Choice([100]))

        tau_E = params['tau_E']
        w_DEC_E = params['w_DEC_E']
        p_DEC = params['p_DEC']
        model.weights_initial_DEC_E = local_random.uniform(size=n_outputs*n_exc).reshape((n_exc, n_outputs)) * w_DEC_E
        for i in range(n_exc):
            dist = cdist(decoder_coords[i,:].reshape((1,-1)), srf_network.output_coordinates).flatten()
            sigma = 0.1
            prob = distance_probs(dist, sigma)    
            sources = np.asarray(local_random.choice(n_outputs, round(p_DEC * n_outputs), replace=False, p=prob), dtype=np.int32)
            model.weights_initial_DEC_E[i, np.logical_not(np.in1d(range(n_outputs), sources))] = 0.

        model.conn_DEC_E = nengo.Connection(srf_network.output.neurons,
                                            decoder.neurons,
                                            transform=model.weights_initial_DEC_E,
                                            synapse=nengo.Alpha(tau_E),
                                            learning_rule_type=HSP(learning_rate=params['learning_rate_D_Exc'], directed=True))

                
        w_DEC_I_ff = params['w_DEC_I']
        weights_initial_DEC_I_ff = local_random.uniform(size=n_inh*n_outputs).reshape((n_inh, n_outputs)) * w_DEC_I_ff
        for i in range(n_inh_decoder):
            dist = cdist(decoder_inh_coords[i,:].reshape((1,-1)), srf_network.output_coordinates).flatten()
            sigma = 1.0
            prob = distance_probs(dist, sigma)    
            sources = np.asarray(local_random.choice(n_outputs, round(p_DEC * n_outputs), replace=False, p=prob), dtype=np.int32)
            weights_initial_DEC_I_ff[i, np.logical_not(np.in1d(range(n_outputs), sources))] = 0.
        conn_DEC_I_ff = nengo.Connection(srf_network.output.neurons,
                                         decoder_inh.neurons,
                                         transform=weights_initial_DEC_I_ff,
                                         synapse=nengo.Alpha(tau_E))
    
        w_DEC_I_fb = params['w_initial_I_DEC_fb']
        weights_initial_DEC_I_fb = local_random.uniform(size=n_inh_decoder*n_exc).reshape((n_exc, n_inh_decoder)) * w_DEC_I_fb
        for i in range(n_exc):
            dist = cdist(decoder_coords[i,:].reshape((1,-1)), decoder_inh_coords).flatten()
            sigma = 1.0
            prob = distance_probs(dist, sigma)
            sources = np.asarray(local_random.choice(n_inh_decoder, round(0.2 * n_inh_decoder), replace=False, p=prob), dtype=np.int32)
            weights_initial_DEC_I_fb[i, np.logical_not(np.in1d(range(n_inh_decoder), sources))] = 0.

        conn_DEC_I_fb = nengo.Connection(decoder_inh.neurons,
                                         decoder.neurons,
                                         transform=weights_initial_DEC_I_fb,
                                         synapse=nengo.Lowpass(params['tau_I']),
                                         learning_rule_type=CDISP(learning_rate=params['learning_rate_D']))


        w_SRF_I_bp = params['w_initial_I']
        weights_initial_SRF_I_bp = local_random.uniform(size=n_inh_decoder*n_outputs).reshape((n_outputs, n_inh_decoder)) * w_SRF_I_bp
        for i in range(n_outputs):
            dist = cdist(srf_network.output_coordinates[i,:].reshape((1,-1)), decoder_inh_coords).flatten()
            sigma = 0.1
            prob = distance_probs(dist, sigma)    
            sources = np.asarray(local_random.choice(n_inh_decoder, round(0.05 * n_inh_decoder), replace=False, p=prob), dtype=np.int32)
            weights_initial_SRF_I_bp[i, np.logical_not(np.in1d(range(n_inh_decoder), sources))] = 0.
        conn_SRF_I_bp = nengo.Connection(decoder_inh.neurons,
                                         srf_network.output.neurons,
                                         transform=weights_initial_SRF_I_bp,
                                         synapse=nengo.Alpha(params['tau_I']))
        
        coincidence_detection = nengo.Node(size_in=2*n_inputs, size_out=n_inputs,
                                           output=lambda t,x: np.subtract(x[:n_inputs], x[n_inputs:]))
        nengo.Connection(coincidence_detection, conn_DEC_I_fb.learning_rule)
        nengo.Connection(srf_network.exc.neurons, coincidence_detection[n_inputs:])
        nengo.Connection(decoder.neurons, coincidence_detection[:n_inputs])

        p_srf_rec_weights = None
        with srf_network:
            p_srf_output_spikes = nengo.Probe(srf_network.output.neurons, 'output', synapse=None)
            p_srf_exc_spikes = nengo.Probe(srf_network.exc.neurons, 'output')
            p_srf_inh_spikes = nengo.Probe(srf_network.inh.neurons, 'output')
            #p_srf_inh_weights = nengo.Probe(srf_network.conn_I, 'weights')
            p_srf_exc_weights = nengo.Probe(srf_network.conn_E, 'weights', sample_every=1.0)
            if srf_network.conn_EE is not None:
                p_srf_rec_weights = nengo.Probe(srf_network.conn_EE, 'weights', sample_every=1.0)
                
        p_decoder_spikes = nengo.Probe(decoder.neurons, synapse=None)
        p_decoder_inh_spikes = nengo.Probe(decoder_inh.neurons, synapse=None)

        p_decoder_weights = nengo.Probe(model.conn_DEC_E, 'weights', sample_every=1.0)

        model.srf_network = srf_network
        model.decoder_ens = decoder
        model.decoder_inh_ens = decoder_inh
        
    return { 'network': autoencoder_network,
             'neuron_probes': {'srf_output_spikes': p_srf_output_spikes,
                               'srf_exc_spikes': p_srf_exc_spikes,
                               'srf_inh_spikes': p_srf_inh_spikes,
                               'decoder_spikes': p_decoder_spikes,
                               'decoder_inh_spikes': p_decoder_inh_spikes,
             },
             'weight_probes': { 'srf_exc_weights': p_srf_exc_weights,
                                'srf_rec_weights': p_srf_rec_weights,
                                'decoder_weights': p_decoder_weights,
#                               'srf_inh_weights': p_srf_inh_weights,
#                               'srf_exc_weights': p_srf_exc_weights,
                                }
    }


def run(model_dict, t_end, dt=0.001, save_results=False):
        
    with nengo.Simulator(model_dict['network'], optimize=True, dt=dt,
                         progress_bar=TerminalProgressBar()) as sim:
        sim.run(np.max(t_end))

    p_srf_output_spikes = model_dict['neuron_probes']['srf_output_spikes']
    p_srf_exc_spikes = model_dict['neuron_probes']['srf_exc_spikes']
    p_srf_inh_spikes = model_dict['neuron_probes']['srf_inh_spikes']
    p_decoder_spikes = model_dict['neuron_probes']['decoder_spikes']
    p_decoder_inh_spikes = model_dict['neuron_probes']['decoder_inh_spikes']
    p_srf_exc_weights = model_dict['weight_probes']['srf_exc_weights']
    p_srf_rec_weights = model_dict['weight_probes']['srf_rec_weights']
    p_decoder_weights = model_dict['weight_probes']['decoder_weights']

    srf_rec_weights = sim.data.get(p_srf_rec_weights, None)
    srf_exc_weights = sim.data[p_srf_exc_weights]
    srf_output_spikes = sim.data[p_srf_output_spikes]
    decoder_spikes = sim.data[p_decoder_spikes]
    decoder_inh_spikes = sim.data[p_decoder_inh_spikes]
    decoder_weights = sim.data[p_decoder_weights]
    srf_exc_rates = rates_kernel(sim.trange(), sim.data[p_srf_exc_spikes], tau=0.2)
    srf_inh_rates = rates_kernel(sim.trange(), sim.data[p_srf_inh_spikes], tau=0.2)
    srf_output_rates = rates_kernel(sim.trange(), srf_output_spikes, tau=0.2)
    decoder_rates = rates_kernel(sim.trange(), decoder_spikes, tau=0.2)
    decoder_inh_rates = rates_kernel(sim.trange(), decoder_inh_spikes, tau=0.2)
    if save_results:
        np.save("srf_autoenc_exc_weights", np.asarray(srf_exc_weights, dtype=np.float32))
        np.save("srf_autoenc_exc_rates", np.asarray(srf_exc_rates, dtype=np.float32))
        np.save("srf_autoenc_inh_rates", np.asarray(srf_inh_rates, dtype=np.float32))
        np.save("srf_autoenc_output_spikes", np.asarray(srf_output_spikes, dtype=np.float32))
        np.save("srf_autoenc_decoder_spikes", np.asarray(decoder_spikes, dtype=np.float32))
        np.save("srf_autoenc_output_rates", np.asarray(srf_output_rates, dtype=np.float32))
        np.save("srf_autoenc_decoder_rates", np.asarray(decoder_rates, dtype=np.float32))
        np.save("srf_autoenc_decoder_inh_rates", np.asarray(decoder_inh_rates, dtype=np.float32))
        np.save("srf_autoenc_time_range", np.asarray(sim.trange(), dtype=np.float32))

    return sim, {'srf_autoenc_output_rates': srf_output_rates,
                 'srf_autoenc_decoder_rates': decoder_rates,
                 'srf_autoenc_decoder_inh_rates': decoder_inh_rates,
                 'srf_autoenc_exc_rates': srf_exc_rates,
                 'srf_autoenc_inh_rates': srf_inh_rates,
                 'srf_autoenc_output_spikes': srf_output_spikes,
                 'srf_autoenc_decoder_spikes': decoder_spikes,
                 'srf_autoenc_exc_weights': srf_exc_weights,
                 'srf_autoenc_rec_weights': srf_rec_weights,
                 'srf_autoenc_decoder_weights': decoder_weights,
            
    }
