import sys
import numpy as np
import nengo
from learning.hsp import HSP
from learning.stdp import RdSTDP, HvSTDP
from learning.isp import ISP
#from gdhl import GDHL
import nengo_extras
import nengo_extras.neurons
import scipy
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cdist

def indep_roll(arr, shifts, axis=1):
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted. 
    """
    arr = np.swapaxes(arr,axis,-1)
    all_idcs = np.ogrid[[slice(0,n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1] 
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result,-1,axis)
    return arr


# Delayed connection example from
# https://www.nengo.ai/nengo/v3.1.0/examples/usage/delay-node.html
class Delay:
    def __init__(self, dimensions, shifts):
        self.history = np.zeros((np.max(shifts), dimensions))

    def step(self, t, x):
        self.history = np.indep_roll(self.history, -shifts)
        self.history[-1] = x
        return self.history[0]


    
    
def distance_probs(dist, sigma):
    weights = np.exp(-dist/sigma**2)
    prob = weights / weights.sum(axis=0)
    return prob
    

def convergent_topo_transform(rng, n_pre, n_post, coords_pre, coords_post, p_initial, w_initial, sigma_scale, exclude_self=False):
    transform = np.zeros((n_post, n_pre))
    for i in range(n_post):
        dist = cdist(coords_post[i,:].reshape((1,-1)), coords_pre).flatten()
        sigma = sigma_scale * p_initial * n_pre
        prob = distance_probs(dist, sigma)
        if exclude_self:
            source_choices = np.asarray([ j for j in range(n_pre) if i != j ])
        else:
            source_choices = np.asarray(range(n_pre))
        sources = np.asarray(rng.choice(source_choices, round(p_initial * n_pre), replace=False, p=prob),
                             dtype=np.int32)
        w = np.clip(rng.lognormal(size=len(sources), sigma=1.0), 1e-4, None)
        w /= w.max()
        transform[i, np.in1d(range(n_pre), sources)] = w
    return np.clip(transform, 0., None) * w_initial

def divergent_topo_transform(rng, n_pre, n_post, coords_pre, coords_post, p_initial, w_initial, sigma_scale, exclude_self=False):
    transform = np.zeros((n_post, n_pre))
    for i in range(n_pre):
        if exclude_self:
            target_choices = np.asarray([ j for j in range(n_post) if i != j ])
        else:
            target_choices = np.asarray(range(n_post))
        dist = cdist(coords_pre[i,:].reshape((1,-1)), coords_post[target_choices,:]).flatten()
        sigma = sigma_scale * p_initial * n_post
        prob = distance_probs(dist, sigma)
        targets = np.asarray(rng.choice(target_choices, round(p_initial * n_post), replace=False, p=prob),
                             dtype=np.int32)
        #w = np.clip(rng.lognormal(size=len(targets), sigma=1.0), 1e-4, None)
        w = np.clip(rng.uniform(size=len(targets)), 1e-4, None)
        w /= w.max()
        transform[targets, i] = w
    return np.clip(transform, 0., None) * w_initial

## Plastic receptive fields network

class PRF(nengo.Network):
    def __init__(self,
                 exc_input = None,
                 inh_input = None,
                 exc_input_process = None,
                 inh_input_process = None,
                 dimensions = 1,
                 n_outputs = 50,
                 n_inhibitory = 250,
                 n_excitatory = 1000,
                 w_initial_I = -1e-2, # baseline inhibitory synaptic weight
                 w_initial_E =  1e-1, # baseline excitatory synaptic weight
                 w_initial_EI =  1e-3, # baseline feedback inhibition synaptic weight
                 w_initial_EE =  1e-3, # baseline recurrent excitatory synaptic weight
                 w_initial_E_Fb =  1e-3, # baseline output to excitatory synaptic weight (when connect_exc_fb = True)
                 w_II = -1e-3, # weight of inhibitory to inhibitory connections (when connect_inh_inh_input = True)
                 w_EI_Ext = 1e-3, # weight of excitatory connection to inhibitory inputs (when connect_exc_inh_input = True)
                 w_input = 1,
                 p_E = 0.3, # uniform probability of connection of excitatory inputs to outputs
                 p_I = 0.3,
                 p_EI = 0.2, # uniform probability of feedback connections to inhibitory cells
                 p_EE = 0.05, # uniform probability of recurrent connections
                 p_EI_Ext = 0.25, # uniform probability of excitatory connection to inhibitory inputs (when connect_exc_inh_input = True)
                 p_E_Fb = 0.05, # uniform probability of outputs to excitatory inputs (when connect_exc_fb = True)
                 p_II = 0.05, # uniform probability of inhibitory to inhibitory inputs (when connect_inh_inh = True)
                 tau_I = 0.03, # filter for inhibitory inputs
                 tau_E = 0.01, # filter for excitatory inputs
                 tau_EI = 0.01, # filter for feedback inhibitory connections
                 tau_EE = 0.01, # filter for recurrent connections
                 tau_II = 0.01, # filter for recurrent inhibitory connections
                 tau_EI_Ext = 0.01, # filter for excitatory connection to inhibitory inputs (when connect_exc_inh_input = True)
                 tau_E_Fb = 0.01, # filter for output connection to excitatory inputs (when connect_exc_fb  = True)
                 tau_input = 0.005, # filter for node input
                 learning_rate_I = 1e-6, # learning rate for homeostatic inhibitory plasticity
                 learning_rate_E = 1e-5, # learning rate for associative excitatory plasticity
                 learning_rate_EE = 1e-5, # learning rate for recurrent excitatory plasticity
                 isp_target_rate_I = 2.0, # target firing rate for inhibitory plasticity
                 isp_target_rate_II = 20.0, # target firing rate for inhibitory plasticity
                 learning_rate_E_func = None, # specify variable learning rate
                 learning_rate_EE_func = None, # specify variable learning rate
                 learning_rate_I_func = None, # specify variable learning rate
                 connect_exc_inh_input = False,
                 connect_inh_inh = False,
                 connect_exc_fb = False,
                 connect_out_out = True,
                 use_stdp = False,
                 exc_coordinates = None,
                 inh_coordinates = None,
                 output_coordinates = None,
                 sigma_scale_E = 0.01,
                 sigma_scale_EE = 0.1,
                 sigma_scale_EI = 0.1,
                 sigma_scale_EI_Ext = 0.1,
                 sigma_scale_E_Fb = 0.1,
                 sigma_scale_I = 0.1,
                 syn_class=nengo.Alpha,
                 label = None,
                 seed = 0,
                 direct_input = True,
                 add_to_container = None,
                 weights_I = None,
                 weights_E = None,
                 weights_EE = None,
                 weights_E_Fb = None,
                 **kwds
                 ):
        super().__init__(label, seed, add_to_container)
        
        self.dimensions = dimensions
        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.n_outputs = n_outputs

        if exc_coordinates is None:
            self.exc_coordinates = np.asarray(range(n_excitatory)).reshape((n_excitatory,1)) / n_excitatory
        else:
            self.exc_coordinates = exc_coordinates
        if inh_coordinates is None:
            self.inh_coordinates = np.asarray(range(n_inhibitory)).reshape((n_inhibitory,1)) / n_inhibitory
        else:
            self.inh_coordinates = inh_coordinates
        if output_coordinates is None:
            self.output_coordinates = np.asarray(range(n_outputs)).reshape((n_outputs,1)) / n_outputs
        else:
            self.output_coordinates = output_coordinates
        
        rng = np.random.RandomState(seed=seed)

        assert(w_initial_I < 0)
        assert(w_initial_E > 0)
        assert(w_initial_EI > 0)
        if w_initial_EE is not None:
            assert(w_initial_EE > 0)

        if weights_I is not None:
            weights_initial_I = weights_I
        else:
            #weights_initial_I = rng.uniform(size=n_inhibitory*n_outputs).reshape((n_outputs, n_inhibitory)) * w_initial_I
            weights_initial_I = divergent_topo_transform(rng, n_inhibitory, n_outputs,
                                                         self.inh_coordinates, self.output_coordinates,
                                                         p_I, w_initial_I, sigma_scale_I)

        self.weights_initial_I = weights_initial_I

        if weights_E is not None:
            weights_initial_E = weights_E
        else:
            weights_initial_E = convergent_topo_transform(rng, n_excitatory, n_outputs,
                                                         self.exc_coordinates, self.output_coordinates,
                                                         p_E, w_initial_E, sigma_scale_E)
        self.weights_initial_E = weights_initial_E
                
        weights_initial_EI = convergent_topo_transform(rng, n_outputs, n_inhibitory,
                                                       self.output_coordinates, self.inh_coordinates, 
                                                       p_EI, w_initial_EI, sigma_scale_EI)
        self.weights_initial_EI = weights_initial_EI
            
        if weights_EE is not None:
            weights_initial_EE = weights_EE
        else:
            weights_initial_EE = divergent_topo_transform(rng, n_outputs, n_outputs,
                                                          self.output_coordinates, self.output_coordinates,
                                                          p_EE, w_initial_EE, sigma_scale_EE,
                                                          exclude_self=True) if w_initial_EE is not None else None
        self.weights_initial_EE = weights_initial_EE
        
        with self:

            self.exc_input = exc_input
            self.inh_input = inh_input
            if exc_input_process is not None:
                if direct_input:
                    self.exc_input = nengo.Node(output=exc_input_process, size_out=n_excitatory)
                else:
                    self.exc_input = nengo.Node(output=exc_input_process, size_out=self.dimensions)
            if inh_input_process is not None:
                self.inh_input = nengo.Node(output=inh_input_process, size_out=n_inhibitory)

            with self.exc_ens_config:
                    
                self.exc_ens = nengo.Ensemble(self.n_excitatory, dimensions=self.dimensions, normalize_encoders=False)

            with self.inh_ens_config:
            
                self.inh_ens = nengo.Ensemble(self.n_inhibitory, dimensions=self.dimensions)
            
            with self.out_ens_config:
                self.output = nengo.Ensemble(self.n_outputs, dimensions=self.dimensions)

            if self.exc_input is not None:
                if direct_input:
                    nengo.Connection(self.exc_input, self.exc_ens.neurons,
                                     synapse=syn_class(tau_input),
                                     transform=np.eye(n_excitatory) * w_input)
                else:
                    assert(self.exc_ens is not None)
                    nengo.Connection(self.exc_input, self.exc_ens,
                                     synapse=syn_class(tau_input))
                    
            
            if self.inh_input is not None:
                nengo.Connection(self.inh_input, self.inh_ens.neurons,
                                synapse=syn_class(tau_input),
                                transform=np.eye(n_inhibitory) * w_input)

            self.conn_EI_Ext = None
            if connect_exc_inh_input:
                weights_initial_EI_Ext = convergent_topo_transform(rng, n_excitatory, n_inhibitory,
                                                                   self.exc_coordinates, self.inh_coordinates, 
                                                                   p_EI_Ext, w_EI_Ext, sigma_scale_EI_Ext)
                self.conn_EI_Ext = nengo.Connection(self.exc_ens.neurons, self.inh_ens.neurons,
                                                    synapse=syn_class(tau_EI_Ext),
                                                    transform=weights_initial_EI_Ext)
                
            self.node_learning_rate_I = None
            self.conn_learning_rate_I = None
            if learning_rate_I_func is not None:
                self.node_learning_rate_I = nengo.Node(learning_rate_I_func)
            elif learning_rate_I is not None:
                self.node_learning_rate_I = nengo.Node(lambda t: learning_rate_I)
                
            self.conn_I = nengo.Connection(self.inh_ens.neurons,
                                           self.output.neurons,
                                           transform=weights_initial_I,
                                           synapse=syn_class(tau_I),
                                           learning_rule_type=ISP(rho0=isp_target_rate_I,
                                                                  pre_synapse=nengo.Lowpass(0.005)) if self.node_learning_rate_I is not None else None)

            if self.node_learning_rate_I is not None:
                self.conn_learning_rate_I = nengo.Connection(self.node_learning_rate_I, self.conn_I.learning_rule)

            self.conn_II = None
            if connect_inh_inh:
                weights_dist_II = np.ones((n_inhibitory, n_inhibitory)) * w_II
                for i in range(n_inhibitory):
                    sources_Inh = np.asarray(rng.choice(n_inhibitory, round(p_II * n_inhibitory), replace=False), dtype=np.int32)
                    weights_dist_II[i, np.logical_not(np.in1d(range(n_inhibitory), sources_Inh))] = 0.
                self.conn_II = nengo.Connection(self.inh_ens.neurons,
                                                self.inh_ens.neurons,
                                                transform=weights_dist_II,
                                                synapse=syn_class(tau_II),
                                                learning_rule_type=ISP(rho0=isp_target_rate_II,
                                                                       pre_synapse=nengo.Lowpass(0.005)) if self.node_learning_rate_I is not None else None)
                if self.node_learning_rate_I is not None:
                    self.conn_learning_rate_II = nengo.Connection(self.node_learning_rate_I, self.conn_II.learning_rule)
                
            self.node_learning_rate_E = None
            self.conn_learning_rate_E = None
            if learning_rate_E_func is not None:
                self.node_learning_rate_E = nengo.Node(learning_rate_E_func)
            elif learning_rate_E is not None:
                self.node_learning_rate_E = nengo.Node(lambda t: learning_rate_E)
                
            self.conn_E = nengo.Connection(self.exc_ens.neurons,
                                           self.output.neurons, 
                                           transform=weights_initial_E,
                                           synapse=syn_class(tau_E),
                                           learning_rule_type=(RdSTDP(r_tau=0.1, pre_tau=0.01, post_tau=0.015, pre_amp=0.3, post_amp=0.6) \
                                                               if use_stdp else HSP(pre_synapse=nengo.Lowpass(0.02), \
                                                                                    post_synapse=nengo.Lowpass(0.04), \
                                                                                    directed=True)) if self.node_learning_rate_E is not None else None)
            if self.node_learning_rate_E is not None:
               self.conn_learning_rate_E = nengo.Connection(self.node_learning_rate_E, self.conn_E.learning_rule)
                
                
            self.conn_EI = nengo.Connection(self.output.neurons,
                                            self.inh_ens.neurons,
                                            transform=weights_initial_EI,
                                            synapse=syn_class(tau_EI))

            self.conn_EE = None
            self.node_learning_rate_EE = None
            self.conn_learning_rate_EE = None
            if connect_out_out and (self.n_outputs > 1):
                if learning_rate_EE_func is not None:
                     self.node_learning_rate_EE = nengo.Node(learning_rate_EE_func)
                elif learning_rate_EE is not None:
                    self.node_learning_rate_EE = nengo.Node(lambda t: learning_rate_EE)
                self.conn_EE = nengo.Connection(self.output.neurons, 
                                                self.output.neurons, 
                                                transform=weights_initial_EE,
                                                synapse=syn_class(tau_EE),
                                                learning_rule_type=(RdSTDP(r_tau=0.1, pre_tau=0.01, post_tau=0.005, pre_amp=0.3, post_amp=0.9) \
                                                                    if use_stdp else HSP(pre_synapse=nengo.Lowpass(0.02),
                                                                                         post_synapse=nengo.Lowpass(0.04),
                                                                                         directed=True) if self.node_learning_rate_EE else None))
                if self.node_learning_rate_EE is not None:
                    self.conn_learning_rate_EE = nengo.Connection(self.node_learning_rate_EE, self.conn_EE.learning_rule)

            self.conn_E_Fb = None
            self.conn_learning_rate_E_Fb = None
            if connect_exc_fb:
                if weights_E_Fb is not None:
                    weights_initial_E_Fb = weights_E_Fb
                else:
                    weights_initial_E_Fb = rng.uniform(size=(n_excitatory*n_outputs)).reshape((n_excitatory, n_outputs))
                    for i in range(n_excitatory):
                        dist = cdist(self.exc_coordinates[i,:].reshape((1,-1)), self.output_coordinates).flatten()
                        sigma = sigma_scale_E_Fb * p_E_Fb * n_outputs
                        prob = distance_probs(dist, sigma)
                        sources_Out = np.asarray(rng.choice(n_outputs, round(p_E_Fb * n_outputs), replace=False, p=prob), dtype=np.int32)
                        weights_initial_E_Fb[i, np.logical_not(np.in1d(range(n_outputs), sources_Out))] = 0.
                self.conn_E_Fb = nengo.Connection(self.output.neurons,
                                                  self.exc_ens.neurons,
                                                  transform=weights_initial_E_Fb,
                                                  synapse=syn_class(tau_E_Fb),
                                                  learning_rule_type=RdSTDP(r_tau=0.1, pre_amp=0.3, post_amp=0.6)
                                                                     if use_stdp else HSP(pre_synapse=nengo.Lowpass(0.01)))
                self.conn_learning_rate_E_Fb = nengo.Connection(self.node_learning_rate_E, self.conn_E_Fb)


                        
                
                             
    @property
    def exc_ens_config(self):
        """(Config) Defaults for excitatory input ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update(
            {
                "neuron_type": nengo.LIF(tau_rc=0.01, tau_ref=0.00233, amplitude=0.1),
                "radius": 1,
                #"intercepts": nengo.dists.Choice([0.01]*self.dimensions),
                "intercepts": nengo.dists.Exponential(0.1, shift=0.01, high=1.0),
                "max_rates": nengo.dists.Uniform(40, 80)

            }
            )
        cfg[nengo.Connection].synapse = None
        return cfg
    
    @property
    def inh_ens_config(self):
        """(Config) Defaults for inhibitory input ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update(
            {
                "neuron_type": nengo.LIF(tau_rc=0.01, tau_ref=0.002),
                "radius": 1,
                "intercepts": nengo.dists.Choice([0.01]*self.dimensions),
                "max_rates": nengo.dists.Choice([150])
            }
            )
        cfg[nengo.Connection].synapse = None
        return cfg
    
    @property
    def out_ens_config(self):
        """(Config) Defaults for excitatory input ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update(
            {
                "neuron_type": nengo.AdaptiveLIF(tau_rc=0.03, tau_ref=0.002, tau_n=1, inc_n=10.0, amplitude=0.1),
                "intercepts": nengo.dists.Exponential(0.1, shift=0.01, high=1.0),
                "radius": 1,
                "max_rates": nengo.dists.Uniform(40, 100)
            }
            )
        cfg[nengo.Connection].synapse = None
        return cfg

