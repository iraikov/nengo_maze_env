
import numpy as np
import nengo
from hsp import HSP
from isp import ISP
from gdhl import GDHL
import nengo_extras
import nengo_extras.neurons

## Plastic receptive fields network

class PRF(nengo.Network):
    def __init__(self,
                 exc_input_func = None,
                 inh_input_func = None,
                 dimensions = 1,
                 n_outputs = 50,
                 n_inhibitory = 250,
                 n_excitatory = 1000,
                 w_initial_I = -1e-2, # baseline inhibitory synaptic weight
                 w_initial_E =  1e-1, # baseline excitatory synaptic weight
                 w_initial_EI =  1e-3, # baseline feedback inhibition synaptic weight
                 w_initial_EE =  1e-3, # baseline recurrent excitatory synaptic weight
                 w_EI_Ext = 1e-3, # weight of excitatory connection to inhibitory inputs (when connect_exc_inh_input = True)
                 p_E = 0.4, # uniform probability of connection of excitatory inputs to outputs
                 p_EI = 0.4, # uniform probability of feedback connections to inhibitory cells
                 p_EE = 0.1, # uniform probability of recurrent connections
                 p_EI_Ext = 0.25, # uniform probability of excitatory connection to inhibitory inputs (when connect_exc_inh_input = True)
                 tau_I = 0.03, # filter for inhibitory inputs
                 tau_E = 0.01, # filter for excitatory inputs
                 tau_EI = 0.01, # filter for feedback inhibitory connections
                 tau_EE = 0.01, # filter for recurrent connections
                 tau_input = 0.005, # filter for node input
                 learning_rate_I = 1e-6, # learning rate for homeostatic inhibitory plasticity
                 learning_rate_E = 1e-5, # learning rate for associative excitatory plasticity
                 learning_rate_EE = 1e-5, # learning rate for recurrent excitatory plasticity
                 isp_target_rate = 2.0, # target firing rate for inhibitory plasticity
                 connect_exc_inh_input = False,
                 use_gdhl = False,
                 gdhl_sigma = { 'pp': 0.1, 'np': -0.1, 'pn': -0.1, 'nn': 0.1 },
                 gdhl_eta = { 'ps': 0.0, 'ns': 0.0, 'sp': 0.0, 'sn': 0.0 },
                 label = None,
                 seed = 0,
                 add_to_container = None,
                 **kwds
                 ):
        super().__init__(label, seed, add_to_container)
        
        self.dimensions = dimensions
        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.n_outputs = n_outputs
        
        rng = np.random.RandomState(seed=seed)

        assert(w_initial_I < 0)
        assert(w_initial_E > 0)
        assert(w_initial_EI > 0)
        assert(w_initial_EE > 0)

        weights_dist_I = rng.normal(size=n_inhibitory*n_outputs).reshape((n_outputs, n_inhibitory))
        weights_initial_I = (weights_dist_I - weights_dist_I.min()) / (weights_dist_I.max() - weights_dist_I.min()) * w_initial_I

        weights_dist_E = rng.normal(size=n_excitatory*n_outputs).reshape((n_outputs, n_excitatory))
        weights_initial_E = (weights_dist_E - weights_dist_E.min()) / (weights_dist_E.max() - weights_dist_E.min()) * w_initial_E
        for i in range(n_outputs):
            sources_Exc = np.asarray(rng.choice(n_excitatory, round(p_E * n_excitatory), replace=False), dtype=np.int32)
            weights_initial_E[i, np.logical_not(np.in1d(range(n_excitatory), sources_Exc))] = 0.

        weights_dist_EI = rng.normal(size=n_outputs*n_inhibitory).reshape((n_inhibitory, n_outputs))
        weights_initial_EI = (weights_dist_EI - weights_dist_EI.min()) / (weights_dist_EI.max() - weights_dist_EI.min()) * w_initial_EI
        for i in range(n_outputs):
            targets_Inh = np.asarray(rng.choice(n_inhibitory, round(p_EI * n_inhibitory), replace=False), dtype=np.int32)
            weights_initial_EI[np.logical_not(np.in1d(range(n_inhibitory), targets_Inh)), i] = 0.
            
        weights_dist_EE = rng.normal(size=n_outputs*n_outputs).reshape((n_outputs, n_outputs))
        weights_initial_EE = (weights_dist_EE - weights_dist_EE.min()) / (weights_dist_EE.max() - weights_dist_EE.min()) * w_initial_EE
        for i in range(n_outputs):
            target_choices = np.asarray([ j for j in range(n_outputs) if i != j ])
            targets_Out = np.asarray(rng.choice(target_choices, round(p_EE * n_outputs), replace=False),
                                     dtype=np.int32)
            weights_initial_EE[i, np.logical_not(np.in1d(range(n_outputs), targets_Out))] = 0.

        with self:

            self.exc_input = None
            self.inh_input = None
            if exc_input_func is not None:
                self.exc_input = nengo.Node(output=exc_input_func, size_out=n_excitatory)
            if inh_input_func is not None:
                self.inh_input = nengo.Node(output=inh_input_func, size_out=n_inhibitory)
                
            with self.exc_ens_config:

                self.exc = nengo.Ensemble(self.n_excitatory, dimensions=self.dimensions)

            with self.inh_ens_config:
            
                self.inh = nengo.Ensemble(self.n_inhibitory, dimensions=self.dimensions)
            
            with self.out_ens_config:
                self.output = nengo.Ensemble(self.n_outputs, dimensions=self.dimensions)

            if self.exc_input is not None:
                nengo.Connection(self.exc_input, self.exc.neurons,
                                synapse=nengo.Alpha(tau_input),
                                transform=np.eye(n_excitatory))
            
            if self.inh_input is not None:
                nengo.Connection(self.inh_input, self.inh.neurons,
                                synapse=nengo.Alpha(tau_input),
                                transform=np.eye(n_inhibitory))

            if connect_exc_inh_input and (self.exc_input is not None):
                weights_dist_EI_Ext = rng.uniform(size=n_excitatory*n_inhibitory).reshape((n_inhibitory, n_excitatory)) * w_EI_Ext
                for i in range(n_inhibitory):
                    sources_Exc = np.asarray(rng.choice(n_excitatory, round(p_EI_Ext * n_excitatory), replace=False), dtype=np.int32)
                    weights_dist_EI_Ext[i, np.logical_not(np.in1d(range(n_excitatory), sources_Exc))] = 0.
                nengo.Connection(self.exc.neurons, self.inh.neurons,
                                synapse=nengo.Lowpass(0.01),
                                transform=weights_dist_EI_Ext)
                
            
            self.conn_I = nengo.Connection(self.inh.neurons,
                                           self.output.neurons,
                                           transform=weights_initial_I,
                                           synapse=nengo.Alpha(tau_I),
                                           learning_rule_type=ISP(learning_rate=learning_rate_I,
                                                                  rho0=isp_target_rate))
            
            self.conn_E = nengo.Connection(self.exc.neurons,
                                           self.output.neurons, 
                                           transform=weights_initial_E,
                                           synapse=nengo.Alpha(tau_E),
                                           learning_rule_type=GDHL(sigma=gdhl_sigma, eta=gdhl_eta,
                                                                   learning_rate=learning_rate_E,
                                                                   pre_synapse=nengo.Lowpass(0.1),
                                                                   post_synapse=nengo.Lowpass(0.1),
                                                                   )
                                               if use_gdhl else HSP(learning_rate=learning_rate_E))

                
            self.conn_EI = nengo.Connection(self.output.neurons,
                                            self.inh.neurons,
                                            transform=weights_initial_EI,
                                            synapse=nengo.Alpha(tau_EI))
            

            self.conn_EE = nengo.Connection(self.output.neurons, 
                                            self.output.neurons, 
                                            transform=weights_initial_EE,
                                            synapse=nengo.Alpha(tau_EE),
                                            learning_rule_type=HSP(learning_rate=learning_rate_EE))

                             
    @property
    def exc_ens_config(self):
        """(Config) Defaults for excitatory input ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update(
            {
            "neuron_type": nengo.RectifiedLinear(),
            "radius": 1,
            "intercepts": nengo.dists.Choice([0.1]*self.dimensions),
            "max_rates": nengo.dists.Choice([20])
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
            "neuron_type": nengo.RectifiedLinear(),
            "radius": 1,
            "intercepts": nengo.dists.Choice([0.1]*self.dimensions),
            "max_rates": nengo.dists.Choice([40])
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
            "neuron_type": nengo.LIF(),
            "radius": 1,
            "intercepts": nengo.dists.Choice([0.1]*self.dimensions),
            "max_rates": nengo.dists.Choice([40])
            }
            )
        cfg[nengo.Connection].synapse = None
        return cfg
