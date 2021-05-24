
import numpy as np
import nengo
from hsp import HSP
from isp import ISP
from gdhl import GDHL
import nengo_extras
import nengo_extras.neurons
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cdist

def distance_probs(dist, sigma):
    weights = np.exp(-dist/sigma**2)
    prob = weights / weights.sum(axis=0)
    return prob

## Plastic excitatory-inhibitory network

class EI(nengo.Network):
    def __init__(self,
                 exc_input_func = None,
                 inh_input_func = None,
                 dimensions = 1,
                 n_inhibitory = 250,
                 n_excitatory = 1000,
                 w_initial_I = -1e-2, # baseline inhibitory synaptic weight
                 w_initial_E =  1e-1, # baseline excitatory synaptic weight
                 w_initial_EI =  1e-3, # baseline feedback inhibition synaptic weight
                 w_initial_EE =  1e-3, # baseline recurrent excitatory synaptic weight
                 w_II = -1e-3, # weight of inhibitory to inhibitory connections (when connect_inh_inh_input = True)
                 p_E = 0.4, # uniform probability of connections to excitatory cells
                 p_EI = 0.4, # uniform probability of feedback connections to inhibitory cells
                 p_EE = 0.1, # uniform probability of recurrent connections
                 p_II = 0.05, # uniform probability of inhibitory to inhibitory inputs (when connect_inh_inh = True)
                 tau_I = 0.03, # filter for inhibitory inputs
                 tau_E = 0.01, # filter for excitatory inputs
                 tau_EI = 0.01, # filter for feedback inhibitory connections
                 tau_EE = 0.01, # filter for recurrent connections
                 tau_input = 0.005, # filter for node input
                 learning_rate_I = 1e-6, # learning rate for homeostatic inhibitory plasticity
                 learning_rate_E = 1e-5, # learning rate for associative excitatory plasticity
                 learning_rate_EE = 1e-5, # learning rate for recurrent excitatory plasticity
                 isp_target_rate = 2.0, # target firing rate for inhibitory plasticity
                 connect_exc_inh = False,
                 connect_inh_inh = False,
                 use_gdhl = False,
                 gdhl_sigma = { 'pp': 0.1, 'np': -0.1, 'pn': -0.1, 'nn': 0.1 },
                 gdhl_eta = { 'ps': 0.0, 'ns': 0.0, 'sp': 0.0, 'sn': 0.0 },
                 label = None,
                 seed = 0,
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

        if exc_coordinates is None:
            self.exc_coordinates = np.asarray(range(n_excitatory)).reshape((n_excitatory,1)) / n_excitatory
        if inh_coordinates is None:
            self.inh_coordinates = np.asarray(range(n_inhibitory)).reshape((n_inhibitory,1)) / n_inhibitory
        
        rng = np.random.RandomState(seed=seed)

        assert(w_initial_I < 0)
        assert(w_initial_E > 0)
        assert(w_initial_EI > 0)
        assert(w_initial_EE > 0)

        if weights_I is not None:
            weights_initial_I = weights_I
        else:
            weights_initial_I = rng.uniform(size=n_inhibitory*n_excitatory).reshape((n_excitatory, n_inhibitory)) * w_initial_I

        
        if weights_E is not None:
            weights_initial_E = weights_E
        else:
            weights_initial_E = np.zeros((n_excitatory, n_excitatory))
            for i in range(n_excitatory):
                dist = cdist(self.exc_coordinates[i,:].reshape((1,-1)), self.exc_coordinates).flatten()
                sigma = p_E * n_excitatory
                prob = distance_probs(dist, sigma)
                sources_Exc = np.asarray(rng.choice(n_excitatory, round(p_E * n_excitatory), replace=False, p=prob), dtype=np.int32)
                weights_initial_E[i, np.logical_not(np.in1d(range(n_excitatory), sources_Exc))] = 0.
                w = rng.normal(size=len(sources_Exc))
                weights_initial_E[i, sources_Exc] = (w - w.min()) / (w.max() - w.min()) * w_initial_E
                
        weights_dist_EI = rng.uniform(size=n_excitatory*n_inhibitory).reshape((n_inhibitory, n_excitatory))
        weights_initial_EI = (weights_dist_EI - weights_dist_EI.min()) / (weights_dist_EI.max() - weights_dist_EI.min()) * w_initial_EI
        for i in range(n_excitatory):
            targets_Inh = np.asarray(rng.choice(n_inhibitory, round(p_EI * n_inhibitory), replace=False), dtype=np.int32)
            weights_initial_EI[np.logical_not(np.in1d(range(n_inhibitory), targets_Inh)), i] = 0.
            
        if weights_EE is not None:
            weights_initial_EE = weights_EE
        else:
            weights_dist_EE = rng.normal(size=n_excitatory*n_excitatory).reshape((n_excitatory, n_excitatory))
            weights_initial_EE = (weights_dist_EE - weights_dist_EE.min()) / (weights_dist_EE.max() - weights_dist_EE.min()) * w_initial_EE
            for i in range(n_excitatory):
                target_choices = np.asarray([ j for j in range(n_excitatory) if i != j ])
                dist = cdist(self.output_coordinates[i,:].reshape((1,-1)), self.output_coordinates[target_choices]).flatten()
                sigma = p_EE * n_excitatory
                prob = distance_probs(dist, sigma)
                targets_Out = np.asarray(rng.choice(target_choices, round(p_EE * n_excitatory), replace=False, p=prob),
                                        dtype=np.int32)
                weights_initial_EE[i, np.logical_not(np.in1d(range(n_excitatory), targets_Out))] = 0.

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

            if self.exc_input is not None:
                nengo.Connection(self.exc_input, self.exc.neurons,
                                synapse=nengo.Alpha(tau_input),
                                transform=weights_initial_E)
            
            if self.inh_input is not None:
                nengo.Connection(self.inh_input, self.inh.neurons,
                                synapse=nengo.Alpha(tau_input),
                                transform=np.eye(n_inhibitory))

                
            
            self.conn_I = nengo.Connection(self.inh.neurons,
                                           self.exc.neurons,
                                           transform=weights_initial_I,
                                           synapse=nengo.Alpha(tau_I),
                                           learning_rule_type=ISP(learning_rate=learning_rate_I,
                                                                  rho0=isp_target_rate))

            self.conn_II = None
            if connect_inh_inh:
                weights_dist_II = np.ones((n_inhibitory, n_inhibitory)) * w_II
                for i in range(n_inhibitory):
                    sources_Inh = np.asarray(rng.choice(n_inhibitory, round(p_II * n_inhibitory), replace=False), dtype=np.int32)
                    weights_dist_II[i, np.logical_not(np.in1d(range(n_inhibitory), sources_Inh))] = 0.
                    self.conn_II = nengo.Connection(self.inh.neurons,
                                                    self.inh.neurons,
                                                    transform=weights_dist_II,
                                                    synapse=nengo.Alpha(tau_I))
                                                                                                                                    

            self.conn_EI = None
            if connect_exc_inh:
                self.conn_EI = nengo.Connection(self.exc.neurons,
                                                self.inh.neurons,
                                                transform=weights_initial_EI,
                                                synapse=nengo.Alpha(tau_EI))
            
            if self.n_excitatory > 1:
                self.conn_EE = nengo.Connection(self.exc.neurons, 
                                                self.exc.neurons, 
                                                transform=weights_initial_EE,
                                                synapse=nengo.Alpha(tau_EE),
                                                learning_rule_type=HSP(learning_rate=learning_rate_EE))
            else:
                self.conn_EE = None
                        
                
                             
    
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
    def exc_ens_config(self):
        """(Config) Defaults for excitatory input ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update(
            {
            "neuron_type": nengo.LIF(tau_rc=0.005),
            "radius": 1,
            "intercepts": nengo.dists.Choice([0.1]*self.dimensions),
            "max_rates": nengo.dists.Choice([40])
            }
            )
        cfg[nengo.Connection].synapse = None
        return cfg
