import numpy as np
import scipy
from scipy.stats import skewnorm
import nengo
from nengolib import RLS, Alpha
from ws import WeightSaver

# xC are the neurons
# sC are the unfiltered currents into each neuron (sC -> Lowpass(tau) -> xC)
# zC is the learned output estimate, decoded by the neurons, and re-encoded back into sC alongside some random feedback (JD)
# eC is a gated error signal for RLS that turns off after T_train seconds. This error signal learns the feedback decoders by minmizing the difference between z (ideal output) and zC (actual output).
# The error signal driving RLS has an additional filter applied (tau_learn) to handle the case when this signal consists of spikes (not rates).

class NengoReservoir(nengo.Network):
    def __init__(self, n_per_dim=1000,
                 n_eval_points = 5000,
                 dimensions = 1,
                 tau = 0.05,       # filter for output
                 tau_input = 0.05, # filter for input
                 tau_learn = 0.05, # filter for error
                 learning_rate = 1e-6,
                 g_exc = 1.5 * 1e-5, # baseline excitatory synaptic weight
                 g_inh = 4 * 1e-5, # baseline inhibitory synaptic weight
                 ie = 0.1, # fraction of inhibitory to excitatory units
                 g_in = 1.0,      # scale the input encoders (usually 1)
                 g_out = 1.0,  # scale the recurrent encoders (usually 1)
                 skewnorm_a_exc = 4.0, # parameter for skew normal synaptic weight distribution
                 skewnorm_a_inh = 4.0, # parameter for skew normal synaptic weight distribution
                 scale_exc = 1.0, # scale parameter for skew normal synaptic weight distribution
                 scale_inh = 1.0, # scale parameter for skew normal synaptic weight distribution
                 label = None,
                 seed = 0,
                 add_to_container = None,
                 weights_path = 'reservoir_weights',
                 **kwds
                 ):
        super().__init__(label, seed, add_to_container)
        self.dimensions = dimensions
        self.n_eval_points = n_eval_points
        self.tau = tau
        self.tau_input = tau_input
        self.tau_learn = tau_learn
        self.learning_rate = learning_rate
        self.g_exc = g_exc
        self.g_inh = g_inh
        self.g_in = g_in
        self.g_out = g_out
        assert ((ie < 1.0) and (ie >= 0.0))
        self.ie = ie
        
        rng = np.random.RandomState(seed=seed)

        n_neurons = n_per_dim*dimensions
        n_inhibitory = int(self.ie * n_neurons)
        n_excitatory = n_neurons - n_inhibitory

        self.n_inhibitory = n_inhibitory
        self.n_excitatory = n_excitatory
        
        # fixed encoders for f_in (u_in)
        self.e_in = g_in * rng.uniform(-1, +1, (n_neurons, dimensions))  
        # fixed encoders for f_out (u)
        self.e_out = g_out * rng.uniform(-1, +1, (n_neurons, dimensions))
        
        # target-generating weights (variance g^2/n)
        WI = skewnorm.rvs(-skewnorm_a_inh, loc=-1, scale=scale_inh, size=n_inhibitory*n_neurons, random_state=rng).reshape(n_inhibitory, n_neurons)
        JI = np.clip(WI * g_inh / np.sqrt(n_neurons), None, 0.)
        WE = skewnorm.rvs(skewnorm_a_exc, loc=1, scale=scale_exc, size=n_excitatory*n_neurons, random_state=rng).reshape(n_excitatory, n_neurons)
        JE = np.clip(WE * g_exc / np.sqrt(n_neurons), 0., None)
        self.JD = np.zeros((n_neurons, n_neurons))
        self.JD[:n_inhibitory,:] = JI
        self.JD[n_inhibitory:,:] = JE
        with self, self.rsvr_ens_config:
            u = nengo.Node(size_in=dimensions)
            z = nengo.Node(size_in=dimensions)
            
            xC = nengo.Ensemble(n_neurons, self.dimensions)
            sC = nengo.Node(size_in=n_neurons)  # pre filter
            eC = nengo.Node(size_in=dimensions+1, size_out=dimensions,
                           output=lambda t, e: e[1:] if e[0] else [0.]*dimensions)
            zC = nengo.Node(size_in=dimensions)  # learned output

            nengo.Connection(u, sC, synapse=None, transform=self.e_in)
            nengo.Connection(sC, xC.neurons, synapse=self.tau_input)
            nengo.Connection(xC.neurons, xC.neurons, synapse=self.tau, transform=self.JD)  # chaos
            connC = nengo.Connection(
                xC.neurons[n_inhibitory:] if n_inhibitory > 0 else xC.neurons, zC,
                synapse=None,
                transform=np.zeros((dimensions, n_excitatory)),
                learning_rule_type=RLS(learning_rate=self.learning_rate,
                                       pre_synapse=self.tau_learn))

            nengo.Connection(zC, sC, synapse=self.tau, transform=self.e_out)

            nengo.Connection(zC, eC[1:], synapse=None)  # actual
            nengo.Connection(z, eC[1:], synapse=None, transform=[-1]*dimensions)  # ideal

            nengo.Connection(eC, connC.learning_rule, synapse=self.tau, transform=[1]*dimensions)

            self.enable_learning = nengo.Node(size_in=1)
            nengo.Connection(self.enable_learning, eC[0], synapse=None)
            self.input = u
            self.output = zC
            self.train = z
            self.error = eC
            self.ensemble = xC
            self.weight_saver = WeightSaver(connC, weights_path)
            
    @property
    def rsvr_ens_config(self):
        """(Config) Defaults for reservoir ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update(
            {
                "radius": 1,
                "intercepts": nengo.dists.Choice([0]*self.dimensions),
                "n_eval_points": self.n_eval_points,
            }
        )
        cfg[nengo.Connection].synapse = None
        return cfg

    
