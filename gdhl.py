
from nengo.exceptions import SimulationError, ValidationError, BuildError
from nengo.neurons import LIF, LIFRate
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import *
from nengo.builder.learning_rules import *
from nengo.params import (NumberParam, BoolParam, DictParam)
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from numba import jit

# Creates new learning rule for Generalized Differential Hebbian Learning (GDHL).
# Based on the paper:
#
# Zappacosta S, Mannella F, Mirolli M, Baldassarre G. General
# differential Hebbian learning: Capturing temporal relations between
# events in neural networks and the brain. PLoS Comput
# Biol. 2018;14(8):e1006227. Published 2018 Aug
# 28. doi:10.1371/journal.pcbi.1006227
# 

class GDHL(LearningRuleType):
    """General differential hebbian learning rule.  Modifies connection
    weights according to several components based on presynaptic and
    postsynaptic firing rates and their derivatives.
    """
    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'delta')

    sigma = DictParam("sigma", default={ 'pp': 0., 'pn': 0., 'np': 0., 'nn': 0. }, readonly=True)
    eta = DictParam("eta", default={ 'sp': 0., 'sn': 0., 'ps': 0., 'ns': 0. }, readonly=True)
    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    jit = BoolParam("jit", default=True, readonly=True)

    def __init__(self,
                 sigma=Default,
                 eta=Default,
                 learning_rate=Default,
                 pre_synapse=Default,
                 post_synapse=Default,
                 jit=Default):
        super().__init__(learning_rate, size_in=0)
        self.eta = eta
        self.sigma = sigma
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        self.jit = jit

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)

@jit(nopython=True)
def step_jit(sigma, eta, kappa, pre_filtered, post_filtered, ppre, ppost, weights, mask, dt, d_pre_filtered, d_post_filtered,
             d_pre_pos, d_pre_neg, d_post_pos, d_post_neg, delta):

    d_pre_filtered[:] = (pre_filtered - ppre) / dt
    d_post_filtered[:] = (post_filtered - ppost) / dt

    ppre[:] = pre_filtered
    ppost[:] = post_filtered
    
    pre_pos_inds = np.argwhere(d_pre_filtered > 0)
    pre_neg_inds = np.argwhere(d_pre_filtered <= 0)
    post_neg_inds = np.argwhere(d_post_filtered <= 0)
    post_pos_inds = np.argwhere(d_post_filtered > 0)

    d_pre_pos.fill(0)
    d_pre_neg.fill(0)
    d_post_pos.fill(0)
    d_post_neg.fill(0)
    
    d_pre_pos[pre_pos_inds] = d_pre_filtered[pre_pos_inds]
    d_post_pos[post_pos_inds] = d_post_filtered[post_pos_inds]
    d_pre_neg[pre_neg_inds] = -d_pre_filtered[pre_neg_inds]
    d_post_neg[post_neg_inds] = -d_post_filtered[post_neg_inds]

    for i in range(weights.shape[0]):
        factor = 1.0 - ((weights[i,:] * weights[i,:].T) / (np.dot(weights[i,:], weights[i,:])))
        dw = sigma['pp'] * d_pre_pos * d_post_pos[i] + \
             sigma['pn'] * d_pre_pos * d_post_neg[i] + \
             sigma['np'] * d_pre_neg * d_post_pos[i] + \
             sigma['nn'] * d_pre_neg * d_post_neg[i] + \
             eta['sp'] * pre_filtered * d_post_pos[i] + \
             eta['sn'] * pre_filtered * d_post_neg[i] + \
             eta['ps'] * d_pre_pos * post_filtered[i] + \
             eta['pn'] * d_pre_neg * post_filtered[i]
        delta[i,:] = kappa * factor * mask[i,:] * dw

        
def step(sigma, eta, kappa, pre_filtered, post_filtered, ppre, ppost, weights, mask, dt, d_pre_filtered, d_post_filtered,
         d_pre_pos, d_pre_neg, d_post_pos, d_post_neg, delta):
    
    d_pre_filtered[:] = (pre_filtered - ppre) / dt
    d_post_filtered[:] = (post_filtered - ppost) / dt

    print("step %f" % dt)
    print('ppre: %s' % str(ppre))
    print('pre_filtered: %s' % str(pre_filtered))
    print('d_pre_filtered: %s' % str(d_pre_filtered))

    ppre[:] = pre_filtered
    ppost[:] = post_filtered
    
    pre_pos_inds = np.argwhere(d_pre_filtered > 0)
    pre_neg_inds = np.argwhere(d_pre_filtered <= 0)
    post_neg_inds = np.argwhere(d_post_filtered <= 0)
    post_pos_inds = np.argwhere(d_post_filtered > 0)

    d_pre_pos.fill(0)
    d_pre_neg.fill(0)
    d_post_pos.fill(0)
    d_post_neg.fill(0)
    
    d_pre_pos[pre_pos_inds] = d_pre_filtered[pre_pos_inds]
    d_post_pos[post_pos_inds] = d_post_filtered[post_pos_inds]
    d_pre_neg[pre_neg_inds] = -d_pre_filtered[pre_neg_inds]
    d_post_neg[post_neg_inds] = -d_post_filtered[post_neg_inds]

    for i in range(weights.shape[0]):
        dw = sigma['pp'] * d_pre_pos * d_post_pos[i] + \
             sigma['pn'] * d_pre_pos * d_post_neg[i] + \
             sigma['np'] * d_pre_neg * d_post_pos[i] + \
             sigma['nn'] * d_pre_neg * d_post_neg[i] + \
             eta['sp'] * pre_filtered * d_post_pos[i] + \
             eta['sn'] * pre_filtered * d_post_neg[i] + \
             eta['ps'] * d_pre_pos * post_filtered[i] + \
             eta['ns'] * d_pre_neg * post_filtered[i]
        print('dw: %s' % str(dw))
        delta[i,:] = kappa * mask[i,:] * dw
        wv = weights[i,:] + delta[i,:]
        factor = 1.0 - ((wv * wv.T) / (np.dot(wv, wv)))
        print('factor: %s' % str(factor))
        delta[i, :] = delta[i, :] * factor
    print('delta: %s' % str(delta))
        

# Builders for GDHL
class SimGDHL(Operator):
    r"""Calculate connection weight change according to generalized differential Hebbian plasticity rule.
    Implements the learning rule of the form:
    .. math:: \delta{} weight_{ij} = \kappa * pre * post
    where
    * :math:`\kappa` is a scalar learning rate
    * :math:`\weight_{ij}` is the connection weight between the two neurons.
    * :math:`a_i` is the activity of the presynaptic neuron.
    * :math:`a_j` is the firing rate of the postsynaptic neuron.
    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \weight_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.
    Attributes
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \weight_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, weights]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_filtered, post_filtered, weights, delta, learning_rate, sigma, eta, jit, tag=None):
        super(SimGDHL, self).__init__(tag=tag)
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.eta = eta
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights]
        self.updates = [delta]
        assert(np.all(np.linalg.norm(weights.initial_value, axis=1) > 0.))
        self.mask = np.logical_not(np.isclose(weights.initial_value, 0.))
        self.d_pre_filtered = np.zeros((weights.initial_value.shape[1],))
        self.d_post_filtered = np.zeros((weights.initial_value.shape[0],))
        self.d_pre_pos = np.zeros((weights.initial_value.shape[1],))
        self.d_pre_neg = np.zeros((weights.initial_value.shape[1],))
        self.d_post_pos = np.zeros((weights.initial_value.shape[0],))
        self.d_post_neg = np.zeros((weights.initial_value.shape[0],))
        self.ppre = np.zeros((weights.initial_value.shape[1],))
        self.ppost = np.zeros((weights.initial_value.shape[0],))
        self.jit = jit
        
    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]
    
    @property
    def weights(self):
        return self.reads[2]
   
    @property
    def _descstr(self):
        return "pre=%s, post=%s -> %s" % (
            self.pre_filtered,
            self.post_filtered,
            self.delta,
        )

    
    def make_step(self, signals, dt, rng):
        u = signals[self.pre_filtered]
        v = signals[self.post_filtered]
        delta = signals[self.delta]
        kappa = self.learning_rate * dt
        eta = self.eta
        sigma = self.sigma
        weights = signals[self.weights]
        mask = self.mask
        jit = self.jit
        u0 = self.ppre
        v0 = self.ppost
        d_pre_pos = self.d_pre_pos
        d_pre_neg = self.d_pre_neg
        d_post_pos = self.d_post_pos
        d_post_neg = self.d_post_neg
        d_pre_filtered = self.d_pre_filtered
        d_post_filtered = self.d_post_filtered

        
        def step_simgdhl():
            if jit:
                step_jit(sigma, eta, kappa, u, v, u0, v0, weights, mask, dt, d_pre_filtered, d_post_filtered,
                         d_pre_pos, d_pre_neg, d_post_pos, d_post_neg, delta)
            else:
                step(sigma, eta, kappa, u, v, u0, v0, weights, mask, dt, d_pre_filtered, d_post_filtered,
                     d_pre_pos, d_pre_neg, d_post_pos, d_post_neg, delta)
            
        return step_simgdhl

    
@Builder.register(GDHL)
def build_gdhl(model, gdhl, rule):
    """Builds a `.GDHL` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimGDHL` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    gdhl : GDHL
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.GDHL` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[conn.pre_obj]["out"]
    if conn.pre_slice is not None:
        pre_activities = pre_activities[conn.pre_slice]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    if conn.post_slice is not None:
        post_activities = post_activities[conn.post_slice]
    pre_filtered = build_or_passthrough(model, gdhl.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, gdhl.post_synapse, post_activities)
    weights = model.sig[conn]["weights"]

    model.add_op(
        SimGDHL(
            pre_filtered,
            post_filtered,
            weights,
            model.sig[rule]["delta"],
            learning_rate=gdhl.learning_rate,
            sigma=gdhl.sigma,
            eta=gdhl.eta,
            jit=gdhl.jit,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered

