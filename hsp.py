
from nengo.exceptions import SimulationError, ValidationError, BuildError
from nengo.neurons import LIF, LIFRate
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import *
from nengo.builder.learning_rules import *
from nengo.params import (NumberParam, BoolParam)
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from numba import jit

# Creates new learning rule for Hebbian synaptic plasticity (HSP).
# Based on the paper:
#
# Learning place cells, grid cells and invariances with excitatory and
# inhibitory plasticity.  Simon Nikolaus Weber, Henning Sprekeler.
# Elife 2018;7:e34560.
# 

class HSP(LearningRuleType):
    """Hebbian synaptic plasticity learning rule.  Modifies connection weights
    according to the presynaptic and postsynaptic firing rates and the
    target firing rate.

    """
    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'delta')
    
    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    jit = BoolParam("jit", default=True, readonly=True)

    def __init__(self,
                 learning_rate=Default,
                 pre_synapse=Default,
                 post_synapse=Default,
                 jit=Default):
        super().__init__(learning_rate, size_in=0)
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        self.jit = jit

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)

@jit(nopython=True)
def step_jit(kappa, post_filtered, pre_filtered, weights, sgn, mask, delta):
    for i in range(weights.shape[0]):
        factor = 1.0 - (np.dot(weights[i,:], weights[i,:].T) / np.linalg.norm(weights[i,:]))
        lt = np.argwhere(pre_filtered < post_filtered[i])
        if len(lt) > 0:
            for j in range(lt.shape[0]):
                sgn[i,j] = -1
        delta[i,:] = sgn[i,:] * kappa * factor * pre_filtered * mask[i,:] * post_filtered[i]
     

# Builders for HSP
class SimHSP(Operator):
    r"""Calculate connection weight change according to the Hebbian plasticity rule.
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

    def __init__(self, pre_filtered, post_filtered, weights, delta, learning_rate, jit, tag=None):
        super(SimHSP, self).__init__(tag=tag)
        self.learning_rate = learning_rate
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights]
        self.updates = [delta]
        assert(np.all(np.linalg.norm(weights.initial_value, axis=1) > 0.))
        self.mask = np.logical_not(np.isclose(weights.initial_value, 0.))
        self.sgn = np.ones(weights.initial_value.shape)
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
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        kappa = self.learning_rate * dt
        weights = signals[self.weights]
        mask = self.mask
        sgn = self.sgn
        jit = self.jit
        
        def step_simhsp():
            ## The code below is an optimized version of:
            #for i in range(weights.shape[0]):
            #    factor = 1.0 - (np.dot(weights[i,:], weights[i,:].T) / np.linalg.norm(weights[i,:]))
            #    delta[i,:] = kappa * factor * pre_filtered * self.mask[i,:] * post_filtered[i]

            sgn[:,:] = 1
            if jit:
                step_jit(kappa, post_filtered, pre_filtered, weights, sgn, mask, delta)
            else:
                factor = 1.0 - (np.einsum('ij,ji->i',weights, weights.T) / np.linalg.norm(weights,axis=1))
                a = kappa * factor * post_filtered
                for i in range(weights.shape[0]):
                    sgn[i,np.flatnonzero(pre_filtered < post_filtered[i])] = -1
                np.multiply(mask, pre_filtered, out=delta)
                np.multiply(np.expand_dims(a, 1), delta, out=delta)
                np.multiply(sgn, delta, out=delta)
                delta_sum = np.add(delta, weights)
                negz = np.nonzero(delta_sum <= 1e-6)
                delta[negz] = 0.
            
        return step_simhsp

    
@Builder.register(HSP)
def build_hsp(model, hsp, rule):
    """Builds a `.HSP` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimHSP` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    hsp : HSP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.HSP` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[conn.pre_obj]["out"]
    if conn.pre_slice is not None:
        pre_activities = pre_activities[conn.pre_slice]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    if conn.post_slice is not None:
        post_activities = post_activities[conn.post_slice]
    pre_filtered = build_or_passthrough(model, hsp.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, hsp.post_synapse, post_activities)
    weights = model.sig[conn]["weights"]

    model.add_op(
        SimHSP(
            pre_filtered,
            post_filtered,
            weights,
            model.sig[rule]["delta"],
            learning_rate=hsp.learning_rate,
            jit=hsp.jit,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered

