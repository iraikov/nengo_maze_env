from nengo.exceptions import SimulationError, ValidationError, BuildError
from nengo.builder import Builder, Operator, Signal
from nengo.learning_rules import *
from nengo.learning_rules import  _remove_default_post_synapse
from nengo.builder.learning_rules import *
from nengo.params import (NumberParam, BoolParam)
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

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
    
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    jit = BoolParam("jit", default=True, readonly=True)
    directed = BoolParam("directed", default=False, readonly=True)

    def __init__(self,
                 pre_synapse=Default,
                 post_synapse=Default,
                 directed=Default,
                 jit=Default):
        super().__init__(size_in=1)
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        self.jit = jit
        self.directed = directed

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)
    
@jax.jit
def step_undirected_jit(kappa, pre_filtered, post_filtered, weights):
    error = pre_filtered - post_filtered
    abs_error = jnp.abs(error)
    max_abs_error = jnp.max(abs_error)
    w = jnp.minimum(weights, 1.0)
    nweights = weights/jnp.max(weights)
    r = (max_abs_error - abs_error)*(1.0 - nweights)
    h = jnp.where(w>0.0, weights*post_filtered, 0.0)
    #n = jnp.where(w>0.0, weights*pre_filtered, 0.0)
    d = kappa * (r - h)
    delta_sum = jnp.add(d, weights)
    return jnp.where(delta_sum <= 0, 0., d)

@jax.jit
def step_directed_jit(kappa, pre_filtered, post_filtered, weights):
    error = pre_filtered - post_filtered
    w = jnp.minimum(weights, 1.0)
    nweights = weights/jnp.max(weights)
    r = jnp.where(error<0.,
                  jnp.where(w>0.0, error/(0.25*nweights), 0.0),
                  jnp.where(w>0.0, error*(1.01 - nweights), 0.0))
    h = jnp.where(error<0, jnp.where(w>0.0, post_filtered*(1.1 - nweights), 0.0), 0.)
    d = kappa * (r - h)
    delta_sum = jnp.add(d, weights)
    return jnp.where(delta_sum <= 0, 0., d)


@jax.jit
def apply_step_undirected_jit(kappa, post_filtered, pre_filtered, weights):
    step_vv = jax.vmap(partial(step_undirected_jit, kappa, pre_filtered))
    return step_vv(jnp.ravel(post_filtered), weights)

@jax.jit
def apply_step_directed_jit(kappa, post_filtered, pre_filtered, weights):
    step_vv = jax.vmap(partial(step_directed_jit, kappa, pre_filtered))
    return step_vv(jnp.ravel(post_filtered), weights)
        
     

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
    learning_rate : Signal
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
    learning_rate : Signal
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

    def __init__(self, pre_filtered, post_filtered, weights, delta, learning_rate, directed, jit, tag=None):
        super(SimHSP, self).__init__(tag=tag)
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights, learning_rate]
        self.updates = [delta]
        self.sgn = np.ones(weights.initial_value.shape)
        self.directed = directed
        self.jit = jit
        self.mask = np.logical_not(np.isclose(weights.initial_value, 0.)).astype(np.int)
        
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
    def learning_rate(self):
        return self.reads[3]
   
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
        learning_rate = signals[self.learning_rate]
        weights = signals[self.weights]
        mask = self.mask
        sgn = self.sgn
        directed = self.directed
        jit = self.jit

        def step_simhsp():
            kappa = learning_rate[0] * dt
            if kappa > 0.:
                if jit:
                    if directed:
                        dw = apply_step_directed_jit(kappa, post_filtered, pre_filtered, weights)
                    else:
                        dw = apply_step_undirected_jit(kappa, post_filtered, pre_filtered, weights)
                    delta[...] = jnp.multiply(mask, dw)
                else:
                    sgn[:,:] = 1
                    for i in range(weights.shape[0]):
                        factor = 1.0 - ((weights[i,:] * weights[i,:].T) / np.dot(weights[i,:], weights[i,:]))
                        if directed:
                            lt = np.argwhere(pre_filtered < post_filtered[i])
                            if len(lt) > 0:
                                for j in range(lt.shape[0]):
                                    sgn[i,j] = -1
                        delta[i,:] = sgn[i,:] * kappa * factor * pre_filtered * mask[i,:] * post_filtered[i]

            
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

    # Create input learning rate signal
    learning_rate = Signal(shape=rule.size_in, name="HSP:learning_rate")
    model.add_op(Reset(learning_rate))
    model.sig[rule]["in"] = learning_rate  # learning_rate connection will attach here

    
    model.add_op(
        SimHSP(
            pre_filtered,
            post_filtered,
            weights,
            model.sig[rule]["delta"],
            learning_rate=learning_rate,
            directed=hsp.directed,
            jit=hsp.jit,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered

