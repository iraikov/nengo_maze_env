from nengo.exceptions import SimulationError, ValidationError, BuildError
from nengo.builder import Builder, Operator, Signal
from nengo.learning_rules import *
from nengo.builder.learning_rules import *
from nengo.params import (NumberParam, BoolParam)
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from functools import partial
import jax
import jax.numpy as jnp


# Creates new learning rule for gating synaptic plasticity (GSP).

class GSP(LearningRuleType):
    """Gating plasticity learning rule.  Modifies connection weights
    according to the postsynaptic firing rates.

    """
    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'delta')
    
    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.01), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=Lowpass(tau=0.1), readonly=True)
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

@jax.jit
def step_jit(kappa, pre_filtered, post_filtered, weights):
    on = jnp.where(pre_filtered > 0, 0., 1.)
    d = kappa * post_filtered * on
    delta_sum = jnp.add(d, weights)
    return jnp.where(delta_sum <= 0, weights, d)

@jax.jit
def apply_step_jit(kappa, pre_filtered, post_filtered, weights):
    step_vv = jax.vmap(partial(step_jit, kappa, pre_filtered))
    return step_vv(post_filtered, weights)


    

# Builders for GSP
class SimGSP(Operator):
    r"""Calculate connection weight change according to the gating plasticity rule.
    Implements the learning rule of the form:
    .. math:: \delta{} weight_{ij} = \kappa * post
    where
    * :math:`\kappa` is a scalar learning rate
    * :math:`\weight_{ij}` is the connection weight between the two neurons.
    * :math:`a_i` is the activity of the presynaptic neuron.
    * :math:`a_j` is the firing rate of the postsynaptic neuron.
    * :math:`\rho_{0}` is the target firing rate
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
        super(SimGSP, self).__init__(tag=tag)
        self.learning_rate = learning_rate
        self.mask = np.logical_not(np.isclose(weights.initial_value, 0.))
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights]
        self.updates = [delta]
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
        weights = signals[self.weights]
        delta = signals[self.delta]
        kappa = self.learning_rate * dt
        mask = self.mask
        jit = self.jit
        
        def step_simgsp():
            ## The code below is an optimized version of the following:
            #for i in range(weights.shape[0]):
            #    delta[i,:] = kappa * pre_filtered * mask[i,:] * (post_filtered[i] - rho0)
            #    delta_sum = np.add(delta[i,:], weights[i,:])
            if jit:
                delta[...] = apply_step_jit(kappa, pre_filtered, post_filtered, weights) * mask
            else:
                on = 0. if pre_filtered > 0 else 1.
                a = kappa * on
                np.multiply(self.mask, post_filtered, out=delta)
                np.multiply(a[:, np.newaxis], delta, out=delta)
                delta_sum = np.add(delta, weights)
                sat = np.nonzero(delta_sum <= 0)
                delta[sat] = 0.
            
        return step_simgsp

    
@Builder.register(GSP)
def build_gsp(model, gsp, rule):
    """Builds a `.GSP` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimGSP` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    gsp : GSP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.GSP` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[conn.pre_obj]["out"]
    if conn.pre_slice is not None:
        pre_activities = pre_activities[conn.pre_slice]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    if conn.post_slice is not None:
        post_activities = post_activities[conn.post_slice]
    pre_filtered = build_or_passthrough(model, gsp.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, gsp.post_synapse, post_activities)
    weights = model.sig[conn]["weights"]
                
    model.add_op(
        SimGSP(
            pre_filtered,
            post_filtered,
            weights,
            model.sig[rule]["delta"],
            learning_rate=gsp.learning_rate,
            jit=gsp.jit
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered

