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


# Creates new learning rule for inhibitory plasticity (ISP).
# Based on the paper:
#
#    Inhibitory Plasticity Balances Excitation and Inhibition in Sensory
#    Pathways and Memory Networks.
#    T. P. Vogels, H. Sprekeler, F. Zenke, C. Clopath, W. Gerstner.
#    Science 334, 2011
# 

class ISP(LearningRuleType):
    """Inhibitory plasticity learning rule.  Modifies connection weights
    according to the presynaptic and postsynaptic firing rates and the
    target firing rate.

    """
    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'delta')
    
    rho0 = NumberParam("rho0", low=0, readonly=True, default=10.)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.01), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    jit = BoolParam("jit", default=True, readonly=True)

    def __init__(self,
                 rho0=Default,
                 pre_synapse=Default,
                 post_synapse=Default,
                 jit=Default):
        super().__init__(size_in=1)
        self.rho0 = rho0
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        self.jit = jit

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)

@jax.jit
def step_jit(kappa, rho0, pre_filtered, post_filtered, weights):
    kappa1 = jnp.where(post_filtered < rho0, 2*kappa, kappa)
    d = -kappa1 * pre_filtered * (post_filtered - rho0)
    delta_sum = jnp.add(d, weights)
    return jnp.where(delta_sum >= 0, 0. - weights, d)

@jax.jit
def apply_step_jit(kappa, rho0, pre_filtered, post_filtered, weights):
    step_vv = jax.vmap(partial(step_jit, kappa, rho0, pre_filtered))
    return step_vv(post_filtered, weights)


    

# Builders for ISP
class SimISP(Operator):
    r"""Calculate connection weight change according to the inhibitory plasticity rule.
    Implements the learning rule of the form:
    .. math:: \delta{} weight_{ij} = \kappa * (pre * post - \rho_0 * pre)
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
    rho0 : float
        The target firing rate, :math:`\rho_{0}`.
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
    rho0 : float
        The target firing rate, :math:`\rho_{0}`.
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

    def __init__(self, pre_filtered, post_filtered, weights, rho0, delta, learning_rate, jit, tag=None):
        super(SimISP, self).__init__(tag=tag)
        self.rho0 = rho0
        self.mask = np.logical_not(np.isclose(weights.initial_value, 0.))
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights, learning_rate]
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
        weights = signals[self.weights]
        delta = signals[self.delta]
        learning_rate = signals[self.learning_rate]
        rho0 = self.rho0
        mask = self.mask
        jit = self.jit
        
        def step_simisp():
            kappa = learning_rate[0] * dt
            ## The code below is an optimized version of the following:
            #for i in range(weights.shape[0]):
            #    delta[i,:] = -kappa * pre_filtered * mask[i,:] * (post_filtered[i] - rho0)
            #    delta_sum = np.add(delta[i,:], weights[i,:])
            if jit:
                delta[...] = apply_step_jit(kappa, rho0, pre_filtered, post_filtered, weights) * mask
            else:
                a = -kappa * (post_filtered - rho0)
                np.multiply(self.mask, pre_filtered, out=delta)
                np.multiply(a[:, np.newaxis], delta, out=delta)
                delta_sum = np.add(delta, weights)
                sat = np.nonzero(delta_sum >= 0)
                delta[sat] = 0.
            
        return step_simisp

    
@Builder.register(ISP)
def build_isp(model, isp, rule):
    """Builds a `.ISP` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimISP` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    isp : ISP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.ISP` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[conn.pre_obj]["out"]
    if conn.pre_slice is not None:
        pre_activities = pre_activities[conn.pre_slice]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    if conn.post_slice is not None:
        post_activities = post_activities[conn.post_slice]
    pre_filtered = build_or_passthrough(model, isp.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, isp.post_synapse, post_activities)
    weights = model.sig[conn]["weights"]
    # Create input learning rate signal
    learning_rate = Signal(shape=rule.size_in, name="ISP:learning_rate")
    model.add_op(Reset(learning_rate))
    model.sig[rule]["in"] = learning_rate  # learning_rate connection will attach here
                
    model.add_op(
        SimISP(
            pre_filtered,
            post_filtered,
            weights,
            isp.rho0,
            model.sig[rule]["delta"],
            learning_rate=learning_rate,
            jit=isp.jit
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered

