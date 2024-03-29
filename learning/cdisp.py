from nengo.exceptions import SimulationError, ValidationError, BuildError
from nengo.builder import Builder, Operator, Signal
from nengo.learning_rules import *
from nengo.learning_rules import _remove_default_post_synapse
from nengo.builder.learning_rules import *
from nengo.params import (NumberParam, BoolParam)
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from functools import partial
import jax
import jax.numpy as jnp


# Creates new learning rule for coincidence detecting inhibitory plasticity (CDISP).

class CDISP(LearningRuleType):
    """Inhibitory plasticity learning rule.  Modifies connection weights
    according to the presynaptic and postsynaptic firing rates and the
    amplitude of the error signal.

    """
    modifies = 'weights'
    probeable = ('error_filtered', 'pre_filtered', 'post_filtered', 'delta')
    
    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.02), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    sigma_synapse = SynapseParam("sigma_synapse", default=Lowpass(tau=0.005), readonly=True)

    jit = BoolParam("jit", default=True, readonly=True)

    def __init__(self,
                 learning_rate=Default,
                 sigma_synapse=Default,
                 pre_synapse=Default,
                 post_synapse=Default,
                 jit=Default):
        super(CDISP, self).__init__(learning_rate, size_in="post")
        self.sigma_synapse = sigma_synapse
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        self.jit = jit

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)

@jax.jit
def step_jit(kappa, post_filtered, pre_filtered, error_filtered, weights, mask):
    d = -kappa * mask * pre_filtered * error_filtered
    delta_sum = jnp.add(d, weights)
    return jnp.where(delta_sum >= 0, 0. - weights, d)

@jax.jit
def apply_step_jit(kappa, mask, error_filtered, pre_filtered, post_filtered, weights):
    step_vv = jax.vmap(partial(step_jit, kappa, post_filtered, pre_filtered))
    return step_vv(error_filtered.reshape((-1,1)), weights, mask)


    

# Builders for CDISP
class SimCDISP(Operator):
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
    sigma : Signal
        Error signal , :math:`\sigma`.
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
    sigma : Signal
        The error signal, :math:`\sigma`.
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
    3. reads ``[error_filtered, pre_filtered, post_filtered, weights]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_filtered, post_filtered, error_filtered, weights, delta, learning_rate, jit, tag=None):
        super(SimCDISP, self).__init__(tag=tag)
        self.learning_rate = learning_rate
        self.mask = np.logical_not(np.isclose(weights.initial_value, 0.))
        self.sets = []
        self.incs = []
        self.reads = [error_filtered, pre_filtered, post_filtered, weights]
        self.updates = [delta]
        self.jit = jit
        
    @property
    def delta(self):
        return self.updates[0]

    @property
    def error_filtered(self):
        return self.reads[0]

    @property
    def pre_filtered(self):
        return self.reads[1]

    @property
    def post_filtered(self):
        return self.reads[2]
    
    @property
    def weights(self):
        return self.reads[3]
   
    @property
    def _descstr(self):
        return "pre=%s, error=%s -> %s" % (
            self.pre_filtered,
            self.error_filtered,
            self.delta,
        )

    
    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        error_filtered = signals[self.error_filtered]
        weights = signals[self.weights]
        delta = signals[self.delta]
        kappa = self.learning_rate * dt
        mask = self.mask
        jit = self.jit
        
        def step_simcdisp():
            ## The code below is an optimized version of the following:
            #for i in range(weights.shape[0]):
            #    delta[i,:] = -kappa * pre_filtered * mask[i,:] * (post_filtered[i] - rho0)
            #    delta_sum = np.add(delta[i,:], weights[i,:])
            if jit:
                delta[...] = apply_step_jit(kappa, mask, error_filtered, pre_filtered, post_filtered, weights)
            else:
                for i in range(weights.shape[0]):
                    d = -kappa * pre_filtered * self.mask[i] * error_filtered[i]
                    delta_sum = np.add(weights[i], d)
                    delta[i,:] = np.where(delta_sum >= 0, 0. - weights[i], d)
            
        return step_simcdisp

    
@Builder.register(CDISP)
def build_cdisp(model, cdisp, rule):
    """Builds a `.CDISP` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimCDISP` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    cdisp : CDISP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.CDISP` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[conn.pre_obj]["out"]
    if conn.pre_slice is not None:
        pre_activities = pre_activities[conn.pre_slice]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    if conn.post_slice is not None:
        post_activities = post_activities[conn.post_slice]
    pre_filtered = build_or_passthrough(model, cdisp.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, cdisp.post_synapse, post_activities)
    error_filtered = build_or_passthrough(model, cdisp.sigma_synapse,
                                          Signal(shape=rule.size_in, name="CDISP:error"))
    model.sig[rule]["in"] = error_filtered  # error connection will attach here

                
    model.add_op(
        SimCDISP(
            pre_filtered,
            post_filtered,
            error_filtered,
            weights = model.sig[conn]["weights"],
            delta = model.sig[rule]["delta"],
            learning_rate=cdisp.learning_rate,
            jit=cdisp.jit
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
    model.sig[rule]["error_filtered"] = error_filtered

