from functools import partial
from nengo.exceptions import SimulationError, ValidationError, BuildError
from nengo.neurons import LIF, LIFRate
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import _remove_default_post_synapse, LearningRuleType
from nengo.builder.learning_rules import *
from nengo.params import (Default, NumberParam, BoolParam)
from nengo.synapses import (Lowpass, SynapseParam)
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
import jax
import jax.numpy as jnp


# Creates new learning rule for reward-modulated synaptic plasticity (RMSP).
# Based on the paper:
#
# Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons
# Nicolas Frémaux, Henning Sprekeler, Wulfram Gerstner
# PLoS Comp Biol 2013
# 
# 

class RMSP(LearningRuleType):
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
                 theta_synapse=Default,
                 jit=Default):
        super().__init__(learning_rate, size_in=1)
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        self.jit = jit

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)

    
    
@jax.jit
def step_jit(kappa, reward, pre_filtered, rdelta, post_filtered, weights):
    factor = (1.0 - (jnp.square(weights) / jnp.dot(weights, weights))) * reward[0]
    return kappa * factor * rdelta * pre_filtered * post_filtered
    
@jax.jit
def apply_step_jit(kappa, post_filtered, pre_filtered, weights, reward):
    rdelta = pre_filtered - reward
    step_vv = jax.vmap(partial(step_jit, kappa, reward, pre_filtered, rdelta))
    return step_vv(post_filtered, weights)

        
        
# Builders for RMSP
class SimRMSP(Operator):
    r"""Calculate connection weight change according to the Hebbian plasticity rule.
    Implements the learning rule of the form:
    .. math:: \delta{} weight_{ij} = \kappa * pre * post * reward
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
    reward : Signal
        The scalar reward, :math:`r`.
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
    reward : Signal
        The scalar reward, :math:`r`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, weights, reward]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_filtered, post_filtered, weights, reward, delta, learning_rate, jit, tag=None):
        super(SimRMSP, self).__init__(tag=tag)
        self.learning_rate = learning_rate
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights, reward]
        self.updates = [delta]
        assert(np.all(np.linalg.norm(weights.initial_value, axis=1) > 0.))
        self.mask = np.logical_not(np.isclose(weights.initial_value, 0.))
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
    def reward(self):
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
        kappa = self.learning_rate * dt
        weights = signals[self.weights]
        reward = signals[self.reward]
        mask = self.mask
        jit = self.jit
        
        def step_simrmsp():

            if jit:
                dw = apply_step_jit(kappa, post_filtered, pre_filtered, weights, reward)
                delta[:, :] = np.clip(dw * mask, 0., None)
            else:
                rdelta = pre_filtered - reward
                for i in range(weights.shape[0]):
                    factor = (1.0 - ((weights[i,:] * weights[i,:].T) / np.dot(weights[i,:], weights[i,:]))) * reward
                    delta[i,:] = rdelta * kappa * factor * pre_filtered * mask[i,:] * post_filtered[i]

        return step_simrmsp

    
@Builder.register(RMSP)
def build_rmsp(model, rmsp, rule):
    """Builds a `.RMSP` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimRMSP` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    rmsp : RMSP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.RMSP` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[conn.pre_obj]["out"]
    if conn.pre_slice is not None:
        pre_activities = pre_activities[conn.pre_slice]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    if conn.post_slice is not None:
        post_activities = post_activities[conn.post_slice]
    pre_filtered = build_or_passthrough(model, rmsp.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, rmsp.post_synapse, post_activities)
    weights = model.sig[conn]["weights"]
    delta = model.sig[rule]["delta"]

    # Create input reward signal
    reward = Signal(shape=rule.size_in, name="RMSP:reward")
    model.add_op(Reset(reward))
    model.sig[rule]["in"] = reward  # reward connection will attach here

    model.add_op(
        SimRMSP(
            pre_filtered,
            post_filtered,
            weights,
            reward,
            delta,
            learning_rate=rmsp.learning_rate,
            jit=rmsp.jit,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered

