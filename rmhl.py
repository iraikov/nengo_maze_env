
from nengo.exceptions import SimulationError, ValidationError, BuildError
from nengo.neurons import LIF, LIFRate
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import *
from nengo.builder.learning_rules import *
from nengo.params import (NumberParam)
from nengo.utils.compat import is_iterable, is_string, itervalues, range
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons

# Creates new learning rule for Reward-Modulated Hebbian Learning (RMHL).
# Based on the paper:
#
# Hoerzer, G.  M., Legenstein, R., & Maass, W.  (2014).  Emergence of
# complex computationalstructures from chaotic neural networks through
# reward-modulated hebbian learning.Cereb.Cort.,24(3), 677–690
# 

class RMHL(LearningRuleType):
    """Reward-Modulated Hebbian Learning rule.  Modifies connection weights
    according to the postsynaptic firing rate and the error signal.

    """
    modifies = 'weights'
    probeable = ('pre_filtered', 'post_filtered', 'error', 'delta')
    
    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

    def __init__(self,
                 learning_rate=Default,
                 pre_synapse=Default,
                 post_synapse=Default):
        super().__init__(learning_rate, size_in=1)
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )


    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)


# Builders for RMHL
class SimRMHL(Operator):
    r"""Calculate connection weight change according to the RMHL rule.
    Implements the learning rule of the form:
    .. math:: \delta{} weight_{ij} = \kappa * \hat{e} * \hat{z} * a_i
    where
    * :math:`\kappa` is a scalar learning rate
    * :math:`\weight_{ij}` is the connection weight between the two neurons.
    * :math:`a_i` is the activity of the presynaptic neuron.
    * :math:`\hat{z}` is the filtered signal of the pre-synaptic ensemble
    * :math:`\hat{e}` is the filtered error signal

    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The decoded output activity, :math:`\hat{z}`.
    error : Signal
        The error signal, :math:`\hat{e}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.
    Attributes
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The decoded output activity, :math:`\hat{z}`.
    error : Signal
        The error signal, :math:`\hat{e}`.
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
    3. reads ``[pre_filtered, error]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_filtered, post_filtered, error, delta, learning_rate, tag=None):
        super(SimRMHL, self).__init__(tag=tag)
        self.learning_rate = learning_rate
        
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, error]
        self.updates = [delta]

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
    def error(self):
        return self.reads[2]
    
    @property
    def _descstr(self):
        return "pre=%s, post=%s, error=%s -> %s" % (
            self.pre_filtered,
            self.post_filtered,
            self.error,
            self.delta,
        )

    
    def make_step(self, signals, dt, rng):
        r = signals[self.pre_filtered]
        z = signals[self.post_filtered]
        kappa = self.learning_rate * dt
        error = signals[self.error]

        def step_simrmhl():
            delta[...] = kappa * error * z * r.T
        return step_simrmhl

    
@Builder.register(RMHL)
def build_rmhl(model, rmhl, rule):
    """Builds a `.RMHL` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimRMHL` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    rmhl : RMHL
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.RMHL` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[conn.pre_obj]["out"]
    pre_filtered = build_or_passthrough(model, rmhl.pre_synapse, pre_activities)
    post_activities = model.sig[conn.post_obj]["out"]
    post_filtered = build_or_passthrough(model, rmhl.post_synapse, post_activities)
    assert pre_filtered.ndim == 1
    
    # Create input error signal
    error = Signal(shape=rule.size_in, name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]['in'] = error
    
    model.add_op(
        SimRMHL(
            pre_filtered=pre_filtered,
            post_filtered=post_filtered,
            error=error,
            delta=model.sig[rule]["delta"],
            learning_rate=rmhl.learning_rate,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
    model.sig[rule]["error"] = error

