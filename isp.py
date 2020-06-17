
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
    
    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    rho0 = NumberParam("rho0", low=0, readonly=True, default=10.)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

    def __init__(self,
                 learning_rate=Default,
                 rho0=Default,
                 pre_synapse=Default,
                 post_synapse=Default,
                 theta_synapse=Default):
        super().__init__(learning_rate, size_in=0)
        self.rho0 = rho0
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)


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
    rho0 : float
        The target firing rate, :math:`\rho_{0}`.
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
    3. reads ``[pre_filtered, post_filtered]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_filtered, post_filtered, rho0, delta, learning_rate, tag=None):
        super(SimISP, self).__init__(tag=tag)
        self.learning_rate = learning_rate
        self.rho0 = rho0
        
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered]
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
        rho0 = self.rho0

        def step_simisp():
            delta[...] = kappa * (pre_filtered * post_filtered - rho0 * pre_filtered)
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
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, isp.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, isp.post_synapse, post_activities)

    model.add_op(
        SimISP(
            pre_filtered,
            post_filtered,
            isp.rho0,
            model.sig[rule]["delta"],
            learning_rate=isp.learning_rate,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered

