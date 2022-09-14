"""Nengo implementations of STDP rules."""

import nengo
from nengo.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens
from nengo.builder.operator import Operator, Reset, Copy
from nengo.builder.signal import Signal
from nengo.params import Default, BoolParam, NumberParam, StringParam
import numpy as np


# ================
# Frontend objects
# ================
#
# These objects are the ones that you include in your model description.
# They are applied to specific connections between groups of neurons.


class STDP(nengo.learning_rules.LearningRuleType):
    """Spike-timing dependent plasticity rule."""

    # Used by other Nengo objects
    modifies = "weights"
    probeable = ("pre_trace", "post_trace", "pre_scale", "post_scale")

    # Parameters
    pre_tau = NumberParam("pre_tau", low=0, low_open=True)
    pre_amp = NumberParam("pre_amp", low=0, low_open=True)
    post_tau = NumberParam("post_tau", low=0, low_open=True)
    post_amp = NumberParam("post_amp", low=0, low_open=True)
    bounds = StringParam("bounds")
    max_weight = NumberParam("max_weight")
    min_weight = NumberParam("min_weight")

    def __init__(
        self,
        pre_tau=0.0168,
        post_tau=0.0337,
        pre_amp=1.0,
        post_amp=1.0,
        bounds="hard",
        max_weight=0.3,
        min_weight=-0.3,
        learning_rate=1e-9,
    ):
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.bounds = str(bounds).lower()
        self.max_weight = max_weight
        self.min_weight = min_weight
        super(STDP, self).__init__(learning_rate)


        
class LogSTDP(nengo.learning_rules.LearningRuleType):
    """Log spike-timing dependent plasticity rule.

    From "Stability versus Neuronal Specialization for STDP: Long-Tail Weight Distributions Solve the Dilemma"
    Matthieu Gilson, Tomoki Fukai, 2011.
    """

    # Used by other Nengo objects
    modifies = "weights"
    probeable = ("pre_trace", "post_trace", "pre_scale", "post_scale")
    
    # Parameters
    pre_tau = NumberParam("pre_tau", low=0, low_open=False)
    pre_amp = NumberParam("pre_amp", low=0, low_open=False)
    post_tau = NumberParam("post_tau", low=0, low_open=False)
    post_amp = NumberParam("post_amp", low=0, low_open=False)
    alpha = NumberParam("alpha", low=0, low_open=False)
    beta = NumberParam("beta", low=0, low_open=False)
    pre_c = NumberParam("pre_c", low=0, low_open=False)
    post_c = NumberParam("post_c", low=0, low_open=False)
    w0 = NumberParam("w0", low=1e-6, low_open=True)

    
    def __init__(self, pre_tau=0.01, post_tau=0.03, pre_amp=1e-3, post_amp=1e-3,
                 alpha=5, beta=50, pre_c=1, post_c=1, w0=1e-4):
        
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.alpha = alpha
        self.beta = beta
        self.pre_c = pre_c
        self.post_c = post_c
        self.w0 = w0
        
        super(LogSTDP, self).__init__(size_in=1)

        
        
class RdSTDP(nengo.learning_rules.LearningRuleType):
    """Resource-dependent spike-timing dependent plasticity rule.

    From "Learning complex temporal patterns with resource-dependent spike timing-dependent plasticity",
    Jason F. Hunzinger, Victor H. Chan, and Robert C. Froemke, 2012.
    """

    # Used by other Nengo objects
    modifies = "weights"
    probeable = ("pre_trace", "post_trace", "pre_scale", "post_scale", "r")
    
    # Parameters
    pre_tau = NumberParam("pre_tau", low=0, low_open=False)
    pre_amp = NumberParam("pre_amp", low=0, low_open=False)
    post_tau = NumberParam("post_tau", low=0, low_open=False)
    post_amp = NumberParam("post_amp", low=0, low_open=False)
    r_tau = NumberParam("r_tau", low=1e-3, low_open=True)
    max_weight = NumberParam("max_weight", low=0)
    min_weight = NumberParam("min_weight", low=0)
    
    def __init__(self, max_weight=1e-3, min_weight=1e-6, pre_tau=0.01, post_tau=0.03, pre_amp=1, post_amp=1, r_tau=1e-4):

        self.max_weight = max_weight
        self.min_weight = min_weight
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.r_tau = r_tau
        
        super(RdSTDP, self).__init__(size_in=1)

class HvSTDP(nengo.learning_rules.LearningRuleType):
    """Homeostatic voltage-dependent spike-timing dependent plasticity rule.
    """

    # Used by other Nengo objects
    modifies = "weights"
    probeable = ("pre_trace", "post_trace", "pre_scale", "post_scale")
    
    # Parameters
    pre_tau = NumberParam("pre_tau", low=0, low_open=False)
    pre_amp = NumberParam("pre_amp", low=0, low_open=False)
    post_tau = NumberParam("pre_tau", low=0, low_open=False)
    post_amp = NumberParam("pre_amp", low=0, low_open=False)
    bounds = StringParam("bounds")
    max_weight = NumberParam("max_weight")
    min_weight = NumberParam("min_weight")
    theta_pos = NumberParam("theta_pos")
    
    def __init__(self, theta_pos=0.5, theta_neg=0.0, pre_tau=0.01, pre_amp=1e-3, post_tau=0.01, post_amp=1e-3,
                 min_weight=1e-6, max_weight=1, bounds="soft"):
        
        self.pre_tau = pre_tau
        self.pre_amp = pre_amp
        self.post_tau = post_tau
        self.post_amp = post_amp
        self.bounds = str(bounds).lower()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.theta_pos = theta_pos
        self.theta_neg = theta_neg
        
        super(HvSTDP, self).__init__(size_in=1)


# ===============
# Backend objects
# ===============
#
# These objects let the Nengo core backend know how to implement the rules
# defined in the model through frontend objects. They require some knowledge
# of the low-level details of how the Nengo core backends works, and will
# be different depending on the backend on which the learning rule is implemented.
# The general architecture of the Nengo core backend is described at
#   https://www.nengo.ai/nengo/backend_api.html
# but in the context of learning rules, each learning rule needs a build function
# that is associated with a frontend object (through the `Builder.register`
# function) that sets up the signals and operators that implement the rule.
# Nengo comes with many general purpose operators that could be combined
# to implement a learning rule, but in most cases it is easier to implement
# them using a custom operator that does the delta update equation.
# See, for example, `step_stdp` in the `SimSTDP` operator to see where the
# learning rule's equation is actually specified. The build function exists
# mainly to make sure to all of the signals used in the operator are the
# correct ones. This requires some knowledge of the Nengo core backend,
# but for learning rules they are all very similar, and this could be made
# more convenient through some new abstractions; see
#  https://github.com/nengo/nengo/pull/553
#  https://github.com/nengo/nengo/pull/1149
# for some initial attempts at making this more convenient.


@Builder.register(STDP)
def build_stdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_trace = Signal(np.zeros(pre_activities.size), name="pre_trace")
    post_trace = Signal(np.zeros(post_activities.size), name="post_trace")
    pre_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="pre_scale")
    post_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="post_scale")

    model.add_op(
        SimSTDP(
            pre_activities,
            post_activities,
            pre_trace,
            post_trace,
            pre_scale,
            post_scale,
            model.sig[conn]["weights"],
            model.sig[rule]["delta"],
            learning_rate=stdp.learning_rate,
            pre_tau=stdp.pre_tau,
            post_tau=stdp.post_tau,
            pre_amp=stdp.pre_amp,
            post_amp=stdp.post_amp,
            bounds=stdp.bounds,
            max_weight=stdp.max_weight,
            min_weight=stdp.min_weight,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_trace"] = pre_trace
    model.sig[rule]["post_trace"] = post_trace
    model.sig[rule]["pre_scale"] = pre_scale
    model.sig[rule]["post_scale"] = post_scale

    model.params[rule] = None  # no build-time info to return


class SimSTDP(Operator):
    def __init__(
        self,
        pre_activities,
        post_activities,
        pre_trace,
        post_trace,
        pre_scale,
        post_scale,
        weights,
        delta,
        learning_rate,
        pre_tau,
        post_tau,
        pre_amp,
        post_amp,
        bounds,
        max_weight,
        min_weight,
    ):
        self.learning_rate = learning_rate
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.bounds = str(bounds).lower()
        self.max_weight = max_weight
        self.min_weight = min_weight

        self.sets = []
        self.incs = []
        self.reads = [pre_activities, post_activities, weights]
        self.updates = [delta, pre_trace, post_trace, pre_scale, post_scale]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def post_activities(self):
        return self.reads[1]

    @property
    def post_scale(self):
        return self.updates[4]

    @property
    def post_trace(self):
        return self.updates[2]

    @property
    def pre_activities(self):
        return self.reads[0]

    @property
    def pre_scale(self):
        return self.updates[3]

    @property
    def pre_trace(self):
        return self.updates[1]

    @property
    def weights(self):
        return self.reads[2]

    def make_step(self, signals, dt, rng):
        pre_activities = signals[self.pre_activities]
        post_activities = signals[self.post_activities]
        pre_trace = signals[self.pre_trace]
        post_trace = signals[self.post_trace]
        pre_scale = signals[self.pre_scale]
        post_scale = signals[self.post_scale]
        weights = signals[self.weights]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        # Could be configurable
        pre_ampscale = 1.0
        post_ampscale = 1.0

        if self.bounds == "hard":

            def update_scales():
                pre_scale[...] = ((self.max_weight - weights) > 0.0).astype(
                    np.float64
                ) * pre_ampscale
                post_scale[...] = (
                    -((self.min_weight + weights) < 0.0).astype(np.float64)
                    * post_ampscale
                )

        elif self.bounds == "soft":

            def update_scales():
                pre_scale[...] = (self.max_weight - weights) * pre_ampscale
                post_scale[...] = (self.min_weight + weights) * post_ampscale

        elif self.bounds == "none":

            def update_scales():
                pre_scale[...] = pre_ampscale
                post_scale[...] = -post_ampscale

        else:
            raise RuntimeError(
                "Unsupported bounds type. Only 'hard', 'soft' and 'none' are supported."
            )

        def step_stdp():
            update_scales()
            pre_trace[...] += (dt / self.pre_tau) * (
                -pre_trace + self.pre_amp * pre_activities
            )
            post_trace[...] += (dt / self.post_tau) * (
                -post_trace + self.post_amp * post_activities
            )
            delta[...] = alpha * (
                pre_scale * pre_trace[np.newaxis, :] * post_activities[:, np.newaxis]
                + post_scale * post_trace[:, np.newaxis] * pre_activities
            )

        return step_stdp


@Builder.register(LogSTDP)
def build_logstdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_trace = Signal(np.zeros(pre_activities.size), name="pre_trace")
    post_trace = Signal(np.zeros(post_activities.size), name="post_trace")
    pre_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="pre_scale")
    post_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="post_scale")

    # Create input learning rate signal
    learning_rate = Signal(shape=rule.size_in, name="LogSTDP:learning_rate")
    model.add_op(Reset(learning_rate))
    model.sig[rule]["in"] = learning_rate  # learning_rate connection will attach here

    model.add_op(
        SimLogSTDP(
            pre_activities,
            post_activities,
            pre_trace,
            post_trace,
            pre_scale,
            post_scale,
            model.sig[conn]["weights"],
            model.sig[rule]["delta"],
            learning_rate=model.sig[rule]["in"],
            pre_tau=stdp.pre_tau,
            post_tau=stdp.post_tau,
            pre_amp=stdp.pre_amp,
            post_amp=stdp.post_amp,
            alpha=stdp.alpha,
            beta=stdp.beta,
            pre_c=stdp.pre_c,
            post_c=stdp.post_c,
            w0=stdp.w0,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_trace"] = pre_trace
    model.sig[rule]["post_trace"] = post_trace
    model.sig[rule]["pre_scale"] = pre_scale
    model.sig[rule]["post_scale"] = post_scale

class SimLogSTDP(Operator):
    def __init__(
        self,
        pre_activities,
        post_activities,
        pre_trace,
        post_trace,
        pre_scale,
        post_scale,
        weights,
        delta,
        learning_rate,
        pre_tau,
        post_tau,
        pre_amp,
        post_amp,
        alpha,
        beta,
        pre_c,
        post_c,
        w0,
    ):
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.alpha = alpha
        self.beta = beta
        self.pre_c = pre_c
        self.post_c = post_c
        self.w0 = w0
        self.nz = np.logical_not(np.isclose(weights.initial_value, 0.)).astype(np.int)
        
        self.sets = []
        self.incs = []
        self.reads = [pre_activities, post_activities, weights, learning_rate]
        self.updates = [delta, pre_trace, post_trace, pre_scale, post_scale]

    @property
    def learning_rate(self):
        return self.reads[3]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def post_activities(self):
        return self.reads[1]

    @property
    def post_trace(self):
        return self.updates[2]

    @property
    def post_scale(self):
        return self.updates[4]

    @property
    def pre_activities(self):
        return self.reads[0]

    @property
    def pre_trace(self):
        return self.updates[1]

    @property
    def pre_scale(self):
        return self.updates[3]

    @property
    def weights(self):
        return self.reads[2]


    def make_step(self, signals, dt, rng):
        w0 = self.w0
        alpha = self.alpha
        beta = self.beta
        pre_c = self.pre_c
        post_c = self.post_c
        learning_rate = signals[self.learning_rate]
        pre_activities = signals[self.pre_activities]
        post_activities = signals[self.post_activities]
        pre_trace = signals[self.pre_trace]
        post_trace = signals[self.post_trace]
        pre_scale = signals[self.pre_scale]
        post_scale = signals[self.post_scale]
        delta = signals[self.delta]
        weights = signals[self.weights]
            
        def step_logstdp():

            pre_t = dt / self.pre_tau
            post_t = dt / self.post_tau

            kappa = learning_rate[0] * dt

            pre_scale = np.exp(-weights/w0*beta) * pre_c
            mask = (weights <= w0)
            post_scale = np.empty_like(weights)
            post_scale[mask] = -(weights[mask] / w0) * post_c
            post_scale[~mask] = -(1 + np.log(1 + alpha * (weights[~mask]/w0 - 1))/alpha) * post_c

            pre_trace[...] += pre_t * (-pre_trace + self.pre_amp * pre_activities)
            post_trace[...] += post_t * (-post_trace + self.post_amp * post_activities)
            
            delta[...] = self.nz * kappa * (pre_scale * pre_trace[np.newaxis, :] * post_activities[:, np.newaxis]
                + post_scale * post_trace[:, np.newaxis] * pre_activities)
                
        return step_logstdp


@Builder.register(RdSTDP)
def build_rdstdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_trace = Signal(np.zeros(pre_activities.size), name="pre_trace")
    post_trace = Signal(np.zeros(post_activities.size), name="post_trace")
    pre_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="pre_scale")
    post_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="post_scale")
    r = Signal(np.zeros(model.sig[conn]["weights"].shape[0]), name="r")

    # Create input learning rate signal
    learning_rate = Signal(shape=rule.size_in, name="RdSTDP:learning_rate")
    model.add_op(Reset(learning_rate))
    model.sig[rule]["in"] = learning_rate  # learning_rate connection will attach here

    model.add_op(
        SimRdSTDP(
            pre_activities,
            post_activities,
            pre_trace,
            post_trace,
            pre_scale,
            post_scale,
            r,
            model.sig[conn]["weights"],
            model.sig[rule]["delta"],
            learning_rate=model.sig[rule]["in"],
            max_weight=stdp.max_weight,
            min_weight=stdp.min_weight,
            pre_tau=stdp.pre_tau,
            post_tau=stdp.post_tau,
            pre_amp=stdp.pre_amp,
            post_amp=stdp.post_amp,
            r_tau=stdp.r_tau,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_trace"] = pre_trace
    model.sig[rule]["post_trace"] = post_trace
    model.sig[rule]["pre_scale"] = pre_scale
    model.sig[rule]["post_scale"] = post_scale
    model.sig[rule]["r"] = r

    
class SimRdSTDP(Operator):
    def __init__(
        self,
        pre_activities,
        post_activities,
        pre_trace,
        post_trace,
        pre_scale,
        post_scale,
        r,
        weights,
        delta,
        learning_rate,
        max_weight,
        min_weight,
        pre_tau,
        post_tau,
        pre_amp,
        post_amp,
        r_tau,
    ):
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.r_tau = r_tau
        self.max_weight=max_weight
        self.min_weight=min_weight
        self.nz = (weights.initial_value > min_weight).astype(np.int)
        
        self.sets = []
        self.incs = []
        self.reads = [pre_activities, post_activities, weights, learning_rate]
        self.updates = [delta, pre_trace, post_trace, pre_scale, post_scale, r]

    @property
    def learning_rate(self):
        return self.reads[3]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def post_activities(self):
        return self.reads[1]

    @property
    def post_trace(self):
        return self.updates[2]

    @property
    def post_scale(self):
        return self.updates[4]

    @property
    def r(self):
        return self.updates[5]

    @property
    def pre_activities(self):
        return self.reads[0]

    @property
    def pre_trace(self):
        return self.updates[1]

    @property
    def pre_scale(self):
        return self.updates[3]

    @property
    def weights(self):
        return self.reads[2]


    def make_step(self, signals, dt, rng):

        r = signals[self.r]
        learning_rate = signals[self.learning_rate]
        pre_activities = signals[self.pre_activities]
        post_activities = signals[self.post_activities]
        pre_trace = signals[self.pre_trace]
        post_trace = signals[self.post_trace]
        pre_scale = signals[self.pre_scale]
        post_scale = signals[self.post_scale]
        delta = signals[self.delta]
        weights = signals[self.weights]

        r_amp = np.max((self.pre_amp, self.post_amp))
        r_t = dt / self.r_tau
        pre_t = dt / self.pre_tau
        post_t = dt / self.post_tau

        
        def step_rdstdp():

            kappa = learning_rate[0] * dt

            pre_scale[...] = (self.max_weight - weights)
            post_scale[...] = -(self.min_weight + weights)
            

            pre_trace[...] += pre_t * (
                -pre_trace + self.pre_amp * pre_activities
            )
            post_trace[...] += post_t * (
                -post_trace + self.post_amp * post_activities
            )

            r[...] = 1 - (1 - r) * np.exp(-r_t)
            
            delta[...] = self.nz * kappa * r[:, np.newaxis] * (
                pre_scale * pre_trace[np.newaxis, :] * post_activities[:, np.newaxis]
                + post_scale * post_trace[:, np.newaxis] * pre_activities
            )

            r[...] = np.clip(r - np.sum(np.abs(delta), axis=1) / r_amp / dt, 0.0, None)
            
        return step_rdstdp
 

@Builder.register(HvSTDP)
def build_hvstdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_voltages = model.sig[get_post_ens(conn).neurons]["voltage"]
    pre_trace = Signal(np.zeros(pre_activities.size), name="pre_trace")
    pre_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="pre_scale")
    post_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="post_scale")
    post_trace = Signal(np.zeros(post_voltages.size), name="post_trace")

    model.add_op(
        SimHvSTDP(
            pre_activities,
            post_voltages,
            pre_trace,
            pre_scale,
            post_trace,
            post_scale,
            model.sig[conn]["weights"],
            model.sig[rule]["delta"],
            learning_rate=stdp.learning_rate,
            pre_tau=stdp.pre_tau,
            pre_amp=stdp.pre_amp,
            post_tau=stdp.post_tau,
            post_amp=stdp.post_amp,
            bounds=stdp.bounds,
            max_weight=stdp.max_weight,
            min_weight=stdp.min_weight,
            theta_pos = stdp.theta_pos,
            theta_neg = stdp.theta_neg
        )
    )

    # expose these for probes
    model.sig[rule]["pre_trace"] = pre_trace
    model.sig[rule]["pre_scale"] = pre_scale
    model.sig[rule]["post_scale"] = post_scale
    model.sig[rule]["post_trace"] = post_trace

    # Create input learning rate signal
    learning_rate = Signal(shape=rule.size_in, name="HvSTDP:learning_rate")
    model.add_op(Reset(learning_rate))
    model.sig[rule]["in"] = learning_rate  # learning_rate connection will attach here

    model.params[rule] = None  # no build-time info to return


class SimHvSTDP(Operator):
    def __init__(
        self,
        pre_activities,
        post_voltages,
        pre_trace,
        pre_scale,
        post_trace,
        post_scale,
        weights,
        delta,
        learning_rate,
        pre_tau,
        pre_amp,
        post_tau,
        post_amp,
        bounds,
        max_weight,
        min_weight,
        theta_pos,
        theta_neg,
    ):
        self.learning_rate = learning_rate
        self.pre_tau = pre_tau
        self.pre_amp = pre_amp
        self.post_tau = post_tau
        self.post_amp = post_amp
        self.bounds = str(bounds).lower()
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.theta_pos = theta_pos
        self.theta_neg = theta_neg
        
        self.sets = []
        self.incs = []
        self.reads = [pre_activities, post_voltages, weights]
        self.updates = [delta, pre_trace, pre_scale, post_trace, post_scale]
    
    @property
    def post_voltages(self):
        return self.reads[1]

    @property
    def weights(self):
        return self.reads[2]

    @property
    def pre_activities(self):
        return self.reads[0]


    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_scale(self):
        return self.updates[2]

    @property
    def post_scale(self):
        return self.updates[4]

    @property
    def pre_trace(self):
        return self.updates[1]
    
    @property
    def post_trace(self):
        return self.updates[3]

    def make_step(self, signals, dt, rng):
        pre_activities = signals[self.pre_activities]
        post_voltages = signals[self.post_voltages]
        pre_trace = signals[self.pre_trace]
        pre_scale = signals[self.pre_scale]
        post_trace = signals[self.post_trace]
        post_scale = signals[self.post_scale]
        weights = signals[self.weights]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt
        theta_neg = self.theta_neg
        theta_pos = self.theta_pos
        
        # Could be configurable
        pre_ampscale = 1.0
        post_ampscale = 1.0

        if self.bounds == "soft":

            def update_scales():
                pre_scale[...] = (self.max_weight - weights) * pre_ampscale
                post_scale[...] = (weights - self.min_weight) * post_ampscale

        elif self.bounds == "none":

            def update_scales():
                pre_scale[...] = pre_ampscale
                post_scale[...] = -post_ampscale

        else:
            raise RuntimeError(
                "Unsupported bounds type. Only 'soft' and 'none' are supported."
            )

        def step_stdp():
            update_scales()
            pre_trace[...] += (dt / self.pre_tau) * (
                -pre_trace + self.pre_amp * pre_activities
            )
            post_trace[...] += (dt / self.post_tau) * (
                -post_trace + self.post_amp * post_voltages
            )

            u = post_voltages[:, np.newaxis] - self.theta_neg
            v = post_trace[:, np.newaxis]
            
            delta[...] = alpha * (
                pre_scale * pre_trace[np.newaxis, :] * np.where(u > 0., u, 0.0)
                - post_scale * pre_activities * v
            )
            
        return step_stdp
