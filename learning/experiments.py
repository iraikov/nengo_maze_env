import warnings

import matplotlib.pyplot as plt
import multiprocess as mp
import nengo
import numpy as np
import seaborn as sns
from hyperopt import fmin, hp, space_eval, tpe
from nengo.cache import NoDecoderCache
from nengo.processes import WhiteSignal
from nengo.utils.numpy import rmse
from nengo.utils.stdlib import Timer

from learning import data, plots

nengo.rc.set("progress", "progress_bar", "False")


# ######
# Models
# ######

class LearnNet(object):
    pulse_len = 0.002
    scale = 2.5

    def __init__(self, learning_rule, freq_hz, pre_size=1, post_size=1, initial_weight=0., learning_rate_func=None):
        self.initial_weight = initial_weight
        self.learning_rate_func = learning_rate_func
        self.learning_rule = learning_rule
        self.freq_hz = freq_hz
        self.period = 1. / freq_hz
        self.pre_size = 1
        self.post_size = 1
        
    def close_to(self, t, target):
        d = abs((t % self.period) - target)
        return float(d < self.pulse_len) * self.scale

    def pre_f(self, t):
        raise NotImplementedError("Must implement `pre_f`")

    def post_f(self, t):
        raise NotImplementedError("Must implement `post_f`")

    def build(self):
        with nengo.Network() as net:
            e_args = dict(neuron_type=nengo.LIF(amplitude=0.1),
                          encoders=nengo.dists.Choice([[1]]),
                          intercepts=nengo.dists.Choice([0.5]),
                          max_rates=nengo.dists.Choice([100]))

            net.pre = nengo.Ensemble(self.pre_size, dimensions=1, **e_args)
            net.post = nengo.Ensemble(self.post_size, dimensions=1, **e_args)
            nengo.Connection(nengo.Node(self.pre_f), net.pre, synapse=None)
            nengo.Connection(nengo.Node(self.post_f), net.post, synapse=None)
            net.conn = nengo.Connection(net.pre.neurons, net.post.neurons,
                                        transform=[[self.initial_weight]],
                                        learning_rule_type=self.learning_rule)
            net.learning_rate_conn = None
            net.learning_rate_node = None
            if self.learning_rate_func is not None:
                net.learning_rate_node = nengo.Node(self.learning_rate_func)
                net.learning_rate_conn = nengo.Connection(net.learning_rate_node,
                                                          net.conn.learning_rule)
        net.freq_hz = self.freq_hz
        return net


class PairNet(LearnNet):
    
    def __init__(self, learning_rule, freq_hz, t_diff, initial_weight=0., learning_rate_func=None):
        super(PairNet, self).__init__(learning_rule, freq_hz,
                                      initial_weight=initial_weight,
                                      learning_rate_func=learning_rate_func)

        if abs(t_diff * 2) > self.period:
            warnings.warn("Too fast for STDP curve pair protocol")
        self.pre_target = self.period * 0.5 + t_diff * 0.5
        self.post_target = self.period * 0.5 - t_diff * 0.5

    def pre_f(self, t):
        return self.close_to(t, self.pre_target)

    def post_f(self, t):
        return self.close_to(t, self.post_target)




class PrePostPreNet(LearnNet):
    def __init__(self, learning_rule, freq_hz, t_diff1, t_diff2, initial_weight=0., learning_rate_func=None):
        super(PrePostPreNet, self).__init__(learning_rule, freq_hz,
                                            initial_weight=initial_weight,
                                            learning_rate_func=learning_rate_func)

        assert t_diff1 * t_diff2 < 0, "One should be negative"
        if (abs(t_diff1) + abs(t_diff2)) * 2 > self.period:
            warnings.warn("Too fast for pre-post-pre protocol")
        self.pre_target1 = self.period * 0.5 + t_diff1
        self.pre_target2 = self.period * 0.5 + t_diff2
        self.post_target = self.period * 0.5

    def pre_f(self, t):
        return (self.close_to(t, self.pre_target1)
                + self.close_to(t, self.pre_target2))

    def post_f(self, t):
        return self.close_to(t, self.post_target)


class PostPrePostNet(LearnNet):
    def __init__(self, learning_rule, freq_hz, t_diff1, t_diff2, initial_weight=0., learning_rate_func=None):
        super(PostPrePostNet, self).__init__(learning_rule, freq_hz,
                                             initial_weight=initial_weight,
                                             learning_rate_func=learning_rate_func)

        assert t_diff1 * t_diff2 < 0, "One should be negative"
        if (abs(t_diff1) + abs(t_diff2)) * 2 > self.period:
            warnings.warn("Too fast for post-pre-post protocol")
        self.pre_target = self.period * 0.5
        self.post_target1 = self.period * 0.5 - t_diff1
        self.post_target2 = self.period * 0.5 - t_diff2

    def pre_f(self, t):
        return self.close_to(t, self.pre_target)

    def post_f(self, t):
        return (self.close_to(t, self.post_target1)
                + self.close_to(t, self.post_target2))


class ConvergentLearnNet(object):

    def __init__(self, learning_rule, pre_spikes, pre_size=1, post_size=1, initial_weight=0., dt=0.001, learning_rate_func=None):
        self.initial_weight = initial_weight
        self.learning_rate_func = learning_rate_func
        self.learning_rule = learning_rule
        self.pre_size = pre_size
        self.post_size = 1
        self.pre_spikes = pre_spikes
        self.dt = dt
        self.t_end = pre_spikes.shape[1]*self.dt
        
    def build(self):
        with nengo.Network() as net:
            e_args = dict(encoders=nengo.dists.Choice([[1]]),
                          intercepts=nengo.dists.Choice([0.5]),
                          max_rates=nengo.dists.Choice([100]))

            net.pre = nengo.Ensemble(self.pre_size, dimensions=1, **e_args)
            net.post = nengo.Ensemble(self.post_size, dimensions=1, **e_args)
            net.input_node = nengo.Node(nengo.processes.PresentInput(self.pre_spikes, self.dt))
            nengo.Connection(net.input_node, net.pre.neurons,
                             transform=np.eye(self.pre_size),
                             synapse=None)
            net.conn = nengo.Connection(net.pre.neurons, net.post.neurons,
                                        transform=np.eye(self.post_size, self.pre_size)*self.initial_weight,
                                        learning_rule_type=self.learning_rule)
            net.learning_rate_conn = None
            net.learning_rate_node = None
            if self.learning_rate_func is not None:
                net.learning_rate_node = nengo.Node(self.learning_rate_func)
                net.learning_rate_conn = nengo.Connection(net.learning_rate_node,
                                                          net.conn.learning_rule)
        return net

# #######
# Helpers
# #######

def weight_change(model, theta=None, spikes=60):
    net = model.build()
    with net:
        w_p = nengo.Probe(net.conn, 'weights', synapse=None)

    # Disable decoder cache
    _model = nengo.builder.Model(dt=0.001, decoder_cache=NoDecoderCache())
    with nengo.Simulator(net, model=_model, progress_bar=False) as sim:
        if theta is not None:
            sim.signals[sim.model.sig[net.conn.learning_rule]['theta']] = theta

        sim.run((1. / net.freq_hz) * spikes + 0.05, progress_bar=False)
    weights = sim.data[w_p]
    return weights[-1, ...] - weights[0, ...]


def weight_changes(models, theta=None, spikes=60):
    try:
        jobs = [
            pool.apply_async(weight_change, (model, theta, spikes))
            for i, model in enumerate(models)
        ]
        w_diffs = [job.get() for job in jobs]
    except Exception as e:
        if pool is not None:
            print("Could not run in parallel. Reason: %s" % e)
        w_diffs = [weight_change(m, theta, spikes)
                   for i, m in enumerate(models)]
    return np.vstack(w_diffs).ravel()


# ###########
# Experiments
# ###########

def modulation_depth(rates):
    mod_depths = []
    for i in range(rates.shape[1]):
        rates_i = rates[:, i]
        peak_pctile = np.percentile(rates_i, 80)
        med_pctile = np.percentile(rates_i, 50)
        peak_idxs = np.argwhere(rates_i >= peak_pctile)
        med_idxs = np.argwhere(rates_i <= med_pctile)
        mean_peak = np.mean(rates_i[peak_idxs])
        mean_med = np.mean(rates_i[med_idxs])
        mod_depth = (mean_peak - mean_med) ** 2.
        mod_depths.append(mod_depth)
    return mod_depths


def selectivity(learning_rule, pre_spikes, dt=0.001, plot=True, secondary=None, initial_weight=0., learning_rate_func=None):
    if plot:
        net = ConvergentLearnNet(learning_rule, pre_spikes=pre_spikes, pre_size=pre_spikes.shape[1],
                                 dt=dt, initial_weight=initial_weight,
                                 learning_rate_func=learning_rate_func)
        
        plots.stdp_selectivity(net, t_end=net.t_end, secondary=secondary)



def pairs(learning_rule, n=60, theta=None, plot=True, compare='data', secondary=None, initial_weight=0., learning_rate_func=None):
    # Make some plots for t_diff = -10 ms and 10 ms
    if plot:
        plots.stdp_detail(
            PairNet(learning_rule, freq_hz=5, t_diff=-0.01,
                    initial_weight=initial_weight,
                    learning_rate_func=learning_rate_func),
            secondary=secondary,
            theta=theta)
        plots.stdp_detail(
            PairNet(learning_rule, freq_hz=5, t_diff=0.01,
                    initial_weight=initial_weight,
                    learning_rate_func=learning_rate_func),
            secondary=secondary,
            theta=theta)

    # Then generate STDP curve for pairs
    t_diffs = np.linspace(-0.12, 0.12, 42)  # match number of expt
    models = [PairNet(learning_rule, t_diff=t, freq_hz=2.5,
                      initial_weight=initial_weight,
                      learning_rate_func=learning_rate_func) for t in t_diffs]
    w_diffs = weight_changes(models, theta=theta)

    if plot:
        expt = 'stdp_pair' if compare == 'stdp' else 'bipoo'
        plots.stdp_curve(t_diffs, w_diffs, expt=expt)
    return w_diffs


def freq_dependence(
        learning_rule, n=60, theta=None, plot=True, compare='data', initial_weight=0., learning_rate_func=None):
    freqs = data.sjostrom_x

    def w_diffs(t_diff):
        models = [PairNet(learning_rule, t_diff=t_diff, freq_hz=f,
                          initial_weight=initial_weight,
                          learning_rate_func=learning_rate_func) 
                  for f in freqs]
        return weight_changes(models, theta=theta)

    prepost = w_diffs(-0.01)
    postpre = w_diffs(0.01)
    if plot:
        plots.freq_curve(freqs, prepost, postpre, expt=compare)
    return np.hstack([prepost, postpre])


def prepostpre(learning_rule, n=60, theta=None, plot=True, compare='data', initial_weight=0., learning_rate_func=None):
    if plot:
        plots.stdp_detail(PrePostPreNet(
            learning_rule,
            freq_hz=5,
            t_diff1=-0.005,
            t_diff2=0.005,
            initial_weight=initial_weight,
            learning_rate_func=learning_rate_func
        ), theta=theta)

    # Then generate bar plots from Pfister & Gerstner
    args = data.wang_prepostpre_x
    models = [PrePostPreNet(learning_rule, t_diff1=t1, t_diff2=t2, freq_hz=1., 
                            initial_weight=initial_weight,
                            learning_rate_func=learning_rate_func)
              for t1, t2 in args]
    diffs = weight_changes(models, theta=theta)
    if plot:
        expt = 'wang_prepostpre' if compare == 'data' else 'stdp_prepostpre'
        plots.tripletbar(args, diffs, expt=expt)
    return diffs


def postprepost(learning_rule, n=60, theta=None, plot=True, compare='data', initial_weight=0., learning_rate_func=None):
    if plot:
        plots.stdp_detail(PostPrePostNet(
            learning_rule,
            freq_hz=5,
            t_diff1=-0.005,
            t_diff2=0.005,
            initial_weight=initial_weight,
            learning_rate_func=learning_rate_func
        ), theta=theta)

    # Then generate bar plots from Pfister & Gerstner
    args = data.wang_postprepost_x
    models = [PostPrePostNet(learning_rule, t_diff1=t1, t_diff2=t2, freq_hz=1.,
                             initial_weight=initial_weight,
                             learning_rate_func=learning_rate_func)
              for t1, t2 in args]
    diffs = weight_changes(models, theta=theta)
    if plot:
        expt = 'wang_postprepost' if compare == 'data' else 'stdp_postprepost'
        plots.tripletbar(args, diffs, expt=expt)
    return diffs


def all(learning_rule, theta=None, compare='data'):
    pairs(learning_rule, theta=theta, compare=compare)
    quadruplets(learning_rule, theta=theta, compare=compare)
    freq_dependence(learning_rule, theta=theta, compare=compare)
    prepostpre(learning_rule, theta=theta, compare=compare)
    postprepost(learning_rule, theta=theta, compare=compare)



# ##########################
# BCM parameter optimization
# ##########################

def optimize(stdp, func, space, max_evals=100):
    stdp_diffs = func(stdp, plot=False)

    results = {'pre_tau': [], 'post_tau': [], 'theta': [], 'rmse': []}

    def objective(args):
        pre_tau, post_tau, theta = args
        bcm_diffs = func(nengo.BCM(pre_tau=pre_tau, post_tau=post_tau),
                         theta=theta, plot=False)
        # Normlize by stddev
        bcm_diffs = data.scale(bcm_diffs, stdp_diffs)
        err = rmse(stdp_diffs, bcm_diffs)

        results['pre_tau'].append(pre_tau)
        results['post_tau'].append(post_tau)
        results['theta'].append(theta)
        results['rmse'].append(err)

        print("args: %f, %f, %f; rmse=%f" % (pre_tau, post_tau, theta, err))
        return err

    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)
    pre_tau, post_tau, theta = space_eval(space, best)

    plt.figure(figsize=(12, 4))

    def plot_param(param, i):
        plt.subplot(1, 3, i)
        plt.plot(results[param], results['rmse'], 'o')
        plt.ylabel(param)
        plt.locator_params(axis='x', nbins=5)
    plot_param('pre_tau', 1)
    plot_param('post_tau', 2)
    plot_param('theta', 3)
    sns.despine()
    plt.tight_layout()
    func(nengo.BCM(pre_tau=pre_tau, post_tau=post_tau),
         theta=theta, compare='stdp')
    return best


def optimize_pairs(stdp, max_evals=100):
    """Match the STDP curve results of triplet STDP with BCM."""
    space = (
        hp.uniform('pre_tau', low=0.014, high=0.016),
        hp.uniform('post_tau', low=0.015, high=0.025),
        hp.normal('theta', mu=21, sigma=0.1),
    )
    return optimize(stdp, pairs, space, max_evals=max_evals)


def optimize_quadruplets(stdp, max_evals=100):
    """Match the quadruplet results of triplet STDP with BCM."""
    space = (
        hp.uniform('pre_tau', low=0.04, high=0.06),
        hp.uniform('post_tau', low=0.032, high=0.04),
        hp.normal('theta', mu=19.9, sigma=1),
    )
    return optimize(stdp, quadruplets, space, max_evals=max_evals)


def optimize_freq_dependence(stdp, max_evals=100):
    """Match the frequency dependence results of triplet STDP with BCM."""
    space = (
        hp.uniform('pre_tau', low=0.01, high=0.06),
        hp.uniform('post_tau', low=0.005, high=0.02),
        hp.normal('theta', mu=35.2, sigma=2),
    )
    return optimize(stdp, freq_dependence, space, max_evals=max_evals)


def optimize_prepostpre(stdp, max_evals=100):
    """Match the triplet results of triplet STDP with BCM."""
    space = (
        hp.uniform('pre_tau', low=0.001, high=0.01),
        hp.uniform('post_tau', low=0.001, high=0.0015),
        hp.normal('theta', mu=20, sigma=1),
    )
    return optimize(stdp, prepostpre, space, max_evals=max_evals)


def optimize_postprepost(stdp, max_evals=100):
    """Match the triplet results of triplet STDP with BCM."""
    space = (
        hp.uniform('pre_tau', low=0.01, high=0.02),
        hp.uniform('post_tau', low=0.03, high=0.06),
        hp.normal('theta', mu=20, sigma=1),
    )
    return optimize(stdp, postprepost, space, max_evals=max_evals)


try:
    pool = mp.Pool(processes=7)
except Exception as e:
    print("Cannot run in parallel. Reason: %s" % e)
    pool = None
