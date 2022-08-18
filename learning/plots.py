import matplotlib.pyplot as plt
import nengo
import numpy as np
import seaborn as sns
from nengo.cache import NoDecoderCache
from nengo.utils.matplotlib import rasterplot

from learning import data


def align_yaxis(ax1, v1, ax2, v2):
    """Adjust ax2 ylim so that v2 in ax2 is aligned to v1 in ax1."""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


def stdp_selectivity(model, t_end, secondary=None):
    with model.build() as net:
        spike_readout = nengo.Node(size_in=model.pre_size+model.post_size)
        nengo.Connection(net.pre.neurons, spike_readout[:model.pre_size], synapse=None)
        nengo.Connection(net.post.neurons, spike_readout[model.pre_size:], synapse=None)
        sp_p = nengo.Probe(spike_readout, synapse=None)
        w_p = nengo.Probe(net.conn, 'weights', synapse=None)
        secondary_p = None
        if secondary is not None:
            secondary_p = nengo.Probe(net.conn.learning_rule, secondary, synapse=None)
        
    _model = nengo.builder.Model(dt=0.001, decoder_cache=NoDecoderCache())
    sim = nengo.Simulator(net, model=_model)
    sim.run(t_end, progress_bar=False)

    n_plots = 2
    if secondary is not None:
        n_plots += 1
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(n_plots, 1, 1)
    rasterplot(sim.trange(), sim.data[sp_p])
    plt.yticks((1, 2), ("Pre", "Post"))
    plt.xticks(())
    sns.despine(bottom=True, ax=ax)
    ax = plt.subplot(n_plots, 1, 2)
    plt.plot(sim.trange(), sim.data[w_p][..., 0], c='k')
    plt.xlim(right=sim.trange()[-1])
    # plt.yticks((0,))
    plt.xlabel("Time (s)")
    plt.ylabel("$\omega_{ij}$")
    if secondary is not None:
        ax = plt.subplot(n_plots, 1, 3)
        plt.plot(sim.trange(), sim.data[secondary_p][:, 0])
        plt.xlabel("Time (s)")
        plt.ylabel(secondary)
        
    sns.despine(ax=ax)
    plt.tight_layout()

    
def stdp_detail(model, spikes=4, theta=None, secondary=None):
    with model.build() as net:
        both_spikes = nengo.Node(size_in=2)
        nengo.Connection(net.pre.neurons, both_spikes[0], synapse=None)
        nengo.Connection(net.post.neurons, both_spikes[1], synapse=None)
        sp_p = nengo.Probe(both_spikes, synapse=None)
        w_p = nengo.Probe(net.conn, 'weights', synapse=None)
        secondary_p = None
        if secondary is not None:
            secondary_p = nengo.Probe(net.conn.learning_rule, secondary, synapse=None)
        
    _model = nengo.builder.Model(dt=0.001, decoder_cache=NoDecoderCache())
    sim = nengo.Simulator(net, model=_model)
    if theta is not None:
        sim.signals[sim.model.sig[net.conn.learning_rule]['theta']] = theta
    sim.run((1. / net.freq_hz) * spikes + 0.05, progress_bar=False)

    n_plots = 2
    if secondary is not None:
        n_plots += 1
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(n_plots, 1, 1)
    rasterplot(sim.trange(), sim.data[sp_p])
    plt.yticks((1, 2), ("Pre", "Post"))
    plt.xticks(())
    sns.despine(bottom=True, ax=ax)
    ax = plt.subplot(n_plots, 1, 2)
    plt.plot(sim.trange(), sim.data[w_p][..., 0], c='k')
    plt.xlim(right=sim.trange()[-1])
    # plt.yticks((0,))
    plt.xlabel("Time (s)")
    plt.ylabel("$\omega_{ij}$")
    if secondary is not None:
        ax = plt.subplot(n_plots, 1, 3)
        plt.plot(sim.trange(), sim.data[secondary_p][:, 0])
        plt.xlabel("Time (s)")
        plt.ylabel(secondary)
        
    sns.despine(ax=ax)
    plt.tight_layout()


def stdp_curve(t_diffs, w_diffs, expt=None):
    plt.figure(figsize=(8, 6))
    plt.axhline(0, c='k', lw=1)

    if expt == 'wang':
        idx = [data.nearest_idx(t_diffs, v) for v in data.wang_x]
        w_diffs *= data.scale(w_diffs[idx], data.wang_y)
    elif expt == 'bipoo':
        idx = [data.nearest_idx(t_diffs, v) for v in data.bipoo_x]
        w_diffs *= data.scale(w_diffs[idx], data.bipoo_y)
    elif expt == 'stdp_quadruplet':
        w_diffs *= data.scale(w_diffs, data.stdp_quadruplet_y)
    elif expt == 'stdp_pair':
        w_diffs *= data.scale(w_diffs, data.stdp_pair_y)

    def plot_fit(x, y, ls='-'):
        # Plot fitted version
        fit_x, fit_y = data.exponential_fit(x, y)
        pre, post = fit_x < 0, fit_x > 0
        plt.plot(fit_x[pre], fit_y[pre], c='g', ls=ls)
        plt.plot(fit_x[post], fit_y[post], c='r', ls=ls)

    if expt not in ('wang', 'stdp_quadruplet'):
        plot_fit(t_diffs, w_diffs)

    if expt == 'bipoo':
        plot_fit(data.bipoo_x, data.bipoo_y, ls=':')
    elif expt == 'stdp_pair':
        plot_fit(t_diffs, data.stdp_pair_y, ls=':')

    l1, = plt.plot(t_diffs, w_diffs, ls='none', marker='o', mfc='0.2')
    plt.yticks((0,))

    if expt == 'bipoo':
        l2, = plt.plot(data.bipoo_x, data.bipoo_y,
                       ls='none', marker='o', mfc='0.7')
        plt.legend([l1, l2], ["Model", "Data (Bi & Poo)"],
                   loc='best', frameon=True)
    elif expt == 'wang':
        l2 = plt.errorbar(data.wang_x, data.wang_y, yerr=data.wang_yerr,
                          fmt='o', mfc='0.7', ecolor='0.7')[0]
        plt.legend([l1, l2], ["Model", "Data (Wang et al.)"],
                   loc='best', frameon=True)


    if expt in ('wang', 'stdp_quadruplet'):
        plt.xlabel("Spike pair difference [s]")
    else:
        plt.xlabel("$t^{pre} - t^{post}$")
    plt.ylabel("$\Delta \omega_{ij}$")
    sns.despine()
    plt.tight_layout()


def freq_curve(freqs, prepost, postpre, expt):
    plt.figure(figsize=(8, 6))

    if expt == 'data':
        e_prepost = data.sjostrom_prepost
        e_prepost_err = data.sjostrom_preposterr
        e_postpre = data.sjostrom_postpre
        e_postpre_err = data.sjostrom_postpreerr
    elif expt == 'stdp':
        e_prepost, e_prepost_err = data.stdp_freqdep_prepost, None
        e_postpre, e_postpre_err = data.stdp_freqdep_postpre, None

    prepost *= data.scale(prepost, e_prepost)
    postpre *= data.scale(postpre, e_postpre)

    label = "data, Sjostrom"
    plt.errorbar(freqs, e_prepost, yerr=e_prepost_err,
                 marker='o', c='b', ls='--', label="Pre-post (%s)" % label)
    plt.errorbar(freqs, e_postpre, yerr=e_postpre_err,
                 marker='o', c='g', ls='--', label="Post-pre (%s)" % label)

    label = "model"
    plt.plot(freqs, prepost, marker='o', c='b', label="Pre-post (%s)" % label)
    plt.plot(freqs, postpre, marker='o', c='g', label="Post-pre (%s)" % label)

    plt.axhline(0, c='k', lw=1)
    plt.xlabel("Frequency [Hz]")
    plt.xlim(8, 52)
    plt.ylabel("$\Delta \omega_{ij}$")
    plt.legend(loc='best', frameon=True)
    sns.despine()
    plt.tight_layout()


def tripletbar(args, w_diffs, expt):
    b_args = dict(width=0.4, align='center')

    plt.figure(figsize=(8, 6))

    if expt == 'wang_prepostpre':
        y, yerr = data.wang_prepostpre_y, data.wang_prepostpre_yerr
    elif expt == 'wang_postprepost':
        y, yerr = data.wang_postprepost_y, data.wang_postprepost_yerr
    elif expt == 'stdp_prepostpre':
        y, yerr = data.stdp_prepostpre_y, None
    elif expt == 'stdp_postprepost':
        y, yerr = data.stdp_postprepost_y, None

    # Determine scale, scale w_diffs
    w_diffs *= data.scale(w_diffs, y)

    b1 = plt.bar(np.arange(len(args))+0.2, y, yerr=yerr,
                 color='k', ecolor='k', **b_args)[0]
    b2 = plt.bar(np.arange(len(args))+0.6, w_diffs, **b_args)[0]
    plt.axhline(0, c='k', lw=1)
    plt.xticks(np.arange(len(args)) + 0.4,
               ["(%d, %d)" % (int(t1 * 1000), int(t2 * 1000))
                for t1, t2 in args])
    plt.ylabel("$\Delta \omega_{ij}$")
    plt.xlabel("($\Delta t_1$, $\Delta t_2$) [ms]")
    if expt.startswith('wang'):
        plt.legend([b1, b2], ["Data (Wang et al.)", "Model"],
                   loc='best', frameon=True)
    sns.despine()
    plt.tight_layout()
