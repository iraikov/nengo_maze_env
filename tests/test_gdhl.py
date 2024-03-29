
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import nengo
from gdhl import GDHL

sigma = { 'pp': 0.0, 'np': 0., 'pn': 0., 'nn': 0. }
eta = { 'ps': 0., 'ns': 0., 'sp': 1., 'sn': -1. }

seed = 21
with nengo.Network(seed=seed) as net:
    a = nengo.Ensemble(10, 1)
    b = nengo.Ensemble(1, 1)
    weights = np.ones((1, 10)) * 1e-3
    conn = nengo.Connection(
        a.neurons, b.neurons, transform=weights, learning_rule_type=GDHL(sigma=sigma, eta=eta, pre_synapse=nengo.Lowpass(0.005)),
    )
    
    p = nengo.Probe(conn, "weights")
    p_a_spikes = nengo.Probe(a.neurons, "spikes")
    p_b_spikes = nengo.Probe(b.neurons, "spikes")

    with nengo.Simulator(net) as sim:
        sim.run(sim.dt * 100)

plt.figure()
plot_spikes(sim.trange(), sim.data[p_a_spikes])
plt.figure()
plot_spikes(sim.trange(), sim.data[p_b_spikes])
plt.figure()
plt.imshow(sim.data[p][0,:,:])
print(sim.data[p][0,:,:])
plt.figure()
plt.imshow(sim.data[p][-1,:,:])
print(sim.data[p][-1,:,:])
plt.show()
    
