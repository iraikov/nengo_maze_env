from functools import partial
import numpy as np
import nengo

N = 10

def comparator_func(t, x):
    R1 = np.correlate(x[:N], x[N:])
    print(R1)
    return R1

with nengo.Network() as net:
    ens1 = nengo.Ensemble(10, dimensions=1, seed=0,
                          intercepts=nengo.dists.Choice([-0.1]),
                          max_rates=nengo.dists.Choice([100]))
    ens2 = nengo.Ensemble(10, dimensions=1, seed=0,
                          intercepts=nengo.dists.Choice([-0.1]),
                          max_rates=nengo.dists.Choice([100]))

    node = nengo.Node(size_in=20, output=comparator_func)
    
    # Neuron to neuron
    weights = np.eye(ens1.n_neurons, ens1.n_neurons)
    nengo.Connection(ens1.neurons, node[:10], transform=weights)
    nengo.Connection(ens2.neurons, node[10:], transform=weights)

with nengo.Simulator(net) as sim:
    sim.run(0.1)
    
