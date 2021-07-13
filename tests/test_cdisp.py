

import matplotlib.pyplot as plt
import numpy as np
import nengo
from cdisp import CDISP

def coincidence(t, x):
    d = np.subtract(x[10:], x[:10]) * 0.1
    return d
    
N = 10    
seed = 21
with nengo.Network(seed=seed) as net:
    a = nengo.Ensemble(N, 1)
    b = nengo.Ensemble(N, 1)
    c = nengo.Ensemble(N, 1)
    exc_weights = np.ones((N, N)) * 1e-3
    conn = nengo.Connection(
        a.neurons, b.neurons, transform=exc_weights
    )
    initial_inh_weights = np.ones((N, N)) * -1e-3
    conn = nengo.Connection(
        c.neurons, b.neurons, transform=initial_inh_weights,
        learning_rule_type=CDISP(learning_rate=1e-3, jit=False)
    )
    
    p_weights = nengo.Probe(conn, "weights")
    p_delta = nengo.Probe(conn.learning_rule, "delta")

    coincidence_detection = nengo.Node(size_in=N*2, size_out=N, output=coincidence)
    nengo.Connection(coincidence_detection, conn.learning_rule)
    
    nengo.Connection(a.neurons, coincidence_detection[:N])
    nengo.Connection(b.neurons, coincidence_detection[N:])
    nengo.Connection(a.neurons, c.neurons, transform=exc_weights)

    with nengo.Simulator(net) as sim:
        sim.run(sim.dt * 5)

delta_weights = sim.data[p_delta]        
inh_weights = sim.data[p_weights]
print(inh_weights[0])
print(inh_weights[-1])
plt.figure()
plt.imshow(inh_weights[0], interpolation="nearest")
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(inh_weights[-1], interpolation="nearest")
plt.colorbar()
plt.show()

    
