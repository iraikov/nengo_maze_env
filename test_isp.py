

import matplotlib.pyplot as plt
import numpy as np
import nengo
from isp import ISP


seed = 21
with nengo.Network(seed=seed) as net:
    a = nengo.Ensemble(10, 1)
    b = nengo.Ensemble(1, 1)
    weights = np.ones((1, 10)) * 1e-3
    conn = nengo.Connection(
        a.neurons, b.neurons, transform=weights, learning_rule_type=ISP()
    )
    
    p = nengo.Probe(conn, "weights")

    with nengo.Simulator(net) as sim:
        sim.run(sim.dt * 5)

plt.figure()
plt.imshow(sim.data[p][0,:,:])
print(sim.data[p][0,:,:])
plt.figure()
plt.imshow(sim.data[p][-1,:,:])
print(sim.data[p][-1,:,:])
plt.show()
    
