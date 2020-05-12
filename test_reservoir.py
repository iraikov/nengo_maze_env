#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import nengo
from nengo.utils.least_squares_solvers import LSMRScipy
import nengolib
from nengolib import RLS
from reservoir_net import NengoReservoir


# In[2]:


# Task parameters
pulse_interval = 1.0
amplitude = 0.1
freq = 3.0
decay = 2.0
dt = 0.002
trials_train = 3
trials_test = 4

ndim = 20
n = 100
learning_rate = 1e-6
seed = 21
tau_probe = 0.05
tau = 0.05

# Pre-computed constants
T_train = trials_train * pulse_interval
T_total = (trials_train + trials_test) * pulse_interval


# In[3]:


with nengo.Network(seed=seed) as model:
    # Input is a random sample every pulse_interval seconds
    rng = np.random.RandomState(seed=seed)
    U = nengo.dists.UniformHypersphere(surface=False).sample(trials_train+trials_test, ndim, rng=rng)
    
    u = nengo.Node(output=nengo.processes.PresentInput(U,  presentation_time=pulse_interval))

    z = nengo.Node(size_in=ndim)
    nengo.Connection(u, z, synapse=nengo.synapses.Lowpass(tau))
    learning = nengo.Node(size_in=1, output=lambda t,v: True if t < T_train else False)

    rsvr = NengoReservoir(n_per_dim = n, dimensions=ndim, learning_rate=learning_rate, tau=tau)
    nengo.Connection(u, rsvr.input, synapse=None)
    nengo.Connection(learning, rsvr.enable_learning, synapse=None)

    reader = nengo.Ensemble(n*ndim, dimensions=ndim)
    error = nengo.Node(size_in=ndim+1, size_out=ndim,
                       output=lambda t, e: e[1:] if e[0] else 0.)

    rsvr_reader_conn = nengo.Connection(rsvr.ensemble.neurons, reader, synapse=None,
                                        transform=np.zeros((ndim, n*ndim)),
                                        learning_rule_type=RLS(learning_rate=learning_rate,
                                                               pre_synapse=tau))
                                                               

    # Error = actual - target = post - pre
    nengo.Connection(reader, error[1:])
    nengo.Connection(z, error[1:], transform=-1)

    nengo.Connection(learning, error[0])
    # Connect the error into the learning rule
    nengo.Connection(error, rsvr_reader_conn.learning_rule)

    
# In[ ]:


with model:
    # Probes
    p_u = nengo.Probe(u, synapse=tau_probe)
    p_z = nengo.Probe(z, synapse=tau_probe)
    p_error = nengo.Probe(error, synapse=tau_probe)
    p_output = nengo.Probe(reader, synapse=tau_probe)

with nengo.Simulator(model, dt=dt) as sim:
    sim.run(T_total)


# In[ ]:




t_train = sim.trange() < T_train
t_test = sim.trange() >= T_train

solver = nengo.solvers.LstsqL2(solver=LSMRScipy(), reg=0.001)
wF, _ = solver(sim.data[p_u][t_train], sim.data[p_z][t_train])
zF = sim.data[p_u].dot(wF)

plt.figure(figsize=(16, 6))
plt.title("Training Output")
plt.plot(sim.trange()[t_train], sim.data[p_output][t_train], label="Online learning")
plt.plot(sim.trange()[t_train], zF[t_train], label="LSMR")
plt.plot(sim.trange()[t_train], sim.data[p_error][t_train],
           alpha=0.8, label="Sup. error readout")
plt.plot(sim.trange()[t_train], sim.data[p_z][t_train], label="Input", linestyle='--')
plt.plot(sim.trange()[t_train], sim.data[p_u][t_train], label="Ideal", linestyle='--')

plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
plt.show()

plt.figure(figsize=(16, 6))
plt.title("Testing Output")
plt.plot(sim.trange()[t_test], sim.data[p_output][t_test], label="Online learning")
plt.plot(sim.trange()[t_test], zF[t_test], label="LSMR")
plt.plot(sim.trange()[t_test], sim.data[p_z][t_test], label="Input", linestyle='--')
plt.plot(sim.trange()[t_test], sim.data[p_u][t_test], label="Ideal", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
plt.show()

plt.figure(figsize=(16, 6))
plt.title("Testing Error")
plt.plot(sim.trange()[t_test], sim.data[p_output][t_test] - sim.data[p_z][t_test],
           alpha=0.8, label="Online learning")
plt.plot(sim.trange()[t_test], sim.data[p_error][t_test],
           alpha=0.8, label="Sup. error readout")
plt.plot(sim.trange()[t_test], zF[t_test] - sim.data[p_z][t_test],
           alpha=0.8, label="LSMR")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[ ]:




