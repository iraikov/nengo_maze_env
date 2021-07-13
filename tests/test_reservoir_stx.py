#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengolib
from nengolib import RLS
from reservoir_stx import NengoReservoirSTX


# In[2]:


# Task parameters
pulse_interval = 1.0
amplitude = 0.1
trials_train = 4
trials_test = 4

ndim = 1
n = 100
learning_rate = 1e-5
seed = 21
tau_probe = 0.05
tau = 0.01

# Pre-computed constants
T_train = trials_train * pulse_interval
T_total = (trials_train + trials_test) * pulse_interval


# In[3]:


with nengo.Network(seed=seed) as model:
    # Input is a random sample every pulse_interval seconds
    rng = np.random.RandomState(seed=seed)
    U = nengo.dists.UniformHypersphere(surface=False).sample(trials_train, ndim, rng=rng)
    Z = nengo.dists.UniformHypersphere(surface=False).sample(trials_train, ndim, rng=rng)
    
    u = nengo.Node(output=nengo.processes.PresentInput(U,  presentation_time=pulse_interval))
    z = nengo.Node(output=nengo.processes.PresentInput(Z,  presentation_time=pulse_interval))
    e = nengo.Node(size_in=1)
    
    learning = nengo.Node(size_in=1, output=lambda t,v: True if t <= T_train else False)

    rsvr = NengoReservoirSTX(n_per_dim = n, dimensions=ndim, learning_rate=learning_rate,
                          tau=tau, tau_learn=tau, 
                          ie = 0.1, scale_inh=50.0, 
                          )
    nengo.Connection(u, rsvr.input, synapse=None)
    nengo.Connection(rsvr.output, e, synapse=0)
    nengo.Connection(z, e, synapse=0, transform=-1)
    nengo.Connection(e, rsvr.train, synapse=None)
    
    nengo.Connection(learning, rsvr.enable_learning, synapse=None)
    


with model:
    # Probes
    p_u = nengo.Probe(u, synapse=tau_probe)
    p_z = nengo.Probe(z, synapse=tau_probe)
    p_output = nengo.Probe(rsvr.output, synapse=tau_probe)

with nengo.Simulator(model) as sim:
    sim.run(1.0)


# In[ ]:




t_train = sim.trange() < T_train
t_test = sim.trange() >= T_train

plt.figure(figsize=(16, 6))
plt.title("Training Output")
plt.plot(sim.trange()[t_train], sim.data[p_output][t_train], label="Rsvr learning")
plt.plot(sim.trange()[t_train], sim.data[p_output][t_train] - sim.data[p_z][t_train], alpha=0.8, label="Learning error")
plt.plot(sim.trange()[t_train], sim.data[p_u][t_train], label="Input", linestyle='--')
plt.plot(sim.trange()[t_train], sim.data[p_z][t_train], label="Output", linestyle='--')

plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
plt.show()

# plt.figure(figsize=(16, 6))
# plt.title("Testing Output")
# plt.plot(sim.trange()[t_test], sim.data[p_output][t_test], label="Rsvr testing")
# plt.plot(sim.trange()[t_test], sim.data[p_u][t_test], label="Input", linestyle='--')
# plt.plot(sim.trange()[t_test], sim.data[p_z][t_test], label="Output", linestyle='--')
# plt.plot(sim.trange()[t_test], sim.data[p_z][t_test] - sim.data[p_u][t_test], alpha=0.8, label="Testing error")
# plt.xlabel("Time (s)")
# plt.ylabel("Output")
# plt.legend()
# plt.show()


# In[ ]:




