import time

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import nengo
import numpy as np
from stdp import STDP

# Network parameters
dim = 16
sim_runtime = 16
# seed = int(time.time())
seed = 1607491550
print("Using seed:", seed)

# Set the random number generator, and generate the input vectors
rng = np.random.RandomState(seed)
v1, v2, v3, v4, v5, v6 = nengo.dists.UniformHypersphere(surface=True).sample(
    6, dim, rng=rng
)
z = np.zeros(dim)

# This file demonstrates how a network of two ensembles can be set up to learn
# vector associations in an unsupervised manner using STDP. The network is presented a
# key-value pair of vectors, and is to learn the association between the key and values.

# The function to determine the vector output of stimulus 1.
# This will be connected to ens_A, and determines the "key" part of the vector
# associations that will be learned by the network. For the first half of the
# simulation, the outputs switch between v1 and v2 every second. For the second
# half of the simulation, one instance of v1 and v2 are presented, followed by 2
# presentations each of v5 and v6, and lastly by one presentation of v1 and v2 again.
# If the network has learned the vector associations to v1 and v2, it should output the
# learned vector outputs only when v1 and v2 are presented.
def stim1(t):
    if t < sim_runtime - 6 or t > sim_runtime - 2:
        if int(t) % 2:
            return v1
        else:
            return v2
    else:
        if int(t) % 2:
            return v5
        else:
            return v6


# The function to determine the vector output of stimulus 2.
# This will be connected to ens_B, and determines the "value" part of the vector
# associations that will be learned by the network. For the first half
# of the simulation the inputs switch between v3 and v4 every second. For the second
# half of the simulation, there is no output so that the network can demonstrate the
# learned associations.
def stim2(t):
    if t < sim_runtime - 8:
        if int(t) % 2:
            return v3
        else:
            return v4
    else:
        return z


with nengo.Network(seed=seed) as model:
    # Create the stimulus nodes
    stim_node1 = nengo.Node(stim1)
    stim_node2 = nengo.Node(stim2)

    # Create the Nengo ensemble objects for ens_A and ens_B populations
    # Each population has 30 * D neurons. Each neuron also has intercepts chosen
    # randomly chosen from an interval from [0.1, 1). The default interval is from
    # (-1, 1), but for this example, the interval is set to [0.1, 1) to ensure that
    # the specific neurons are *only* active when the input vector is aligned with the
    # neuron's encoding vector. This works in concert with the STDP rule, since we
    # want connections to be formed when the input vectors activate the respesective
    # neurons in the ens_A and ens_B populations. Note that if we use the default
    # intervals, the neurons would be active when the input vectors are *exactly*
    # opposite from the neurons encoding vectors, which is not desirable.
    ens_A = nengo.Ensemble(
        30 * dim,
        dimensions=dim,
        intercepts=nengo.dists.Uniform(0.1, 1),
    )
    ens_B = nengo.Ensemble(
        30 * dim,
        dimensions=dim,
        intercepts=nengo.dists.Uniform(0.1, 1),
    )

    # Create connections from stim to ens_A and ens_B
    nengo.Connection(stim_node1, ens_A, synapse=None)
    nengo.Connection(stim_node2, ens_B, synapse=None)

    # Create a connection from ens_A to ens_B, with an STDP learning rule.
    # The learning rate of 5e-8 is tuned for this example, to be used with the
    # parameters (e.g., max_rates, number of neurons, etc.). Changing any of these
    # parameters will probably necessitate modifying the learning rate as well.
    conn = nengo.Connection(
        ens_A.neurons,
        ens_B.neurons,
        transform=np.zeros((ens_A.n_neurons, ens_B.n_neurons)),
        learning_rule_type=STDP(
            learning_rate=5e-8,
            bounds="none",
        ),
    )

    # Probe the stimuli
    p_stim1 = nengo.Probe(stim_node1)
    p_stim2 = nengo.Probe(stim_node2)

    # Probe the ensemble outputs
    p_ens_A = nengo.Probe(ens_A, synapse=0.01)
    p_ens_B = nengo.Probe(ens_B, synapse=0.01)


# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(sim_runtime)

# Make the plots and show
plt.figure(figsize=(12, 10))

# Plot the vector output for stimulus 1
plt.subplot(6, 1, 1)
plt.plot(sim.trange(), sim.data[p_stim1])
plt.xlim(0, sim_runtime)
plt.title("Stim1 vector output")

# Plot the vector output for stimulus 1
plt.subplot(6, 1, 2)
plt.plot(sim.trange(), sim.data[p_stim2])
plt.xlim(0, sim_runtime)
plt.title("Stim2 vector output")

# Plot the vector output for ens_A. This should mirror the vector output of stimulus 1.
plt.subplot(6, 1, 3)
plt.plot(sim.trange(), sim.data[p_ens_A])
plt.xlim(0, sim_runtime)
plt.title("ens_A vector output")

# Plot the vector output for ens_B. This should mirror the vector output for stimulus 2
# for the first half of the simuluation. For the second half of the simulation, the
# output of ens_B should be dependent on the output of ens_A.
plt.subplot(6, 1, 4)
plt.plot(sim.trange(), sim.data[p_ens_B])
plt.xlim(0, sim_runtime)
plt.title("ens_B vector output")

# Plot the vector similarity plots for ens_A. This makes the vector output graphs
# easier to parse.
plt.subplot(6, 1, 5)
plt.plot(sim.trange(), np.dot(sim.data[p_ens_A], v1.T))
plt.plot(sim.trange(), np.dot(sim.data[p_ens_A], v2.T))
plt.plot(sim.trange(), np.dot(sim.data[p_ens_A], v5.T))
plt.plot(sim.trange(), np.dot(sim.data[p_ens_A], v6.T))
plt.xlim(0, sim_runtime)
plt.legend(["v1", "v2", "v5", "v6"], loc="center left")
plt.title("ens_A output similarity")

# Plot the vector similarity plots of stimulus 2 and for ens_B.
plt.subplot(6, 1, 6)
plt.plot(sim.trange(), np.dot(sim.data[p_stim2], v3.T))
plt.plot(sim.trange(), np.dot(sim.data[p_stim2], v4.T))
plt.plot(sim.trange(), np.dot(sim.data[p_ens_B], v3.T))
plt.plot(sim.trange(), np.dot(sim.data[p_ens_B], v4.T))
plt.xlim(0, sim_runtime)
plt.legend(["Stim2 v3", "Stim2 v4", "ens_B v3", "ens_B v4"], loc="center left")
plt.title("Stim2 and ens_B output similarities")

plt.tight_layout()
plt.show()
