import matplotlib.pyplot as plt
import numpy as np
from nengo_extras.neurons import (rates_kernel, rates_isi )
import nengo
from prf_net import PRF

N_Inputs = 200
N_Outputs = 200
N_Exc = N_Inputs
N_Inh = 50

tau_input = 0.1
w_input = 0.001

model = nengo.Network(label="2D Representation")
with model:
    # Our ensemble consists of 100 leaky integrate-and-fire neurons,
    # and represents a 2-dimensional signal
    input_ens = nengo.Ensemble(N_Inputs, dimensions=2)


with model:
    # Create input nodes representing the sine and cosine
    x = nengo.Node(output=np.sin)
    y = nengo.Node(output=np.cos)

with model:
    # The indices in neurons define which dimension the input will project to
    nengo.Connection(x, input_ens[0])
    nengo.Connection(y, input_ens[1])

with model:
    srf_network = PRF(n_excitatory = N_Exc,
                    n_inhibitory = N_Inh,
                    n_outputs = N_Outputs,
                    connect_exc_inh_input = True,
                    p_E = 0.02,
                    p_EE = 0.01,
                    learning_rate_E = 0.05,
                    learning_rate_I = 0.01,
                    tau_E = 0.005,
                    tau_I = 0.020,
                    label="Spatial receptive field network")
    
    nengo.Connection(input_ens.neurons, srf_network.exc.neurons,
                     synapse=nengo.Alpha(tau_input),
                     transform=np.eye(N_Exc) * w_input)


with model:
    x_probe = nengo.Probe(x, "output")
    y_probe = nengo.Probe(y, "output")
    input_probe = nengo.Probe(input_ens, "decoded_output", synapse=0.01)
    input_spikes_probe = nengo.Probe(input_ens.neurons, synapse=0.01)
    output_probe = nengo.Probe(srf_network.output.neurons, synapse=0.01)
    exc_probe = nengo.Probe(srf_network.exc.neurons, synapse=0.01)

    

with nengo.Simulator(model) as sim:
    sim.run(15)

# Plot the decoded output of the ensemble
plt.figure()
plt.plot(sim.trange(), sim.data[input_probe], label="Decoded output")
plt.plot(sim.trange(), sim.data[x_probe], "r", label="X")
plt.plot(sim.trange(), sim.data[y_probe], "k", label="Y")
plt.legend()
plt.xlabel("time [s]")

plt.figure()
input_spikes = sim.data[input_spikes_probe]
input_rates = rates_kernel(sim.trange(), input_spikes, tau=0.1)
plt.imshow(input_rates.T, interpolation="nearest", aspect="auto")
plt.colorbar()
plt.xlabel("time [s]")
plt.ylabel("Unit")

plt.figure()
output_spikes = sim.data[output_probe]
output_rates = rates_kernel(sim.trange(), output_spikes, tau=0.1)
plt.imshow(output_rates.T, interpolation="nearest", aspect="auto")
plt.xlabel("time [s]")
plt.ylabel("Unit")
plt.colorbar()
plt.show()
