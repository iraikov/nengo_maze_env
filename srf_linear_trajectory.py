

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from scipy.interpolate import Rbf, Akima1DInterpolator
from rbf.pde.nodes import disperse, poisson_disc_nodes
import nengo
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    rates_kernel, rates_isi )
from prf_net import PRF
from linear_trajectory import generate_linear_trajectory, generate_input_rates


arena_margin = 0.25
arena_dimension = 200
arena_extent = arena_dimension * (1. + arena_margin)
vert = np.array([[-arena_extent,-arena_extent],[-arena_extent,arena_extent],
                 [arena_extent,arena_extent],[arena_extent,-arena_extent]])
smp = np.array([[0,1],[1,2],[2,3],[3,0]])
    
arena_res = 5
arena_x = np.arange(-arena_extent, arena_extent, arena_res)
arena_y = np.arange(-arena_extent, arena_extent, arena_res)

arena_xx, arena_yy = np.meshgrid(arena_x, arena_y, indexing='ij')
peak_rate = 1.
nmodules_exc = 4
nmodules_inh = 2

exc_field_width_params = [35.0, 0.32]
exc_field_width  = lambda x: 40. + exc_field_width_params[0] * (np.exp(x / exc_field_width_params[1]) - 1.)
inh_field_width_params = [60.0]
inh_field_width  = lambda x: 100. + (inh_field_width_params[0] * x)

exc_module_field_width_dict = {i : exc_field_width( float(i) / float(nmodules_exc) ) for i in range(nmodules_exc)}
inh_module_field_width_dict = {i : inh_field_width( float(i) / float(nmodules_inh) ) for i in range(nmodules_inh)}
    
print(f"exc_module_field_width_dict: {exc_module_field_width_dict}")

exc_input_nodes_dict, exc_input_groups_dict, exc_input_rates_dict = \
    generate_input_rates((vert,smp), exc_module_field_width_dict, spacing_factor=0.8, peak_rate=peak_rate)
inh_input_nodes_dict, inh_input_groups_dict, inh_input_rates_dict = \
    generate_input_rates((vert,smp), inh_module_field_width_dict, basis_function='inverse', spacing_factor=1.4, peak_rate=peak_rate)

def make_input_rate_matrix(input_rates_dict):

    input_rate_maps = []
    for m in input_rates_dict:
        for i in range(len(input_rates_dict[m])):
            input_rates = exc_input_rates_dict[m][i](arena_xx, arena_yy)
            input_rate_maps.append(input_rates.ravel())
    input_rate_matrix = np.column_stack(input_rate_maps)

    return input_rate_matrix

#exc_input_rate_matrix = make_input_rate_matrix(exc_input_rates_dict)
#inh_input_rate_matrix = make_input_rate_matrix(inh_input_rates_dict)


# In[4]:


diag_trajectory = np.asarray([[-100, -100], [100, 100]])
trj_t, trj_x, trj_y, trj_d = generate_linear_trajectory(diag_trajectory, temporal_resolution=0.001, n_trials=3)
    
exc_trajectory_input_rates = { m: {} for m in exc_input_rates_dict }
inh_trajectory_input_rates = { m: {} for m in inh_input_rates_dict }
exc_trajectory_inputs = []
inh_trajectory_inputs = []
    
for m in exc_input_rates_dict:
    for i in exc_input_rates_dict[m]:
        input_rates = exc_input_rates_dict[m][i](trj_x, trj_y)
        input_rates[np.isclose(input_rates, 0., atol=1e-4, rtol=1e-4)] = 0.
        exc_trajectory_input_rates[m][i] = input_rates
        input_rates_ip = Akima1DInterpolator(trj_t, input_rates)
        exc_trajectory_inputs.append(input_rates_ip)
        
for m in inh_input_rates_dict:
    for i in inh_input_rates_dict[m]:
        input_rates = inh_input_rates_dict[m][i](trj_x, trj_y)
        input_rates[np.isclose(input_rates, 0., atol=1e-4, rtol=1e-4)] = 0.
        inh_trajectory_input_rates[m][i] = input_rates
        input_rates_ip = Akima1DInterpolator(trj_t, input_rates)
        inh_trajectory_inputs.append(input_rates_ip)
            

def plot_input_rates(input_rates_dict):
    for m in input_rates_dict:
        plt.figure()
        arena_map = np.zeros(arena_xx.shape)
        for i in range(len(input_rates_dict[m])):
            input_rates = input_rates_dict[m][i](arena_xx, arena_yy)
            arena_map += input_rates
        plt.pcolor(arena_xx, arena_yy, arena_map, cmap=cm.jet)
        #plt.plot(trj_x, trj_y)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.colorbar()
        plt.show()
        
plot_input_rates(exc_input_rates_dict)
plot_input_rates(inh_input_rates_dict)

# In[5]:


def trajectory_input(trajectory_inputs, t, centered=False):
    if centered:
        result = np.asarray([ 2.*y(t) - 1. for y in trajectory_inputs ])
    else:
        result = np.asarray([ y(t) for y in trajectory_inputs ])
    return result
   
#E_train = 2.*exc_input_rate_matrix - 1.
#I_train = 2.*inh_input_rate_matrix - 1.


# In[ ]:

N_Outputs = 50
N_Exc = len(exc_trajectory_inputs)
N_Inh = len(inh_trajectory_inputs)

seed = 19
srf_network = PRF(exc_input_func = partial(trajectory_input, exc_trajectory_inputs),
                  inh_input_func = partial(trajectory_input, inh_trajectory_inputs),
                  n_excitatory = N_Exc,
                  n_inhibitory = N_Inh,
                  n_outputs = N_Outputs,
                  sigma_scale_E = 0.005,
                  p_EE = 0.02,
                  label="Spatial receptive field network",
                  seed=seed)

with srf_network:
    p_output_spikes = nengo.Probe(srf_network.output.neurons)
    p_inh_weights = nengo.Probe(srf_network.conn_I, 'weights', sample_every=1.0)
    p_exc_weights = nengo.Probe(srf_network.conn_E, 'weights', sample_every=1.0)
    p_rec_weights = nengo.Probe(srf_network.conn_EE, 'weights', sample_every=1.0)
    p_exc_rates = nengo.Probe(srf_network.exc.neurons)
    p_inh_rates = nengo.Probe(srf_network.inh.neurons)

with nengo.Simulator(srf_network, optimize=True) as sim:
    sim.run(0.01)
    
exc_weights = srf_network.conn_E.transform
inh_weights = srf_network.conn_I.transform
rec_weights = srf_network.conn_EE.transform

plt.imshow(np.asarray(exc_weights.sample()).T, aspect="auto", interpolation="nearest")
plt.colorbar()
plt.show()
plt.imshow(np.asarray(rec_weights.sample()).T, aspect="auto", interpolation="nearest")
plt.colorbar()
plt.show()
plt.imshow(np.asarray(inh_weights.sample()).T, aspect="auto", interpolation="nearest")
plt.colorbar()
plt.show()

print(f"t_max = {np.max(trj_t)}")    
with nengo.Simulator(srf_network, optimize=True, dt=0.01) as sim:
    sim.run(np.max(trj_t))


rec_weights = sim.data[p_rec_weights]
exc_weights = sim.data[p_exc_weights]
inh_weights = sim.data[p_inh_weights]
exc_rates = sim.data[p_exc_rates]
inh_rates = sim.data[p_inh_rates]
output_spikes = sim.data[p_output_spikes]
np.save("srf_output_spikes", np.asarray(output_spikes, dtype=np.float32))
np.save("srf_time_range", np.asarray(sim.trange(), dtype=np.float32))
output_rates = rates_kernel(sim.trange(), output_spikes, tau=0.1)
np.save("srf_output_rates", np.asarray(output_rates, dtype=np.float32))
np.save("srf_exc_rates", np.asarray(exc_rates, dtype=np.float32))
np.save("srf_inh_rates", np.asarray(inh_rates, dtype=np.float32))
np.save("exc_weights", np.asarray(exc_weights, dtype=np.float32))
np.save("rec_weights", np.asarray(rec_weights, dtype=np.float32))
np.save("inh_weights", np.asarray(inh_weights, dtype=np.float32))

sorted_idxs = np.argsort(-np.argmax(output_rates[3928:].T, axis=1))
plt.imshow(output_rates[:,sorted_idxs].T, aspect="auto", interpolation="nearest")
plt.colorbar()
plt.show()
#plot_spikes(sim.trange(), sim.data[p_inh_rates][0,:])



