

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from scipy.interpolate import Akima1DInterpolator
import nengo
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    rates_kernel, rates_isi )
from random_trajectory import generate_random_trajectory, generate_input_rates
from srf_autoenc import build_network, run


def plot_input_rates(input_rates_dict, path):
    trj_x, trj_y = path
    for m in input_rates_dict:
        plt.figure()
        arena_map = np.zeros(arena_xx.shape)
        for i in range(len(input_rates_dict[m])):
            input_rates = input_rates_dict[m][i](arena_xx, arena_yy)
            arena_map += input_rates
        plt.pcolor(arena_xx, arena_yy, arena_map, cmap=cm.jet)
        plt.plot(trj_x, trj_y)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.colorbar()
        plt.show()


def wrap_oob(ip, ub, oob_value, x):
    if np.isclose(x, 1e-3, 1e-3):
        return oob_value
    else:
        return ip(x)

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
nmodules_exc = 3
nmodules_inh = 1

exc_field_width_params = [35.0, 0.8]
exc_field_width  = lambda x: 40. + exc_field_width_params[0] * (np.exp(x / exc_field_width_params[1]) - 1.)
inh_field_width_params = [60.0]
inh_field_width  = lambda x: 100. + (inh_field_width_params[0] * x)

exc_module_field_width_dict = {i : exc_field_width( float(i) / float(nmodules_exc) ) for i in range(nmodules_exc)}
inh_module_field_width_dict = {i : inh_field_width( float(i) / float(nmodules_inh) ) for i in range(nmodules_inh)}
    

exc_input_nodes_dict, exc_input_groups_dict, exc_input_rates_dict = \
    generate_input_rates((vert,smp), exc_module_field_width_dict,
                         spacing_factor=[ 0.015*exc_field_width( (float(i) / float(nmodules_exc)) ) for i in range(nmodules_exc) ],
                         peak_rate=peak_rate)
inh_input_nodes_dict, inh_input_groups_dict, inh_input_rates_dict = \
    generate_input_rates((vert,smp), inh_module_field_width_dict, basis_function='inverse',
                         spacing_factor=1.4,
                         peak_rate=peak_rate)


trj_t, trj_x, trj_y, trj_d = generate_random_trajectory((vert,smp), spacing=15.0, max_distance=10000, temporal_resolution=0.001, n_trials=3)
t_end = np.max(trj_t)
    
exc_trajectory_input_rates = { m: {} for m in exc_input_rates_dict }
inh_trajectory_input_rates = { m: {} for m in inh_input_rates_dict }
exc_trajectory_inputs = []
inh_trajectory_inputs = []

n_exc_inputs=0
for m in exc_input_rates_dict:
    for i in exc_input_rates_dict[m]:
        n_exc_inputs += 1

n_inh_inputs=0
for m in inh_input_rates_dict:
    for i in inh_input_rates_dict[m]:
        n_inh_inputs += 1

        
#oob_exc_input = 0.001*np.ones((n_exc_inputs,))
#oob_inh_input = 0.001*np.ones((n_inh_inputs,))

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
   
plot_input_rates(exc_input_rates_dict, (trj_x, trj_y))
plot_input_rates(inh_input_rates_dict, (trj_x, trj_y))


seed = 19

n_outputs=50
n_exc=len(exc_trajectory_inputs)
n_inh=100
                
params = {'w_initial_E': 0.01, 
          'w_initial_EI': 0.012681074, 
          'w_initial_I': -0.028862255, 
          'w_EI_Ext': 0.02956312, 
          'w_DEC_E': 0.005, 
          'w_DEC_I': 0.001, 
          'p_E_srf': 0.2, 
          'p_EE': 0.01, 
          'p_EI': 0.1,
          'p_EI_Ext': 0.007, 
          'p_DEC': 0.2, 
          'tau_E': 0.005, 
          'tau_I': 0.020, 
          'tau_input': 0.1,
          'learning_rate_I': 0.01, 
          'learning_rate_E': 0.04596530}
                
params = {'w_initial_E': 0.1, 
          'w_initial_EI': 1e-3,
          'w_initial_I': -0.01, 
          'w_EI_Ext': 1e-3,
          'w_DEC_E': 0.005, 
          'w_DEC_I': 0.002, 
          'p_E_srf': 0.2, 
          'p_EE': 0.01, 
          'p_EI': 0.1,
          'p_EI_Ext': 0.007, 
          'p_DEC': 0.2, 
          'tau_E': 0.005, 
          'tau_I': 0.020, 
          'tau_input': 0.1,
          'learning_rate_I': 0.001, 
          'learning_rate_E': 0.004,
          'learning_rate_D': 0.01}


dt = 0.01
model_dict = build_network(params, inputs=exc_trajectory_inputs, 
                           n_outputs=n_outputs, n_exc=n_exc, n_inh=n_inh, n_inh_decoder=n_inh,
                           coords=None, seed=seed)
print(f"t_end = {t_end}")
results = run(model_dict, t_end, dt=dt, save_results=True)
