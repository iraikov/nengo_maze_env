

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from scipy.interpolate import Rbf, Akima1DInterpolator
import nengo
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    rates_kernel, rates_isi )
from linear_trajectory import generate_linear_trajectory, generate_input_rates
from srf_autoenc import build_network, run


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


def trajectory_input(trajectory_inputs, t, centered=False):
    if centered:
        result = np.asarray([ 2.*y(t) - 1. for y in trajectory_inputs ])
    else:
        result = np.asarray([ y(t) for y in trajectory_inputs ])
    return result


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
    

exc_input_nodes_dict, exc_input_groups_dict, exc_input_rates_dict = \
    generate_input_rates((vert,smp), exc_module_field_width_dict, spacing_factor=0.8, peak_rate=peak_rate)
inh_input_nodes_dict, inh_input_groups_dict, inh_input_rates_dict = \
    generate_input_rates((vert,smp), inh_module_field_width_dict, basis_function='inverse', spacing_factor=1.4, peak_rate=peak_rate)


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
   


seed = 19

n_outputs=50
n_exc=len(exc_trajectory_inputs)
n_inh=100

#srf_exc_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_y), indexing='ij')])
#srf_inh_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_inh), np.linspace(0, 1, n_inh), indexing='ij')])
#srf_output_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_outputs), np.linspace(0, 1, n_outputs), indexing='ij')])

#decoder_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_y), indexing='ij')])
#decoder_inh_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_inh), np.linspace(0, 1, n_inh), indexing='ij')])

#coords_dict = { 'srf_output': srf_output_coords,
#                'srf_exc': srf_exc_coords,
#                'srf_inh': srf_inh_coords,
#                'decoder': decoder_coords,
#                'decoder_inh': decoder_inh_coords }
                
params = {'w_initial_E': 0.01, 
          'w_initial_EI': 0.012681074, 
          'w_initial_I': -0.028862255, 
          'w_EI_Ext': 0.02956312, 
          'w_PV_E': 0.005, 
          'w_PV_I': 0.001, 
          'p_E_srf': 0.2, 
          'p_EE': 0.01, 
          'p_EI': 0.1,
          'p_EI_Ext': 0.007, 
          'p_PV': 0.2, 
          'tau_E': 0.005, 
          'tau_I': 0.020, 
          'tau_input': 0.1,
          'learning_rate_I': 0.01, 
          'learning_rate_E': 0.04596530}
                
params = {'w_initial_E': 0.1, 
          'w_initial_EI': 1e-3,
          'w_initial_I': -0.01, 
          'w_EI_Ext': 1e-3,
          'w_PV_E': 0.005, 
          'w_PV_I': 0.002, 
          'p_E_srf': 0.2, 
          'p_EE': 0.01, 
          'p_EI': 0.1,
          'p_EI_Ext': 0.007, 
          'p_PV': 0.2, 
          'tau_E': 0.005, 
          'tau_I': 0.020, 
          'tau_input': 0.1,
          'learning_rate_I': 0.001, 
          'learning_rate_E': 0.004}

dt = 0.01
model_dict = build_network(params, inputs=exc_trajectory_inputs,
                           n_outputs=n_outputs, n_exc=n_exc, n_inh=n_inh, n_inh_decoder=n_inh,
                           coords=None, seed=seed)
t_end = np.max(trj_t)
print(f"t_end = {t_end}")
results = run(model_dict, t_end, dt=dt, save_results=True)
