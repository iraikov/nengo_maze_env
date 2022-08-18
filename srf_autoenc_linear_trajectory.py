import random
from functools import partial
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from scipy.interpolate import Akima1DInterpolator
import nengo
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    rates_kernel, rates_isi )
from linear_trajectory import generate_linear_trajectory, generate_input_rates
from srf_autoenc import build_network, run

mpl.rcParams['font.size'] = 18


def mse(rates, target_rates):
    mses = []
    for i in range(rates.shape[1]):
        rates_i = rates[:, i]
        target_rates_i = target_rates[:, i]
        mean_error = np.mean(rates_i - target_rates_i)
        mse = mean_error ** 2.
        mses.append(mse)
        #logger.info(f"modulation_depth {i}: peak_pctile: {peak_pctile} med_pctile: {med_pctile} mod_depth: {mod_depth}")
    return mses
    

    
def modulation_depth(rates):
    mod_depths = []
    for i in range(rates.shape[1]):
        rates_i = rates[:, i]
        peak_pctile = np.percentile(rates_i, 80)
        med_pctile = np.percentile(rates_i, 50)
        peak_idxs = np.argwhere(rates_i >= peak_pctile)
        med_idxs = np.argwhere(rates_i <= med_pctile)
        mean_peak = np.mean(rates_i[peak_idxs])
        mean_med = np.mean(rates_i[med_idxs])
        mod_depth = (mean_peak - mean_med) ** 2.
        mod_depths.append(mod_depth)
        #logger.info(f"modulation_depth {i}: peak_pctile: {peak_pctile} med_pctile: {med_pctile} mod_depth: {mod_depth}")
    return mod_depths


def fraction_active(rates):
    n = rates.shape[1]
    bin_fraction_active = []
    for i in range(rates.shape[0]):
        rates_i = rates[i, :]
        a = len(np.argwhere(rates_i >= 1.0))
        bin_fraction_active.append(float(a) / float(n))
        #logger.info(f"fraction_active {i}: a: {a} n: {n} fraction: {float(a) / float(n)}")
    return bin_fraction_active

def plot_input_rates(input_rates_dict, trajectory=None):
    for m in input_rates_dict:
        plt.figure()
        arena_map = np.zeros(arena_xx.shape)
        for i in range(len(input_rates_dict[m])):
            input_rates = input_rates_dict[m][i](arena_xx, arena_yy)
            arena_map += input_rates
        plt.pcolor(arena_xx, arena_yy, arena_map, cmap=cm.jet)
        if trajectory is not None:
            _, trj_x, trj_y, _ = trajectory
            plt.plot(trj_x, trj_y, linewidth=4.0)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlabel('X position [cm]')
        ax.set_ylabel('Y position [cm]')
        plt.colorbar(label='Firing rate [Hz]')
        plt.tight_layout()
        
        plt.show()


arena_margin = 0.50
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

exc_module_field_width_dict = {i : exc_field_width( float(i) / float(nmodules_exc+2) ) for i in range(nmodules_exc)}
inh_module_field_width_dict = {i : inh_field_width( float(i) / float(nmodules_inh) ) for i in range(nmodules_inh)}

print(f"exc_module_field_width_dict: {exc_module_field_width_dict}")
    
exc_input_nodes_dict, exc_input_groups_dict, exc_input_rates_dict = \
    generate_input_rates((vert,smp), exc_module_field_width_dict,
                         spacing_factor=[ 0.025*exc_field_width( (float(i) / float(nmodules_exc)) ) for i in range(nmodules_exc) ],
                         peak_rate=[peak_rate*(1 - (float(i) / float(nmodules_exc)))  for i in range(nmodules_exc) ])
inh_input_nodes_dict, inh_input_groups_dict, inh_input_rates_dict = \
    generate_input_rates((vert,smp), inh_module_field_width_dict, basis_function='inverse',
                         spacing_factor=1.4, peak_rate=peak_rate)


diag_trajectory = np.asarray([[-100, -100], [100, 100]])
trj_t, trj_x, trj_y, trj_d = generate_linear_trajectory(diag_trajectory, temporal_resolution=0.001, n_trials=2)
    
t_learn = np.max(trj_t)

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
#        exc_trajectory_inputs.append(lambda t: input_rates_ip(t) if t < t_learn else 0.)
        exc_trajectory_inputs.append(input_rates_ip)
        
for m in inh_input_rates_dict:
    for i in inh_input_rates_dict[m]:
        input_rates = inh_input_rates_dict[m][i](trj_x, trj_y)
        input_rates[np.isclose(input_rates, 0., atol=1e-4, rtol=1e-4)] = 0.
        inh_trajectory_input_rates[m][i] = input_rates
        input_rates_ip = Akima1DInterpolator(trj_t, input_rates)
#        inh_trajectory_inputs.append(lambda t: input_rates_ip(t) if t < t_learn else 0.)
        inh_trajectory_inputs.append(input_rates_ip)

        
plot_input_rates(exc_input_rates_dict, (trj_t, trj_x, trj_y, trj_d))
plot_input_rates(inh_input_rates_dict, (trj_t, trj_x, trj_y, trj_d))


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
                
params = {'w_initial_E': 0.01, 
          'w_initial_EI': 1e-3,
          'w_initial_EE': 0.01, 
          'w_initial_I': -0.01, 
          'w_initial_I_DEC_fb': -0.05, 
          'w_EI_Ext': 1e-3,
          'w_RCL': 0.0001, 
          'w_DEC_E': 0.005, 
          'w_DEC_I': 0.002, 
          'p_E_srf': 0.05, 
          'p_I_srf': 0.3, 
          'p_EE': 0.05, 
          'p_EI': 0.1,
          'p_EI_Ext': 0.007,
          'p_DEC': 0.3, 
          'p_RCL': 0.4, 
          'tau_E': 0.005, 
          'tau_I': 0.020, 
          'tau_input': 0.1,
          'learning_rate_I': 0.01, 
          'learning_rate_E': 0.001,
          'learning_rate_EE': 1e-5,
          'learning_rate_D': 0.08,
          'learning_rate_D_Exc': 0.005,
          'learning_rate_RCL': 0.0001}

dt = 0.01
model_dict = build_network(params, inputs=exc_trajectory_inputs, dimensions=1, oob_value=0.,
                           n_outputs=n_outputs, n_exc=n_exc, n_inh=n_inh, n_inh_decoder=n_inh, 
                           coords=None, seed=seed, dt=dt)
t_end = np.max(trj_t)
n_timesteps_trj = int(np.max(trj_t)/dt)

network = model_dict['network']

plt.imshow(network.srf_network.weights_initial_E.T, aspect='auto', interpolation='nearest')
plt.colorbar(label='Synaptic weight')
plt.show()

plt.imshow(network.weights_initial_DEC_E.T, aspect='auto', interpolation='nearest')
plt.colorbar(label='Synaptic weight')
plt.show()

sim, results = run(model_dict, t_end, dt=dt, save_results=True)
srf_autoenc_output_rates = results['srf_autoenc_output_rates']
srf_autoenc_decoder_rates = results['srf_autoenc_decoder_rates']
srf_autoenc_decoder_inh_rates = results['srf_autoenc_decoder_inh_rates']
srf_autoenc_exc_rates = results['srf_autoenc_exc_rates']
srf_autoenc_exc_weights = results['srf_autoenc_exc_weights']

print(f"output modulation depth: {np.mean(modulation_depth(srf_autoenc_output_rates))}")
print(f"decoder modulation depth: {np.mean(modulation_depth(srf_autoenc_decoder_rates))}")

print(f"input fraction active: {np.mean(fraction_active(srf_autoenc_exc_rates))}")
print(f"output fraction active: {np.mean(fraction_active(srf_autoenc_output_rates))}")
print(f"decoder fraction active: {np.mean(fraction_active(srf_autoenc_decoder_rates))}")
print(f"decoder mse: {np.mean(mse(srf_autoenc_decoder_rates, srf_autoenc_exc_rates))}") 

plt.imshow(srf_autoenc_exc_weights[-1,:,:].T, aspect="auto", interpolation="nearest", cmap='jet')
ax = plt.gca()
plt.colorbar(label='Synaptic weights')
plt.tight_layout()
plt.show()

plt.imshow(srf_autoenc_exc_rates.T, aspect="auto", interpolation="nearest", cmap='jet')
ax = plt.gca()
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron')
plt.colorbar(label='Firing rate [Hz]')
plt.tight_layout()
plt.show()
sorted_idxs_output = np.argsort(-np.argmax(srf_autoenc_output_rates[:n_timesteps_trj].T, axis=1))
plt.imshow(srf_autoenc_output_rates[:,sorted_idxs_output].T, aspect="auto", interpolation="nearest", cmap='jet')
ax = plt.gca()
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron')
plt.colorbar(label='Firing rate [Hz]')
plt.tight_layout()
plt.show()
plt.imshow(srf_autoenc_decoder_rates.T, aspect="auto", interpolation="nearest", cmap='jet')
ax = plt.gca()
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron')
plt.colorbar(label='Firing rate [Hz]')
plt.tight_layout()
plt.show()

