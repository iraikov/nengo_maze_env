
import sys, gc
import nengo
import numpy as np
import scipy.interpolate
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo.utils.progress import TerminalProgressBar
from mnist_data import generate_inputs
from srf_autoenc import build_network, run
import matplotlib.pyplot as plt

#output_rates = sim.data[p_output_rates]
#plot_spikes(sim.trange(), sim.data[p_inh_rates][0,:])

seed=23

train_image_array, test_labels = generate_inputs(plot=False, dataset='training', seed=seed)

n_x = train_image_array.shape[1]
n_y = train_image_array.shape[2]

n_steps = 100

normed_train_image_array = train_image_array / np.max(train_image_array)
train_data = np.repeat(normed_train_image_array[:30], n_steps, axis=0)
print(f'train_data shape: {train_data.shape}')

n_outputs=50
n_exc=n_x*n_y
n_inh=100

srf_exc_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_y), indexing='ij')])
srf_inh_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_inh), np.linspace(0, 1, n_inh), indexing='ij')])
srf_output_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_outputs), np.linspace(0, 1, n_outputs), indexing='ij')])

print(f'n_outputs = {n_outputs} srf_output_coords = {srf_output_coords.shape}')
decoder_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_y), indexing='ij')])
decoder_inh_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_inh), np.linspace(0, 1, n_inh), indexing='ij')])

coords_dict = { 'srf_output': srf_output_coords,
                'srf_exc': srf_exc_coords,
                'srf_inh': srf_inh_coords,
                'decoder': decoder_coords,
                'decoder_inh': decoder_inh_coords }
                

params = {'w_initial_E': 0.01, 
          'w_initial_EI': 1e-3,
          'w_initial_EE': 0.01, 
          'w_initial_I': -0.01, 
          'w_initial_I_DEC_fb': -0.05, 
          'w_EI_Ext': 1e-3,
          'w_DEC_E': 0.005, 
          'w_DEC_I': 0.002, 
          'p_E_srf': 0.05, 
          'p_EE': 0.05, 
          'p_EI': 0.1,
          'p_EI_Ext': 0.007,
          'p_DEC': 0.3, 
          'tau_E': 0.005, 
          'tau_I': 0.020, 
          'tau_input': 0.1,
          'learning_rate_I': 0.01, 
          'learning_rate_E': 0.001,
          'learning_rate_EE': 1e-5,
          'learning_rate_D': 0.08,
          'learning_rate_D_Exc': 0.005}

dt = 0.01
t_end = float(train_data.shape[0]) * dt
t_end = 1.0
print(f't_end: {t_end}')
model_dict = build_network(params, inputs=train_data, oob_value=0.,
                           coords=coords_dict, dt=dt)
output_dict = run(model_dict, t_end, dt=dt, save_results=True)

