
import sys, gc
import nengo
import numpy as np
import scipy.interpolate
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo.utils.progress import TerminalProgressBar
from mnist_data import generate_inputs
from srf_autoenc import build_network, run
    
#output_rates = sim.data[p_output_rates]
#plot_spikes(sim.trange(), sim.data[p_inh_rates][0,:])

seed=23

train_image_array, test_labels = generate_inputs(plot=False, dataset='training', seed=seed)

n_x = train_image_array.shape[1]
n_y = train_image_array.shape[2]

n_steps = 100

normed_train_image_array = train_image_array / np.max(train_image_array)
train_data = np.repeat(normed_train_image_array[:30], n_steps, axis=0) * 10.

print(f'train_data shape: {train_data.shape}')
n_outputs=50
n_exc=n_x*n_y
n_inh=100

srf_exc_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_y), indexing='ij')])
srf_inh_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_inh), np.linspace(0, 1, n_inh), indexing='ij')])
srf_output_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_outputs), np.linspace(0, 1, n_outputs), indexing='ij')])

decoder_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_y), indexing='ij')])
decoder_inh_coords = np.column_stack([x.flat for x in np.meshgrid(np.linspace(0, 1, n_inh), np.linspace(0, 1, n_inh), indexing='ij')])

coords_dict = { 'srf_output': srf_output_coords,
                'srf_exc': srf_exc_coords,
                'srf_inh': srf_inh_coords,
                'decoder': decoder_coords,
                'decoder_inh': decoder_inh_coords }
                
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


dt = 0.01
model_dict = build_network(params, inputs=train_data,
                           coords=coords_dict,
                           seed=seed, dt=dt)
t_end = float(train_data.shape[0]) * dt
output_dict = run(model_dict, t_end, dt=dt, save_results=True)
