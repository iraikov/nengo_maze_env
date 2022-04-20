
import sys, gc
import nengo
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo.utils.progress import TerminalProgressBar
from nengo.dists import Choice, Uniform
from nengo_extras.matplotlib import tile
from mnist_data import generate_inputs
from srf_autoenc import build_network, run
from input_mask import InputMask, Gabor

#output_rates = sim.data[p_output_rates]
#plot_spikes(sim.trange(), sim.data[p_inh_rates][0,:])

seed=23

train_image_array, test_labels = generate_inputs(plot=False, train_size=10, dataset='training', seed=seed)

n_x = train_image_array.shape[1]
n_y = train_image_array.shape[2]

n_steps = 100

normed_train_image_array = train_image_array / np.max(train_image_array)
train_data = np.repeat(normed_train_image_array, n_steps, axis=0)
print(f'train_data shape: {train_data.shape}')

n_outputs=50
n_exc=n_x*n_y
n_inh=100

srf_exc_coords = np.asarray(range(n_exc)).reshape((n_exc,1)) / n_exc
srf_inh_coords = np.asarray(range(n_inh)).reshape((n_inh,1)) / n_inh
srf_output_coords = np.asarray(range(n_outputs)).reshape((n_outputs,1)) / n_outputs

coords_dict = { 'srf_output': srf_output_coords,
                'srf_exc': srf_exc_coords,
                'srf_inh': srf_inh_coords,
                }
                

params = {'w_initial_E': 0.01, 
          'w_initial_EI': 0.01,
          'w_initial_EE': 0.01, 
          'w_initial_I': -0.01, 
          'w_initial_I_DEC_fb': -0.05, 
          'w_EI_Ext': 1e-3,
          'w_DEC_E': 0.005, 
          'w_DEC_I': 0.002, 
          'p_E_srf': 0.05, 
          'p_EE': 0.05, 
          'p_EI': 0.2,
          'p_EI_Ext': 0.007,
          'p_DEC': 0.3, 
          'tau_E': 0.0025, 
          'tau_I': 0.010, 
          'tau_input': 0.05,
          'learning_rate_I': 0.01, 
          'learning_rate_E': 0.001,
          'learning_rate_EE': 1e-3,
          'learning_rate_D': 0.08,
          'learning_rate_D_Exc': 0.005}

dt = 0.01
t_end = float(train_data.shape[0]) * dt
print(f't_end: {t_end}')


gabor_size = (5, 5)  # Size of the gabor filter for image inputs
rng = np.random.RandomState(seed)

# Generate the encoders for the sensory ensemble
input_dimensions = n_x*n_y

input_encoders = Gabor(freq=Uniform(0.5,2),phase=Uniform(-4,4)).generate(n_exc, gabor_size, rng=rng)
input_encoders = InputMask((n_x, n_y)).populate(input_encoders, rng=rng, random_positions=False, flatten=True)

tile(input_encoders.reshape((-1, n_x, n_y)), rows=4, cols=6, grid=True)
plt.show()

model_dict = build_network(params, dimensions=input_dimensions, inputs=train_data,
                           input_encoders=input_encoders, direct_input=False,
                           oob_value=0., coords=coords_dict, dt=dt)
output_dict = run(model_dict, t_end, dt=dt, save_results=False)

srf_autoenc_output_rates = output_dict[1]['srf_autoenc_output_rates']
srf_autoenc_exc_rates = output_dict[1]['srf_autoenc_exc_rates']
