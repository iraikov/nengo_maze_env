


import sys, gc
import nengo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.interpolate
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo.utils.progress import TerminalProgressBar
from nengo.dists import Choice, Uniform
from nengo_extras.dists import Tile
from nengo_extras.matplotlib import tile
from mnist_data import generate_inputs
from srf_autoenc import build_network, run
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from inputs import Mask, SequentialMask, Gabor, OOCS
from decoding import predict_ngram, fit_ngram_decoder, fit_rate_decoder, predict_rate

plt.rcParams['font.size'] = 18


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
        a = len(np.argwhere(rates_i >= 0.1))
        bin_fraction_active.append(float(a) / float(n))
        #logger.info(f"fraction_active {i}: a: {a} n: {n} fraction: {float(a) / float(n)}")
    return bin_fraction_active


seed=23
presentation_time=0.06
pause_time=0.06
skip_time=0.0

train_size=100
test_size=40

train_image_array, train_labels = generate_inputs(plot=False, train_size=train_size, dataset='train', seed=seed)
#test_image_array, test_labels = generate_inputs(plot=True, test_size=test_size, dataset='test', seed=seed)
test_image_array, test_labels = train_image_array[:test_size], train_labels[:test_size]

n_labels = len(np.unique(train_labels))

n_x = train_image_array.shape[1]
n_y = train_image_array.shape[2]

train_image_array = np.concatenate((train_image_array[0].reshape(1,n_x,n_y), train_image_array), axis=0)
train_labels = np.concatenate((np.asarray([train_labels[0]]), train_labels))


normed_train_image_array = train_image_array / np.linalg.norm(train_image_array, axis=(1,2))[:,np.newaxis,np.newaxis]
normed_test_image_array = test_image_array / np.linalg.norm(test_image_array, axis=(1,2))[:,np.newaxis,np.newaxis]


reg_input = LogisticRegression(multi_class="multinomial", solver="saga", tol=0.01, penalty='l1')
reg_input = reg_input.fit(normed_train_image_array.reshape((normed_train_image_array.shape[0], -1)), train_labels)
print(f'reg model train score: {reg_input.score(normed_train_image_array.reshape((normed_train_image_array.shape[0], -1)), train_labels)}')
print(f'reg model test score: {reg_input.score(normed_test_image_array.reshape((normed_test_image_array.shape[0], -1)), test_labels)}')

print(f'train labels: {train_labels}')
print(f'test labels: {test_labels}')

input_data = np.concatenate((normed_train_image_array, normed_test_image_array), axis=0)

n_outputs=625
n_exc=4000
n_inh=100

srf_exc_coords = np.asarray(range(n_exc)).reshape((n_exc,1)) / n_exc - 0.5
srf_inh_coords = np.asarray(range(n_inh)).reshape((n_inh,1)) / n_inh - 0.5
srf_output_coords = (np.asarray(range(n_outputs)).reshape((n_outputs,1)) / n_outputs) - 0.5

coords_dict = { 'srf_output': srf_output_coords,
                'srf_exc': srf_exc_coords,
                'srf_inh': srf_inh_coords,
                }
                

params = {'w_initial_E': 0.1, 
          'w_initial_EI': 0.005,
          'w_initial_EE': 0.0025,
          'w_initial_I': -0.008, 
          'w_initial_I_DEC_fb': -0.05, 
          'w_EI_Ext': 0.04,
          'w_DEC_E': 0.005, 
          'w_DEC_I': 0.002, 
          'p_E_srf': 0.2, 
          'p_I_srf': 0.4,
          'p_EE': 0.05, 
          'p_EI': 0.15,
          'p_EI_Ext': 0.3,
          'p_DEC': 0.3, 
          'tau_E': 0.005, 
          'tau_I': 0.010, 
          'tau_input': 0.01,
          'w_input': 1.0,
          'isp_target_rate': 1.0,
          'learning_rate_I': 0.01, 
          'learning_rate_E': 1e-3,
          'learning_rate_EE': 1e-3,
          'learning_rate_D': 0.1,
          'learning_rate_D_Exc': 0.005,
          'sigma_scale_E': 0.004,
          'sigma_scale_EI': 0.002,
          'sigma_scale_EI_Ext': 0.002,
          'sigma_scale_EE': 0.001,
          'sigma_scale_I': 0.002,
          }

dt = 0.01
t_train = (train_size+1)*(presentation_time + pause_time)
t_test = test_size*(presentation_time + pause_time)
t_end = t_train + t_test
print(f't_train: {t_train} t_test: {t_test}')

input_dimensions = n_x*n_y

rng = np.random.RandomState(seed)

# Generate the encoders for the sensory ensemble
#input_encoders = rng.normal(size=(n_exc, 3, 3))
input_encoders = OOCS().generate(n_exc, shape=(8, 8))
print(f'input_encoders.shape = {input_encoders.shape}')
input_encoders = SequentialMask((n_x, n_y)).populate(input_encoders, rng=rng, flatten=True)

print(f'input_encoders.shape = {input_encoders.shape}')
tile(input_encoders.reshape((-1, n_x, n_y)), rows=10, cols=10, grid=True)
plt.show()

model_dict = build_network(params, dimensions=input_dimensions, inputs=input_data,
                           input_encoders=None, direct_input=False,
                           presentation_time=presentation_time, pause_time=pause_time,
                           coords=coords_dict,
                           t_learn_exc=t_train, t_learn_inh=t_end,
                           sample_input_every=0.5,
                           sample_weights_every=t_end // 5.0)
network = model_dict['network']

plt.imshow(network.srf_network.weights_initial_E.T, aspect='auto', interpolation='nearest')
plt.colorbar(label='Synaptic weight')
plt.show()

if network.srf_network.weights_initial_EE is not None:
    plt.imshow(network.srf_network.weights_initial_EE.T, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Synaptic weight')
    plt.show()

plt.imshow(network.srf_network.weights_initial_EI.T, aspect='auto', interpolation='nearest')
plt.colorbar(label='Synaptic weight')
plt.show()

plt.imshow(network.srf_network.weights_initial_I.T, aspect='auto', interpolation='nearest')
plt.colorbar(label='Synaptic weight')
plt.show()

sim, sim_output_dict = run(model_dict, t_end, dt=dt, save_results=False)

n_steps_frame = int((presentation_time + pause_time) / dt)
n_steps_present = int((presentation_time) / dt)
n_steps_skip = int(skip_time / dt)
n_steps_train = n_steps_frame * (train_size+1)
n_steps_test = n_steps_frame * test_size

srf_autoenc_output_spikes_train = sim_output_dict['srf_autoenc_output_spikes'][:n_steps_train]
srf_autoenc_output_spikes_test = sim_output_dict['srf_autoenc_output_spikes'][n_steps_train:]

srf_autoenc_exc_spikes_train = sim_output_dict['srf_autoenc_exc_spikes'][:n_steps_train]
srf_autoenc_exc_spikes_test = sim_output_dict['srf_autoenc_exc_spikes'][n_steps_train:]

srf_autoenc_output_rates_train = sim_output_dict['srf_autoenc_output_rates'][:n_steps_train]
srf_autoenc_exc_rates_train = sim_output_dict['srf_autoenc_exc_rates'][:n_steps_train]
srf_autoenc_inh_rates_train = sim_output_dict['srf_autoenc_inh_rates'][:n_steps_train]
#srf_autoenc_decoder_rates_train = sim_output_dict['srf_autoenc_decoder_rates'][:n_steps_train]

srf_autoenc_output_rates_test = sim_output_dict['srf_autoenc_output_rates'][n_steps_train:]
srf_autoenc_exc_rates_test = sim_output_dict['srf_autoenc_exc_rates'][n_steps_train:]
srf_autoenc_inh_rates_test = sim_output_dict['srf_autoenc_inh_rates'][n_steps_train:]

srf_autoenc_rec_weights = sim_output_dict['srf_autoenc_rec_weights']
srf_autoenc_exc_weights = sim_output_dict['srf_autoenc_exc_weights']
srf_autoenc_inh_weights = sim_output_dict['srf_autoenc_inh_weights']
srf_autoenc_r_exc = sim_output_dict['srf_autoenc_r_exc']

exc_spikes_train = np.split(srf_autoenc_exc_spikes_train[1*n_steps_frame:,:],
                            (n_steps_train - 1*n_steps_frame)//n_steps_frame)
exc_spikes_test = np.split(srf_autoenc_exc_spikes_test, n_steps_test/n_steps_frame)
exc_rates_train = np.split(srf_autoenc_exc_rates_train[1*n_steps_frame:,:],
                           (n_steps_train - 1*n_steps_frame)//n_steps_frame)
exc_rates_test = np.split(srf_autoenc_exc_rates_test, n_steps_test/n_steps_frame)

ngram_n = 2
ngram_decoder_exc = fit_ngram_decoder(exc_spikes_train, train_labels[1:], n_labels, ngram_n, {})
exc_predictions_train = predict_ngram(exc_spikes_train, ngram_decoder_exc, n_labels, ngram_n)
exc_train_score = accuracy_score(train_labels[1:], exc_predictions_train)
print(f'exc_predictions_train = {exc_predictions_train} exc_train_score = {exc_train_score}')

exc_predictions_test = predict_ngram(exc_spikes_test, ngram_decoder_exc, n_labels, ngram_n)
exc_test_score = accuracy_score(test_labels, exc_predictions_test)
print(f'exc_predictions_test = {exc_predictions_test} exc_test_score = {exc_test_score}')

example_spikes_train = [x[n_steps_skip:n_steps_present] for x in np.split(srf_autoenc_output_spikes_train[1*n_steps_frame:,:], (n_steps_train - 1*n_steps_frame)//n_steps_frame)]
example_spikes_test = [x[n_steps_skip:n_steps_present] for x in np.split(srf_autoenc_output_spikes_test, n_steps_test/n_steps_frame)]

output_rates_train = np.split(srf_autoenc_output_rates_train[1*n_steps_frame:,:],
                              (n_steps_train - 1*n_steps_frame)//n_steps_frame)
output_rates_test = np.split(srf_autoenc_output_rates_test, n_steps_test/n_steps_frame)

ngram_n = 2
ngram_decoder = fit_ngram_decoder(example_spikes_train, train_labels[1:], n_labels, ngram_n, {})
output_predictions_train = predict_ngram(example_spikes_train, ngram_decoder, n_labels, ngram_n)
output_train_score = accuracy_score(train_labels[1:], output_predictions_train)

rate_decoder, rate_kdt, rate_kdt_matrix = fit_rate_decoder(output_rates_train, train_labels[1:], n_labels, {})

output_predictions_test = predict_ngram(example_spikes_test, ngram_decoder, n_labels, ngram_n)
output_test_score = accuracy_score(test_labels, output_predictions_test)

output_rate_predictions_train = predict_rate(output_rates_train, rate_decoder, rate_kdt, rate_kdt_matrix, n_labels)
output_rate_predictions_test = predict_rate(output_rates_test, rate_decoder, rate_kdt, rate_kdt_matrix, n_labels)
output_rate_train_score = accuracy_score(train_labels[1:], output_rate_predictions_train)
output_rate_test_score = accuracy_score(test_labels, output_rate_predictions_test)
print(f'output_predictions_train = {output_predictions_train} output_train_score = {output_train_score}')
print(f'output_rate_predictions_train = {output_rate_predictions_train} output_rate_train_score = {output_rate_train_score}')
print(f'output_predictions_test = {output_predictions_test} output_test_score = {output_test_score}')
print(f'output_rate_predictions_test = {output_rate_predictions_test} output_rate_test_score = {output_rate_test_score}')

print(f"output modulation depth (train): {np.mean(modulation_depth(srf_autoenc_output_rates_train))}")
print(f"output modulation depth (test): {np.mean(modulation_depth(srf_autoenc_output_rates_test))}")
#print(f"decoder modulation depth: {np.mean(modulation_depth(srf_autoenc_decoder_rates_train))}")

print(f"input fraction active: {np.mean(fraction_active(srf_autoenc_exc_rates_train))}")
print(f"output fraction active (train): {np.mean(fraction_active(srf_autoenc_output_rates_train))}")
print(f"output fraction active (test): {np.mean(fraction_active(srf_autoenc_output_rates_test))}")
#print(f"decoder fraction active: {np.mean(fraction_active(srf_autoenc_decoder_rates_train))}")
#print(f"decoder mse: {np.mean(mse(srf_autoenc_decoder_rates_train, srf_autoenc_exc_rates_train))}")

#print(f"output train score: {output_train_score} test score: {output_test_score}")

im1 = plt.imshow(srf_autoenc_exc_weights[-1].T, aspect="auto", interpolation="nearest", cmap='jet')
cbar1 = plt.colorbar(im1, label='Synaptic weights')
plt.show()

if srf_autoenc_rec_weights is not None:
    im1 = plt.imshow(srf_autoenc_rec_weights[-1].T, aspect="auto", interpolation="nearest", cmap='jet')
    cbar1 = plt.colorbar(im1, label='Synaptic weights')
    plt.show()

fig, axs = plt.subplots(2,2)

im1 = axs[0,0].imshow(srf_autoenc_exc_rates_train.T, aspect="auto", interpolation="nearest", cmap='jet')
divider1 = make_axes_locatable(axs[0,0])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1)
axs[0,0].set_xlabel('Time [ms]')
axs[0,0].set_ylabel('Neuron')

im2 = axs[0,1].imshow(srf_autoenc_exc_rates_test.T, aspect="auto", interpolation="nearest", cmap='jet')
#cbar2 = plt.colorbar(im2, cax=axs[0,1], label='Firing rate [Hz]')
axs[0,1].set_xlabel('Time [ms]')
axs[0,1].set_ylabel('Neuron')

im3 = axs[1,0].imshow(srf_autoenc_output_rates_train.T, aspect="auto", interpolation="nearest", cmap='jet')
divider3 = make_axes_locatable(axs[1,0])
axs[1,0].set_xlabel('Time [ms]')
axs[1,0].set_ylabel('Neuron')
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
cbar3 = plt.colorbar(im3, cax=cax3, label='Firing rate [Hz]')

im4 = axs[1,1].imshow(srf_autoenc_output_rates_test.T, aspect="auto", interpolation="nearest", cmap='jet')
axs[1,1].set_xlabel('Time [ms]')
axs[1,1].set_ylabel('Neuron')
#cbar4 = plt.colorbar(im4, cax=axs[1,1], label='Firing rate [Hz]')

plt.show()
