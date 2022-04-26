
import sys, gc
import nengo
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo.utils.progress import TerminalProgressBar
from nengo.dists import Choice, Uniform
from nengo_extras.dists import Tile
from nengo_extras.matplotlib import tile
from mnist_data import generate_inputs
from srf_autoenc import build_network, run
from sklearn.linear_model import LogisticRegression
from input_mask import InputMask, Gabor

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


def fit_labels(rates, labels, n_samples, return_model=False, rate_sample_fraction=1.0, model_kwargs={}):
    """
    Assign labels to the neurons based on highest average spiking activity.
    """
    assert (rate_sample_fraction <= 1.0) and (rate_sample_fraction > 0.)
    n_steps, n_neurons = rates.shape
    rate_samples = [x[int(round(n_samples*(1.0 - rate_sample_fraction))):,:]
                    for x in np.split(rates, n_steps//n_samples)]

    X = [np.max(rate, axis=0) for rate in rate_samples]
    y = labels
    if not 'tol' in model_kwargs:
        model_kwargs['tol'] = 0.01
    if not 'C' in model_kwargs:
        model_kwargs['C'] = 1.
    if not 'penalty' in model_kwargs:
        model_kwargs['penalty'] = 'l1'
    reg = LogisticRegression(multi_class="multinomial", solver="saga", **model_kwargs)
    reg = reg.fit(X, y)

    score = reg.score(X, y)
    if return_model:
        return score, reg
    else:
        return score


def predict_labels(model, rates, labels, n_samples, rate_sample_fraction=1.0):

    n_steps, n_neurons = rates.shape
    rate_samples = [x[int(round(n_samples*(1.0 - rate_sample_fraction))):,:]
                    for x in np.split(rates, n_steps//n_samples)]

    X = [np.max(rate, axis=0) for rate in rate_samples]
    y = labels
    score = model.score(X, y)
    prediction = model.predict(X)
    
    return prediction, score



seed=23
n_frame_steps = 100

train_image_array, train_labels = generate_inputs(plot=False, train_size=100, dataset='train', seed=seed)
test_image_array, test_labels = generate_inputs(plot=False, test_size=10, dataset='test', seed=seed)

n_x = train_image_array.shape[1]
n_y = train_image_array.shape[2]

train_image_array = np.concatenate((train_image_array[0].reshape(1,n_x,n_y), train_image_array), axis=0)
train_labels = np.concatenate((np.asarray([train_labels[0]]), train_labels))

normed_train_image_array = train_image_array / np.max(train_image_array)
normed_test_image_array = test_image_array / np.max(test_image_array)
train_data = np.repeat(normed_train_image_array, n_frame_steps, axis=0)
test_data = np.repeat(normed_test_image_array, n_frame_steps, axis=0)

print(f'train_data shape: {train_data.shape}')
print(f'train labels: {train_labels}')
print(f'test_data shape: {test_data.shape}')
print(f'test labels: {test_labels}')

input_data = np.concatenate((train_data, test_data), axis=0)

n_outputs=1000
n_exc=n_x*n_y
n_inh=200

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
          'p_I_srf': 0.5,
          'p_EE': 0.01, 
          'p_EI': 0.3,
          'p_EI_Ext': 0.1,
          'p_DEC': 0.1, 
          'tau_E': 0.0025, 
          'tau_I': 0.010, 
          'tau_input': 0.005,
          'learning_rate_I': 0.01, 
          'learning_rate_E': 0.001,
          'learning_rate_EE': 0.001,
          'learning_rate_D': 0.08,
          'learning_rate_D_Exc': 0.005}

dt = 0.01
t_train = float(train_data.shape[0]) * dt
t_test = float(test_data.shape[0]) * dt
t_end = t_train + t_test
print(f't_train: {t_train} t_test: {t_test}')


gabor_size = (7, 7) # Size of the gabor filter for image inputs
rng = np.random.RandomState(seed)

# Generate the encoders for the sensory ensemble
input_dimensions = n_x*n_y
n_gabor = 9

input_encoders = Gabor(sigma_x=Choice([0.45]),
                       sigma_y=Choice([0.45]),
                       theta=Tile(np.linspace(-np.pi,np.pi,n_gabor)),
                       freq=Tile(np.linspace(0.5,2,n_gabor)),
                       phase=Tile(np.linspace(-np.pi,np.pi,n_gabor)),).generate(n_exc, gabor_size, rng=rng)
input_encoders = InputMask((n_x, n_y)).populate(input_encoders, rng=rng, random_positions=False, flatten=True)

#input_encoders = rng.normal(size=(n_exc, n_x * n_y))

tile(input_encoders.reshape((-1, n_x, n_y)), rows=4, cols=6, grid=True)
plt.show()

model_dict = build_network(params, dimensions=input_dimensions, inputs=input_data,
                           input_encoders=input_encoders, direct_input=False,
                           oob_value=0., coords=coords_dict, t_learn=t_train, dt=dt)
network = model_dict['network']

plt.imshow(network.srf_network.weights_initial_E.T, aspect='auto', interpolation='nearest')
plt.colorbar(label='Synaptic weight')
plt.show()

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

srf_autoenc_output_rates_train = sim_output_dict['srf_autoenc_output_rates'][:train_data.shape[0]]
srf_autoenc_exc_rates_train = sim_output_dict['srf_autoenc_exc_rates'][:train_data.shape[0]]
srf_autoenc_decoder_rates_train = sim_output_dict['srf_autoenc_decoder_rates'][:train_data.shape[0]]
srf_autoenc_exc_weights_train = sim_output_dict['srf_autoenc_exc_weights'][:train_data.shape[0]]

srf_autoenc_output_rates_test = sim_output_dict['srf_autoenc_output_rates'][train_data.shape[0]:]

output_r2, output_reg = fit_labels(srf_autoenc_output_rates_train[n_frame_steps:,:],
                                   train_labels[1:], n_frame_steps, return_model=True)

output_train, train_score = predict_labels(output_reg, srf_autoenc_output_rates_train[n_frame_steps:,:],
                                           train_labels[1:], n_frame_steps)
print(f'output_train = {output_train} train_score = {train_score}')
output_test, test_score = predict_labels(output_reg, srf_autoenc_output_rates_test,
                                         test_labels, n_frame_steps)
print(f'output_test = {output_test} test_score = {test_score}')

print(f"output modulation depth (train): {np.mean(modulation_depth(srf_autoenc_output_rates_train))}")
print(f"output modulation depth (test): {np.mean(modulation_depth(srf_autoenc_output_rates_test))}")
print(f"decoder modulation depth: {np.mean(modulation_depth(srf_autoenc_decoder_rates_train))}")

print(f"input fraction active: {np.mean(fraction_active(srf_autoenc_exc_rates_train))}")
print(f"output fraction active (train): {np.mean(fraction_active(srf_autoenc_output_rates_train))}")
print(f"output fraction active (test): {np.mean(fraction_active(srf_autoenc_output_rates_test))}")
print(f"decoder fraction active: {np.mean(fraction_active(srf_autoenc_decoder_rates_train))}")
print(f"decoder mse: {np.mean(mse(srf_autoenc_decoder_rates_train, srf_autoenc_exc_rates_train))}")
print(f"regression r2: {output_r2}")
#print(f"regression prediction accuracy: {output_accuracy * 100}%") 


plt.imshow(srf_autoenc_exc_weights_train[-1,:,:].T, aspect="auto", interpolation="nearest", cmap='jet')
ax = plt.gca()
plt.colorbar(label='Synaptic weights')
plt.tight_layout()
plt.show()

plt.imshow(srf_autoenc_exc_rates_train.T, aspect="auto", interpolation="nearest", cmap='jet')
ax = plt.gca()
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron')
plt.colorbar(label='Firing rate [Hz]')
plt.tight_layout()
plt.show()

sorted_idxs_output = np.argsort(-np.argmax(srf_autoenc_output_rates_train.T, axis=1))
#plt.imshow(srf_autoenc_output_rates[:,sorted_idxs_output].T, aspect="auto", interpolation="nearest", cmap='jet')
plt.imshow(srf_autoenc_output_rates_train.T, aspect="auto", interpolation="nearest", cmap='jet')
ax = plt.gca()
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron')
plt.colorbar(label='Firing rate [Hz]')
plt.tight_layout()
plt.show()

#plt.imshow(srf_autoenc_output_rates[:,sorted_idxs_output].T, aspect="auto", interpolation="nearest", cmap='jet')
plt.imshow(srf_autoenc_output_rates_test.T, aspect="auto", interpolation="nearest", cmap='jet')
ax = plt.gca()
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron')
plt.colorbar(label='Firing rate [Hz]')
plt.tight_layout()
plt.show()
