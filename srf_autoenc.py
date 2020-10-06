from tqdm import tqdm
import glob
from functools import partial
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from scipy.interpolate import Rbf, PchipInterpolator
import nengo
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    rates_kernel, rates_isi )
from prf_net import PRF
import tensorflow as tf
# In[2]:


xdim = 128
ydim = 96

coords_data = []
time_data = []
with open("./rgbd_dataset_freiburg1_xyz/groundtruth.txt", 'r') as f:
    for line in f.readlines():
        if line.startswith('#'):
            continue
        items = line.split(' ')
        t = float(items[0])
        tx, ty, tz = map(float, items[1:4])
        time_data.append(t)
        coords_data.append([tx, ty, tz])

frame_time_data = []
with open("./rgbd_dataset_freiburg1_xyz/rgb.txt", 'r') as f:
    for line in f.readlines():
        if line.startswith('#'):
            continue
        items = line.split(' ')
        t = float(items[0])
        frame_time_data.append(t)

input_time_array = np.asarray(frame_time_data)        
input_time_array = (input_time_array - input_time_array[0])

list_data = glob.glob("./rgbd_dataset_freiburg1_xyz/rgb/*.png")

n_input = len(list_data)
all_data = np.zeros((n_input,xdim,ydim))
for i in range(n_input):
    data_frame = np.array(Image.open(list_data[i]).resize((xdim,ydim)).convert('LA'))/255.
    all_data[i,:,:] = data_frame[:,:,0].T

vae = tf.keras.models.load_model("vae_freiburg_dataset1")

_, _, encoded_input = vae.encoder.predict(all_data)
encoded_input_min = np.min(encoded_input)
if encoded_input_min < 0:
    encoded_input = encoded_input - encoded_input_min

def generate_input_ip(time_array, input_array, peak_rate=1., basis_function='gaussian', n_trials=1, nsample=None, sample_seed=0, trial_dt=0.001):

    trial_ts = []
    t_end = np.max(time_array)
    for i in range(n_trials):
        trial_t = np.copy(time_array)
        trial_t += i*(t_end + trial_dt)
        trial_ts.append(trial_t)
    all_t = np.concatenate(trial_ts)

    input_ip_dict = {}
    norm_input = input_array / input_array.max()
    if nsample is None:
        idxs = range(input_array.shape[1])
    else:
        local_random = np.random.default_rng(sample_seed)
        idxs = local_random.choice(range(input_array.shape[1]), size=nsample, replace=False)
    for i in tqdm(idxs):
        
        u_obs = np.tile(norm_input[:, i] * peak_rate, n_trials)
        #input_rate_ip  = Rbf(all_t, u_obs, function=basis_function)
        input_rate_ip  = PchipInterpolator(all_t, u_obs)
        input_ip_dict[i] = input_rate_ip
            
    return input_ip_dict


def generate_const_input_ip(time_array, n, peak_rate=1., basis_function='inverse', n_trials=1):

    dt = 0.001
    trial_ts = []
    t_end = np.max(time_array)
    for i in range(n_trials):
        trial_t = np.copy(time_array)
        trial_t += i*(t_end + dt)
        trial_ts.append(trial_t)
    all_t = np.concatenate(trial_ts)

    input_ip_dict = {}
    for i in tqdm(range(n)):
        
        u_obs = np.ones((len(all_t),)) * peak_rate
        
        #input_rate_ip  = Rbf(all_t, u_obs, function=basis_function)
        input_rate_ip  = PchipInterpolator(all_t, u_obs)
        input_ip_dict[i] = input_rate_ip
            
    return input_ip_dict


n_trials = 3
peak_rate = 2.
n_inh = 50
print("Generating inputs...")
exc_input_ip_dict = generate_input_ip(input_time_array, encoded_input, peak_rate=peak_rate, n_trials=n_trials)
#inh_input_ip_dict = generate_const_input_ip(input_time_array, n_inh, peak_rate=peak_rate, n_trials=n_trials)
inh_input_ip_dict = generate_input_ip(input_time_array, encoded_input, peak_rate=peak_rate, n_trials=n_trials, nsample=n_inh)


exc_trajectory_inputs = []
inh_trajectory_inputs = []

t_end = np.max(input_time_array) * n_trials

for i in sorted(exc_input_ip_dict.keys()):
    input_rates_ip = exc_input_ip_dict[i]
    exc_trajectory_inputs.append(input_rates_ip)
        
for i in sorted(inh_input_ip_dict.keys()):
    input_rates_ip = inh_input_ip_dict[i]
    inh_trajectory_inputs.append(input_rates_ip)
            

def plot_input_rates(input_rate_ips, t_end, dt=0.001):
    sim_t = np.arange(0., t_end, dt)
    rates = []
    for input_rate_ip in input_rate_ips:
        rate = input_rate_ip(sim_t)
        rates.append(rate)
    rate_matrix = np.column_stack(rates)
    plt.figure()
    plt.imshow(rate_matrix.T, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.show()
        
plot_input_rates(exc_trajectory_inputs, t_end)
#plot_input_rates(inh_input_rates_dict)

# In[5]:


def trajectory_input(trajectory_inputs, t, centered=False):
    if centered:
        result = np.asarray([ 2.*y(t) - 1. for y in trajectory_inputs ])
    else:
        result = np.asarray([ y(t) for y in trajectory_inputs ])
    return np.clip(result, 0., None)


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
                  w_initial_E = 1e-2,
                  label="Spatial receptive field network",
                  seed=seed)

with srf_network:
    p_output_spikes = nengo.Probe(srf_network.output.neurons, 'spikes', synapse=None)
    p_inh_weights = nengo.Probe(srf_network.conn_I, 'weights')
    p_exc_weights = nengo.Probe(srf_network.conn_E, 'weights')
    p_rec_weights = nengo.Probe(srf_network.conn_EE, 'weights')
        
with nengo.Simulator(srf_network, optimize=True) as sim:
    sim.run(np.max(t_end))
    
output_spikes = sim.data[p_output_spikes]
np.save("srf_output_spikes", np.asarray(output_spikes, dtype=np.float32))
np.save("srf_time_range", np.asarray(sim.trange(), dtype=np.float32))
output_rates = rates_kernel(sim.trange(), output_spikes, tau=0.1)
sorted_idxs = np.argsort(-np.argmax(output_rates[53144:].T, axis=1))

#output_rates = sim.data[p_output_rates]
#plot_spikes(sim.trange(), sim.data[p_inh_rates][0,:])



