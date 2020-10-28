from tqdm import tqdm
import glob
from functools import partial
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from scipy.interpolate import Rbf, PchipInterpolator, Akima1DInterpolator
import nengo
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    rates_kernel, rates_isi )
from prf_net import PRF
import tensorflow as tf
from vlae import flatten_binary_crossentropy 
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

vlae = tf.keras.models.load_model("vlae_dataset_freiburg1",
                                  custom_objects={ 'flatten_binary_crossentropy': flatten_binary_crossentropy })

n_sample = 300
encoder = vlae.get_layer("encoder")
encoded_inputs = encoder.predict(all_data[:n_sample])

def generate_input_ip(time_array, input_arrays, peak_rate=1., basis_function='gaussian', n_trials=1, nsample=None, sample_seed=0, trial_dt=0.001):

    input_ip_dict = {}
    norm_input_arrays = []
    for m, input_array in enumerate(input_arrays):

        norm_input_array = np.copy(input_array)
        if np.min(norm_input_array) < 0.:
            norm_input_array += np.abs(np.min(norm_input_array))
        norm_input_arrays.append(norm_input_array)
            
    max_norm_input_array = np.max([np.max(x) for x in norm_input_arrays])
    for m, norm_input_array in enumerate(norm_input_arrays):
        norm_input_array /= max_norm_input_array
            
    for m, norm_input_array in enumerate(norm_input_arrays):

        trial_ts = []
        t_end = np.max(time_array)
        for i in range(n_trials):
            trial_t = np.copy(time_array)
            trial_t += i*(t_end + trial_dt)
            trial_ts.append(trial_t)
        all_t = np.concatenate(trial_ts)

        this_input_ip_dict = {}
        if nsample is None:
            idxs = range(input_array.shape[1])
        else:
            local_random = np.random.default_rng(sample_seed)
            idxs = local_random.choice(range(input_array.shape[1]), size=nsample, replace=False)
            
        for i in tqdm(idxs):
            u_obs = np.tile(norm_input_array[:,i] * peak_rate, n_trials)
            input_rate_ip  = Akima1DInterpolator(all_t, u_obs)
            this_input_ip_dict[i] = input_rate_ip

        input_ip_dict[m] = this_input_ip_dict
            
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
peak_rate = 1.
n_inh = 50
print("Generating inputs...")
exc_input_ip_dict = generate_input_ip(input_time_array[:n_sample], encoded_inputs, peak_rate=peak_rate, n_trials=n_trials)
#inh_input_ip_dict = generate_input_ip(input_time_array[:n_sample], encoded_inputs, peak_rate=peak_rate, n_trials=n_trials, nsample=n_inh)


exc_trajectory_inputs = []
#inh_trajectory_inputs = []

t_end = np.max(input_time_array[:n_sample]) * n_trials

for m in sorted(exc_input_ip_dict.keys()):
    for i in sorted(exc_input_ip_dict[m].keys()):
        input_rates_ip = exc_input_ip_dict[m][i]
        exc_trajectory_inputs.append(input_rates_ip)

#for m in sorted(inh_input_ip_dict.keys()):
#for i in sorted(inh_input_ip_dict.keys()):
#    input_rates_ip = inh_input_ip_dict[i]
#    inh_trajectory_inputs.append(input_rates_ip)
            

def plot_input_rates(input_rate_ips, t_end, dt=0.001, num=None):
    sim_t = np.arange(0., t_end, dt)
    rates = []
    if num is not None:
        input_rate_ips = input_rate_ips[:num]
    for input_rate_ip in input_rate_ips:
        rate = input_rate_ip(sim_t)
        rates.append(rate)
    rate_matrix = np.column_stack(rates)
    plt.figure()
    plt.imshow(rate_matrix.T, interpolation="nearest", aspect="auto",
               extent=[0., t_end, 0, len(rates)])
    plt.colorbar()
    plt.show()
        
plot_input_rates(exc_trajectory_inputs, t_end)
plot_input_rates(exc_trajectory_inputs, t_end, num=50)
#plot_input_rates(inh_trajectory_inputs, t_end)

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
N_Inh = n_inh

seed = 19
srf_network = PRF(exc_input_func = partial(trajectory_input, exc_trajectory_inputs),
                  #inh_input_func = partial(trajectory_input, inh_trajectory_inputs),
                  connect_exc_inh_input = True,
                  n_excitatory = N_Exc,
                  n_inhibitory = N_Inh,
                  n_outputs = N_Outputs,
                  w_initial_E = 4e-2,
                  w_initial_I = -5e-2,
                  w_initial_EI = 5e-4,
                  w_EI_Ext = 1e-3,
                  p_EI_Ext = 0.25,
                  p_E = 0.3,
                  tau_E = 0.01,
                  tau_I = 0.03,
                  isp_target_rate = 4.0,
                  label="Spatial receptive field network",
                  seed=seed)

with srf_network:
    p_output_spikes = nengo.Probe(srf_network.output.neurons, 'spikes', synapse=None)
    p_exc_rates = nengo.Probe(srf_network.exc.neurons, 'rates')
    p_inh_rates = nengo.Probe(srf_network.inh.neurons, 'rates')
    p_inh_weights = nengo.Probe(srf_network.conn_I, 'weights')
    p_exc_weights = nengo.Probe(srf_network.conn_E, 'weights')
    p_rec_weights = nengo.Probe(srf_network.conn_EE, 'weights')
        
with nengo.Simulator(srf_network, optimize=True) as sim:
    sim.run(np.max(t_end))
    
output_spikes = sim.data[p_output_spikes]
np.save("srf_output_spikes", np.asarray(output_spikes, dtype=np.float32))
np.save("srf_time_range", np.asarray(sim.trange(), dtype=np.float32))
output_rates = rates_kernel(sim.trange(), output_spikes, tau=0.1)
#sorted_idxs = np.argsort(-np.argmax(output_rates[53144:].T, axis=1))

#output_rates = sim.data[p_output_rates]
#plot_spikes(sim.trange(), sim.data[p_inh_rates][0,:])



