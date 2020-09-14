

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from scipy.interpolate import Rbf, make_interp_spline
from rbf.pde.nodes import disperse, poisson_disc_nodes
import nengo
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    NumbaLIF, rates_kernel, rates_isi )
from prf_net import PRF
from hsp import HSP
from rmsp import RMSP

def consecutive(data):
    """
    Returns a list of arrays with consecutive values from data.
    """
    return np.split(data, np.where(np.diff(data) != 1)[0]+1)


def generate_linear_trajectory(input_trajectory, temporal_resolution=1., velocity=30., reward_pos=None, reward_delay=0.5, equilibration_duration=None, n_trials=1):
    """
    Construct coordinate arrays for a spatial trajectory, considering run velocity to interpolate at the specified
    temporal resolution. Optionally, the trajectory can be prepended with extra distance traveled for a specified
    network equilibration time, with the intention that the user discards spikes generated during this period before
    analysis.

    :param input_trajectory: list of positions
    :param temporal_resolution: float (s)
    :param equilibration_duration: float (s)
    :return: tuple of array
    """

    trajectory_lst = []
    for i_trial in range(n_trials):
        trajectory_lst.append(input_trajectory)

    trajectory = np.concatenate(trajectory_lst)
        
    velocity = velocity  # (cm / s)
    spatial_resolution = velocity * temporal_resolution
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    
    if equilibration_duration is not None:
        equilibration_distance = velocity / equilibration_duration
        x = np.insert(x, 0, x[0] - equilibration_distance)
        y = np.insert(y, 0, y[0])
    else:
        equilibration_duration = 0.
        equilibration_distance = 0.
    
    segment_lengths = np.sqrt((np.diff(x) ** 2. + np.diff(y) ** 2.))
    distance = np.insert(np.cumsum(segment_lengths), 0, 0.)
    
    interp_distance = np.arange(distance.min(), distance.max() + spatial_resolution / 2., spatial_resolution)
    interp_x = np.interp(interp_distance, distance, x)
    interp_y = np.interp(interp_distance, distance, y)

    t = interp_distance / velocity  # s

    if reward_pos is not None:
        reward_x = reward_pos[0]
        reward_y = reward_pos[1]
        reward_locs = np.argwhere(np.logical_and(np.isclose(interp_x, reward_x, atol=1e-2, rtol=1e-2),
                                                 np.isclose(interp_y, reward_y, atol=1e-2, rtol=1e-2)))
        reward_loc_ranges = consecutive(reward_locs.flat)
        for loc_array in reward_loc_ranges[:-1]:
            loc = loc_array[0]
            t[loc:] += reward_delay

    t = np.subtract(t, equilibration_duration)
    interp_distance -= equilibration_distance
    
    return t, interp_x, interp_y, interp_distance



# In[3]:

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
reward_peak_rate = 3.
nmodules_exc = 4
nmodules_inh = 2

exc_field_width_params = [35.0, 0.32]
exc_field_width  = lambda x: 40. + exc_field_width_params[0] * (np.exp(x / exc_field_width_params[1]) - 1.)
inh_field_width_params = [60.0]
inh_field_width  = lambda x: 100. + (inh_field_width_params[0] * x)

exc_module_field_width_dict = {i : exc_field_width( float(i) / float(nmodules_exc) ) for i in range(nmodules_exc)}
inh_module_field_width_dict = {i : inh_field_width( float(i) / float(nmodules_inh) ) for i in range(nmodules_inh)}
    

def generate_input_rates(module_field_width_dict, basis_function='gaussian', spacing_factor=1.0, peak_rate=1.):
    input_nodes_dict = {}
    input_groups_dict = {}
    input_rates_dict = {}
    
    for m in module_field_width_dict:
        nodes, groups, _ = poisson_disc_nodes(module_field_width_dict[m], (vert, smp))
        input_groups_dict[m] = groups
        input_nodes_dict[m] = nodes
        input_rates_dict[m] = {}
        
        for i in range(nodes.shape[0]):
            xs = [[nodes[i,0], nodes[i,1]]]
            x_obs = np.asarray(xs).reshape((1,-1))
            u_obs = np.asarray([[peak_rate]]).reshape((1,-1))
            if basis_function == 'constant':
                input_rate_ip  = lambda xx, yy: xx, yy
            else:
                input_rate_ip  = Rbf(x_obs[:,0], x_obs[:,1], u_obs,
                                    function=basis_function, 
                                    epsilon=module_field_width_dict[m] * spacing_factor)
            input_rates_dict[m][i] = input_rate_ip
            
    return input_nodes_dict, input_groups_dict, input_rates_dict

exc_input_nodes_dict, exc_input_groups_dict, exc_input_rates_dict = \
    generate_input_rates(exc_module_field_width_dict, spacing_factor=0.8, peak_rate=peak_rate)
inh_input_nodes_dict, inh_input_groups_dict, inh_input_rates_dict = \
    generate_input_rates(inh_module_field_width_dict, basis_function='inverse', spacing_factor=1.4, peak_rate=peak_rate)

def make_input_rate_matrix(input_rates_dict):

    input_rate_maps = []
    for m in input_rates_dict:
        for i in range(len(input_rates_dict[m])):
            input_rates = exc_input_rates_dict[m][i](arena_xx, arena_yy)
            input_rate_maps.append(input_rates.ravel())
    input_rate_matrix = np.column_stack(input_rate_maps)

    return input_rate_matrix

#exc_input_rate_matrix = make_input_rate_matrix(exc_input_rates_dict)
#inh_input_rate_matrix = make_input_rate_matrix(inh_input_rates_dict)


# In[4]:

reward_pos = [0., 0.]
diag_trajectory = np.asarray([[-100, -100], [100, 100]])
trj_t, trj_x, trj_y, trj_d = generate_linear_trajectory(diag_trajectory, temporal_resolution=0.001,
                                                        reward_pos=reward_pos, n_trials=3)
exc_trajectory_input_rates = { m: {} for m in exc_input_rates_dict }
inh_trajectory_input_rates = { m: {} for m in inh_input_rates_dict }
exc_trajectory_inputs = []
inh_trajectory_inputs = []
    
for m in exc_input_rates_dict:
    for i in exc_input_rates_dict[m]:
        input_rates = exc_input_rates_dict[m][i](trj_x, trj_y)
        input_rates[np.isclose(input_rates, 0., atol=1e-4, rtol=1e-4)] = 0.
        exc_trajectory_input_rates[m][i] = input_rates
        input_rates_ip = make_interp_spline(trj_t, input_rates, k=3)
        exc_trajectory_inputs.append(input_rates_ip)
        
for m in inh_input_rates_dict:
    for i in inh_input_rates_dict[m]:
        input_rates = inh_input_rates_dict[m][i](trj_x, trj_y)
        input_rates[np.isclose(input_rates, 0., atol=1e-4, rtol=1e-4)] = 0.
        inh_trajectory_input_rates[m][i] = input_rates
        input_rates_ip = make_interp_spline(trj_t, input_rates, k=3)
        inh_trajectory_inputs.append(input_rates_ip)

reward_idxs = np.argwhere(np.logical_and(np.isclose(trj_x, reward_pos[0], atol=1e-2, rtol=1e-2),
                                         np.isclose(trj_y, reward_pos[1], atol=1e-2, rtol=1e-2)))
reward_loc_ranges = consecutive(reward_idxs.flat)[:-1]
reward_input_rates = np.zeros((trj_t.shape[0], 1))
for reward_loc_idxs in reward_loc_ranges:
    reward_input_rates[reward_loc_idxs,0] = 1.
reward_input_rates = rates_kernel(trj_t, reward_input_rates, tau=0.5)
reward_input_rates = (reward_input_rates / np.max(reward_input_rates)) * reward_peak_rate
reward_input_rates_ip = [make_interp_spline(trj_t, reward_input_rates, k=3)]
plt.plot(trj_t, reward_input_rates_ip[0](trj_t))
plt.show()
        
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
        
#plot_input_rates(exc_input_rates_dict)
#plot_input_rates(inh_input_rates_dict)

n_outputs = 50
n_excitatory = len(exc_trajectory_inputs)
n_inhibitory = len(inh_trajectory_inputs)
n_excitatory_value = 50
n_inhibitory_value = 10

def trajectory_input(trajectory_inputs, t, centered=False):
    n = len(trajectory_inputs)
    result = np.zeros((n,))
    if centered:
        for i in range(n):
            result[i] = 2.*trajectory_inputs[i](t) - 1.
    else:
        for i in range(n):
            result[i] = trajectory_inputs[i](t)
    return result

def trajectory_input_matrix(trajectory_inputs, t, centered=False):
    n = len(trajectory_inputs)
    m = len(t)
    result = np.zeros((n,m))
    if centered:
        for i in range(n):
            result[i,:] = 2.*trajectory_inputs[i](t).flat - 1.
    else:
        for i in range(n):
            result[i,:] = trajectory_inputs[i](t).flat
    return result

def input_matrix_at(mat, t, dt=0.001):
    n = mat.shape[1]
    return mat[:,int(t/dt) % n]

seed = 21

srf_rmrl_network = nengo.Network(label="Reward-modulated replay learning with spatial receptive fields", seed=seed)

with srf_rmrl_network as model:

    rng = np.random.RandomState(seed=seed)

    place_network = PRF(exc_input_func = partial(input_matrix_at,
                                                 trajectory_input_matrix(exc_trajectory_inputs, trj_t)),
                        inh_input_func = partial(input_matrix_at,
                                                 trajectory_input_matrix(inh_trajectory_inputs, trj_t)),
                        n_excitatory = n_excitatory,
                        n_inhibitory = n_inhibitory,
                        n_outputs = n_outputs,
                        label="Spatial receptive field network",
                        seed=seed)

    value_network = PRF(n_excitatory = n_excitatory_value,
                        n_inhibitory = n_inhibitory_value,
                        n_outputs = n_outputs,
                        label="Value network",
                        seed=seed)

    weights_dist_PV_E = rng.normal(size=n_outputs*n_excitatory_value).reshape((n_excitatory_value, n_outputs))
    weights_initial_PV_E = (weights_dist_PV_E - weights_dist_PV_E.min()) / (weights_dist_PV_E.max() - weights_dist_PV_E.min()) * 1e-2
    for i in range(n_excitatory_value):
        sources = np.asarray(rng.choice(n_outputs, round(0.2 * n_outputs), replace=False), dtype=np.int32)
        weights_initial_PV_E[i, np.logical_not(np.in1d(range(n_outputs), sources))] = 0.

    conn_PV_E = nengo.Connection(place_network.output.neurons,
                                 value_network.exc.neurons,
                                 transform=weights_initial_PV_E,
                                 synapse=nengo.Alpha(0.01),
                                 learning_rule_type=RMSP(learning_rate=2e-4))

    weights_dist_PV_I = rng.uniform(size=n_inhibitory_value*n_outputs).reshape((n_inhibitory_value, n_outputs))
    weights_initial_PV_I = (weights_dist_PV_I - weights_dist_PV_I.min()) / (weights_dist_PV_I.max() - weights_dist_PV_I.min()) * 1e-1
    for i in range(n_inhibitory_value):
        sources = np.asarray(rng.choice(n_outputs, round(0.5 * n_outputs), replace=False), dtype=np.int32)
        weights_initial_PV_I[i, np.logical_not(np.in1d(range(n_outputs), sources))] = 0.

    conn_PV_I = nengo.Connection(place_network.output.neurons,
                                 value_network.inh.neurons,
                                 transform=weights_initial_PV_I,
                                 synapse=nengo.Alpha(0.01))

    reward_input = nengo.Node(output=partial(input_matrix_at,
                                             trajectory_input_matrix(reward_input_rates_ip, trj_t)),
                              size_out=1)

    nengo.Connection(reward_input, conn_PV_E.learning_rule,
                     synapse=nengo.Alpha(0.5))
    
    p_reward = nengo.Probe(reward_input, synapse=nengo.Alpha(0.5))
                     
    with place_network:
        p_output_spikes_place = nengo.Probe(place_network.output.neurons, 'spikes')
        p_inh_weights_place = nengo.Probe(place_network.conn_I, 'weights')
        p_exc_weights_place = nengo.Probe(place_network.conn_E, 'weights')
        p_rec_weights_place = nengo.Probe(place_network.conn_EE, 'weights')
        p_exc_rates_place = nengo.Probe(place_network.exc.neurons, 'rates')
        p_inh_rates_place = nengo.Probe(place_network.inh.neurons, 'rates')
                     
    with value_network:
        p_output_spikes_value = nengo.Probe(value_network.output.neurons, 'spikes')
        p_exc_rates_value = nengo.Probe(value_network.exc.neurons, 'rates')
        p_inh_rates_value = nengo.Probe(value_network.inh.neurons, 'rates')
        p_inh_weights_value = nengo.Probe(value_network.conn_I, 'weights')
        p_exc_weights_value = nengo.Probe(value_network.conn_E, 'weights')

    p_PV_E_weights = nengo.Probe(conn_PV_E, 'weights')
    p_PV_I_weights = nengo.Probe(conn_PV_I, 'weights')
    
    
with nengo.Simulator(model, optimize=True) as sim:
    sim.run(np.max(trj_t))

reward = sim.data[p_reward]
output_spikes_place = sim.data[p_output_spikes_place]
output_spikes_value = sim.data[p_output_spikes_value]
np.save("srf_rmrl_place_output_spikes", np.asarray(output_spikes_place, dtype=np.float32))
np.save("srf_rmrl_value_output_spikes", np.asarray(output_spikes_value, dtype=np.float32))
np.save("srf_rmrl_time_range", np.asarray(sim.trange(), dtype=np.float32))
output_rates_place = rates_kernel(sim.trange(), output_spikes_place, tau=0.1)
output_rates_value = rates_kernel(sim.trange(), output_spikes_value, tau=0.1)
sorted_idxs_value = np.argsort(-np.argmax(output_rates_value[37712:].T, axis=1))
sorted_idxs_place = np.argsort(-np.argmax(output_rates_place[37712:].T, axis=1))



