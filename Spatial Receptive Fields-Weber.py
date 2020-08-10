

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import nengo

import numpy as np
import scipy.interpolate
from scipy.interpolate import Rbf, CubicSpline
from rbf.pde.nodes import disperse, poisson_disc_nodes
from hsp import HSP
from isp import ISP

from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from nengo_extras.neurons import (
    rates_kernel, rates_isi )

# In[2]:


def generate_linear_trajectory(input_trajectory, temporal_resolution=1., velocity=30., equilibration_duration=None, n_trials=1):
    """
    Construct coordinate arrays for a spatial trajectory, considering run velocity to interpolate at the specified
    temporal resolution. Optionally, the trajectory can be prepended with extra distance traveled for a specified
    network equilibration time, with the intention that the user discards spikes generated during this period before
    analysis.

    :param trajectory: namedtuple
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
nmodules = 4

field_width_params = [35.0, 0.32]
exc_field_width  = lambda x: 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
inh_field_width  = lambda x: 100. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)

exc_module_field_width_dict = {i : exc_field_width( float(i) / float(nmodules) ) for i in range(nmodules)}
inh_module_field_width_dict = {i : inh_field_width( float(i) / float(nmodules) ) for i in range(2)}
    

def generate_input_rates(module_field_width_dict, basis_function='gaussian', spacing_factor=1.0, peak_rate=1.):
    input_nodes_dict = {}
    input_groups_dict = {}
    input_rates_dict = {}
    
    for m in module_field_width_dict:
        nodes, groups, _ = poisson_disc_nodes(module_field_width_dict[m], (vert, smp))
        input_groups_dict[m] = groups
        input_nodes_dict[m] = nodes
        input_rates_dict[m] = {}
        vertlst = [[v[0], v[1]] for v in vert]
        
        for i in range(nodes.shape[0]):
            xs = [[nodes[i,0], nodes[i,1]]] + vertlst
            x_obs = np.asarray(xs).reshape((1,-1))
            u_obs = np.asarray([[peak_rate]]).reshape((1,-1))
            input_rate_ip  = Rbf(x_obs[:,0], x_obs[:,1], u_obs,
                                 function=basis_function, 
                                 epsilon=module_field_width_dict[m] * spacing_factor)
            input_rates_dict[m][i] = input_rate_ip
            
    return input_nodes_dict, input_groups_dict, input_rates_dict

exc_input_nodes_dict, exc_input_groups_dict, exc_input_rates_dict = \
    generate_input_rates(exc_module_field_width_dict, spacing_factor=0.8, peak_rate=peak_rate)
inh_input_nodes_dict, inh_input_groups_dict, inh_input_rates_dict = \
    generate_input_rates(inh_module_field_width_dict, spacing_factor=1.0, peak_rate=peak_rate)

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

def plot_input_rates(input_rates_dict):
    for m in input_rates_dict:
        plt.figure()
        arena_map = np.zeros(arena_xx.shape)
        for i in range(len(input_rates_dict[m])):
            input_rates = input_rates_dict[m][i](arena_xx, arena_yy)
            arena_map += input_rates
        plt.pcolor(arena_xx, arena_yy, arena_map, cmap=cm.jet)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()
        
plot_input_rates(exc_input_rates_dict)
plot_input_rates(inh_input_rates_dict)


# In[4]:


diag_trajectory = np.asarray([[-100, -100], [100, 100]])
trj_t, trj_x, trj_y, trj_d = generate_linear_trajectory(diag_trajectory, temporal_resolution=0.001, n_trials=4)
    
exc_trajectory_input_rates = { m: {} for m in exc_input_rates_dict }
inh_trajectory_input_rates = { m: {} for m in inh_input_rates_dict }
exc_trajectory_inputs = []
inh_trajectory_inputs = []
    
for m in exc_input_rates_dict:
    for i in exc_input_rates_dict[m]:
        input_rates = exc_input_rates_dict[m][i](trj_x, trj_y)
        input_rates[np.isclose(input_rates, 0., atol=1e-4, rtol=1e-4)] = 0.
        exc_trajectory_input_rates[m][i] = input_rates
        input_rates_ip = CubicSpline(trj_t, input_rates)
        exc_trajectory_inputs.append(input_rates_ip)
        
for m in inh_input_rates_dict:
    for i in inh_input_rates_dict[m]:
        input_rates = inh_input_rates_dict[m][i](trj_x, trj_y)
        input_rates[np.isclose(input_rates, 0., atol=1e-4, rtol=1e-4)] = 0.
        inh_trajectory_input_rates[m][i] = input_rates
        input_rates_ip = CubicSpline(trj_t, input_rates)
        inh_trajectory_inputs.append(input_rates_ip)
            

# In[5]:


def trajectory_input(trajectory_inputs, t, centered=False):
    if centered:
        result = np.asarray([ 2.*y(t) - 1. for y in trajectory_inputs ])
    else:
        result = np.asarray([ y(t) for y in trajectory_inputs ])
    return result
   
#E_train = 2.*exc_input_rate_matrix - 1.
#I_train = 2.*inh_input_rate_matrix - 1.


# In[ ]:


def make_weber_srf_model():
        with nengo.Network(label="Weber spatial receptive field model", seed=19) as model:

            rng = np.random.RandomState(seed=19)
            
            N_Exc = len(exc_trajectory_inputs)
            N_Inh = len(inh_trajectory_inputs)
            model.ExcInput = nengo.Node(partial(trajectory_input, exc_trajectory_inputs),
                                        size_out=N_Exc)
            model.InhInput = nengo.Node(partial(trajectory_input, inh_trajectory_inputs),
                                        size_out=N_Inh)
                                         
            model.Exc = nengo.Ensemble(N_Exc, dimensions=1,
                                       neuron_type=nengo.RectifiedLinear(),
                                       intercepts=nengo.dists.Choice([0.1]),
                                       max_rates=nengo.dists.Choice([20]))
            
            model.Inh = nengo.Ensemble(N_Inh, dimensions=1,
                                       neuron_type=nengo.RectifiedLinear(),
                                       intercepts=nengo.dists.Choice([0.1]),
                                       max_rates=nengo.dists.Choice([40]))
            
            model.Output = nengo.Ensemble(1, dimensions=1,
                                          neuron_type=nengo.LIFRate(),
                                          intercepts=nengo.dists.Choice([0.1]),
                                          max_rates=nengo.dists.Choice([20]))

            nengo.Connection(model.ExcInput, model.Exc.neurons, synapse=nengo.Lowpass(0.01),
                             transform=np.eye(N_Exc))
            nengo.Connection(model.InhInput, model.Inh.neurons, synapse=nengo.Lowpass(0.01),
                             transform=np.eye(N_Inh))

            
            #model.conn_EI = nengo.Connection(model.Exc.neurons, model.Inh.neurons,
            #                                 transform=np.ones((N_Inh, N_Exc)) * 1e-6,
            #                                 synapse=nengo.Alpha(0.01))
            weights_dist_I =rng.normal(size=N_Inh).reshape((1, N_Inh))
            weights_initial_I = (weights_dist_I - weights_dist_I.min()) / (weights_dist_I.max() - weights_dist_I.min()) * -1e-2
            plt.hist(weights_initial_I.flat)
            plt.show()
            model.conn_I = nengo.Connection(model.Inh.neurons, model.Output.neurons,
                                            transform=weights_initial_I,
                                            synapse=nengo.Alpha(0.01), 
                                            learning_rule_type=ISP(learning_rate=4e-6, rho0=1.))
            weights_dist_E = rng.normal(size=N_Exc).reshape((1, N_Exc))
            weights_initial_E = (weights_dist_E - weights_dist_E.min()) / (weights_dist_E.max() - weights_dist_E.min()) * 1e-1
            plt.hist(weights_initial_E.flat)
            plt.show()
            model.conn_E = nengo.Connection(model.Exc.neurons, model.Output.neurons, 
                                            transform=weights_initial_E,
                                            synapse=nengo.Alpha(0.01),
                                            learning_rule_type=HSP(learning_rate=2e-6))
                             
            return model

      
srf_model = make_weber_srf_model()

with srf_model:
    p_output_rates = nengo.Probe(srf_model.Output.neurons, 'rates')
    p_inh_rates = nengo.Probe(srf_model.Inh.neurons, 'rates')
    p_inh_weights = nengo.Probe(srf_model.conn_I, 'weights')
    p_exc_rates = nengo.Probe(srf_model.Exc.neurons, 'rates')
    p_exc_weights = nengo.Probe(srf_model.conn_E, 'weights')
        
with nengo.Simulator(srf_model) as sim:
    sim.run(np.max(trj_t))
    
#plot_spikes(sim.trange(), sim.data[p_inh_rates][0,:])




