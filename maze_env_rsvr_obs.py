import math
import numpy as np
import scipy.ndimage
from functools import partial
from nengo_maze_env.maze_env import NengoMazeEnvironment, MazeShape
import nengo
from nengo.utils.least_squares_solvers import LSMRScipy
import nengolib
from nengolib import RLS
import nengo_extras
from nengo_extras.plot_spikes import (
    cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance, sample_by_activity)
from reservoir_net import NengoReservoir
from ws import LoadFrom, WeightSaver

# need to install mazelab to use this maze generator
# https://github.com/zuoxingdong/mazelab


seed = 21
dt = 0.001

dim_env = 15

# parameters related to sensing and angular/linear velocity control
n_sensors = 9
tau_sensory = 0.004
n_motor = 50

# sensory binding network
n_sensory = 10
ndim_sensory_distance = n_sensors
ndim_sensory_texture = n_sensors
ndim_sensory = ndim_sensory_texture  + ndim_sensory_distance
sensory_time_delay = 0.25 # s

# parameters for place learning module
n_place = 100
learning_rate_place = 1e-5
tau_place_probe = 0.05
tau_place = 0.05
tau_reward = 0.1
T_train = 5.0
T_test = 5.0



# We'll make a simple object to implement the delayed connection
class Delay:
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((dimensions, timesteps))

    def step(self, t, x):
        result = np.copy(self.history[:,0])
        self.history = np.roll(self.history, (0, -1))
        self.history[:,-1] = x
        return result

def inst_vel(dx):
    du = np.sqrt(dx[0]**2. + dx[1]**2)
    return du
    


def sense_to_ang_vel(x, n_sensors):
    rotation_weights = np.linspace(-1, 1, n_sensors)
    res = np.dot(rotation_weights, np.array(x))
    return res

def sense_to_lin_vel(x, n_sensors, v=0.5):
    min_dist = np.min(x)
    max_dist = np.max(x)
    res = 0.
    if max_dist > 0.:
        res = v * max_dist
    else:
        if min_dist > 0.:
            res = -v * min_dist
        else:
            res = -v 
    return res

rng = np.random.RandomState(seed=seed)

model = nengo.Network(seed=seed)

with model:

    map_selector = nengo.Node([0])

    environment = nengo.Node(
        NengoMazeEnvironment(
            n_sensors=n_sensors,
            height=dim_env,
            width=dim_env,
            fov=125,
            normalize_sensor_output=True,
            reward_location=(10,10),
            maze_shape=MazeShape.MAZE_HANLON,
            maze_kwargs={'n_objects': 15}
        ),
        size_in=4,
        size_out=2*n_sensors + 6,
    )

    linear_velocity = nengo.Ensemble(n_neurons=n_motor, dimensions=1)
    angular_velocity = nengo.Ensemble(n_neurons=n_motor, dimensions=1)

    nengo.Connection(map_selector, environment[3]) # dimension 4
    
    ang_sensors = nengo.Node(output=lambda t, x: x, size_in=n_sensors)
    nengo.Connection(environment[5:n_sensors+5], ang_sensors)

    nengo.Connection(linear_velocity, environment[0], synapse=tau_sensory) # dimension 1
    nengo.Connection(angular_velocity, environment[2], synapse=tau_sensory) 

    ang_control_func = partial(sense_to_ang_vel, n_sensors = n_sensors)
    nengo.Connection(ang_sensors, angular_velocity,
                     function=ang_control_func,
                     synapse=tau_sensory)

    lin_control_func = partial(sense_to_lin_vel, n_sensors = n_sensors)
    nengo.Connection(ang_sensors, linear_velocity,
                     function=lin_control_func,
                     synapse=tau_sensory)
    
    ang_exc_const = nengo.Node([0.25])
    ang_exc = nengo.Ensemble(n_neurons=n_motor, dimensions=1)
    nengo.Connection(ang_exc_const, ang_exc, synapse=tau_sensory)
    nengo.Connection(ang_exc, angular_velocity)

    node_xy = nengo.Node(size_in=2, size_out=2,
                         output=lambda t,v: v)

    node_reward = nengo.Node(size_in=1, size_out=1,
                             output=lambda t,v: v)
    
    
    node_sensory = nengo.Node(output=lambda t,v: v,
                              size_in=ndim_sensory,
                              size_out=ndim_sensory)
    ## Derivative of x and y position
    #nengo.Connection(environment[2],
    #                 node_sensory[0], synapse=None)
    #nengo.Connection(environment[3],
    #                 node_sensory[1], synapse=None)
    nengo.Connection(environment[5+ndim_sensory_distance:ndim_sensory_texture+ndim_sensory_distance+5],
                     node_sensory[:ndim_sensory_texture],
                     synapse=None)
    nengo.Connection(environment[5:ndim_sensory_distance+5],
                     node_sensory[ndim_sensory_texture:ndim_sensory_distance+ndim_sensory_texture],
                     synapse=None)

    
    node_sensory_cur = nengo.Node(output=lambda t,v: v,
                                  size_in=ndim_sensory,
                                  size_out=ndim_sensory)
    sensory_delay = Delay(ndim_sensory, timesteps=int(sensory_time_delay / dt))
    node_sensory_del = nengo.Node(sensory_delay.step,
                                  size_in=ndim_sensory,
                                  size_out=ndim_sensory)
    
    nengo.Connection(node_sensory, node_sensory_cur, synapse=None)
    nengo.Connection(node_sensory, node_sensory_del, synapse=None)

    nengo.Connection(environment[:2], node_xy, synapse=None)
    nengo.Connection(environment[ndim_sensory+5], node_reward, synapse=None)


    # Place learning
    place_learning = nengo.Node(size_in=1, output=lambda t,v: True if t < T_train else False)

    rsvr_place = NengoReservoir(n_per_dim = n_place, dimensions=ndim_sensory, 
                                learning_rate=learning_rate_place, synapse=tau_place,
                                ie = 0.2, skewnorm_a_inh=9.0, g_inh = 9 * 1e-5,
                                weights_path='maze_env_rsvr_place_rsvr_weights'  )
    
    nengo.Connection(place_learning, rsvr_place.enable_learning, synapse=None)

    nengo.Connection(node_sensory_del, rsvr_place.input, synapse=tau_sensory)
    nengo.Connection(node_sensory_cur, rsvr_place.train, synapse=tau_sensory)

#    place_reader = nengo.Ensemble(n_place*ndim_sensory, dimensions=ndim_sensory)
#    place_error = nengo.Node(size_in=ndim_sensory+1, size_out=ndim_sensory,
#                             output=lambda t, e: e[1:] if e[0] else 0.)

#    place_reader_conn = nengo.Connection(rsvr_place.ensemble.neurons, place_reader, synapse=None,
#                                        transform=np.zeros((ndim_sensory, n_place*ndim_sensory)),
#                                        learning_rule_type=RLS(learning_rate=learning_rate_place,
#                                                               pre_synapse=tau_place))
                                                               
#    place_reader_ws = WeightSaver(place_reader_conn, 'maze_env_rsvr_place_reader_weights')
    
    # Error = actual - target = post - pre
#    nengo.Connection(place_reader, place_error[1:])
#    nengo.Connection(node_sensory_cur, place_error[1:ndim_sensory+1], transform=-1, synapse=tau_sensory)

#    nengo.Connection(place_learning, place_error[0])
    # Connect the error into the learning rule
#    nengo.Connection(place_error, place_reader_conn.learning_rule)
    
    ## Dimensionality reduction readout
#    xy_transform = np.random.choice(np.asarray([-1,0,1]),
#                                    size=(2, n_place*ndim_sensory),
#                                    p=[1/6,2/3,1/6])
#    nengo.Connection(rsvr_place, xy_reader, transform=xy_transform)

    value = nengo.Ensemble(n_place, dimensions=1)
    value_conn = nengo.Connection(rsvr_place.output, value, function=lambda x: 0,
                                  learning_rule_type=RLS(learning_rate=learning_rate_place,
                                                         pre_synapse=tau_place))
    
    # this connection adds the reward to the learning signal
    nengo.Connection(node_reward, value_conn.learning_rule, transform=-1, synapse=tau_value)

    # this connection adds the observed value
    nengo.Connection(value, value_conn.learning_rule, transform=-0.9, synapse=0.01)

    # this connection subtracts the predicted value
    nengo.Connection(value, learn_conn.learning_rule, transform=1, synapse=tau_value)
                   

    
def on_sim_exit(sim): 
    # this will get triggered when the simulation is over
    rsvr_place.weight_saver.save(sim)
    place_reader_ws.save(sim)    


#from nengo_extras.gexf import CollapsingGexfConverter
#CollapsingGexfConverter().convert(model).write('maze_env_rsvr_obs.gexf')

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    runtime = T_train + T_test
    
    with model:
        tau_probe = 0.05
        p_place_error = nengo.Probe(place_error, synapse=tau_probe)
        p_xy = nengo.Probe(node_xy, synapse=tau_probe)
        p_place_reader = nengo.Probe(place_reader, synapse=tau_probe)
        p_place_reader_spikes = nengo.Probe(place_reader.neurons,
                                            attr='spikes',
                                            synapse=tau_probe)
        p_xy_reader = nengo.Probe(xy_reader, synapse=tau_probe)
        p_xy_reader_spikes = nengo.Probe(xy_reader.neurons,
                                         attr='spikes',
                                         synapse=tau_probe)

    with nengo.Simulator(model, progress_bar=True, dt=dt) as sim:
        sim.run(runtime)

        on_sim_exit(sim)

        t_train_idxs = np.where(sim.trange() <= T_train)[0]
        
        np.save("place_reader_spikes",
                sim.data[p_place_reader_spikes])
        np.save("place_reader",
                sim.data[p_place_reader])
        np.save("xy", sim.data[p_xy])
        np.save("xy_reader",
                sim.data[p_xy_reader])
        np.save("xy_reader_spikes",
                sim.data[p_xy_reader_spikes])

        
        plt.figure(figsize=(16, 6))
        plt.title("XY Position")
        plt.plot(sim.trange(), sim.data[p_xy])
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.show()
        
        plt.figure(figsize=(16, 6))
        plt.title("XY Reader")
        plt.plot(sim.trange(), sim.data[p_xy_reader])
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.show()

        plt.figure(figsize=(16, 6))
        plt.title("Training Error")
        plt.plot(sim.trange()[t_train_idxs], np.mean(sim.data[p_place_error][t_train_idxs], axis=1),
                 alpha=0.8, label="Mean place error")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.show()

        plt.figure(figsize=(16, 6))
        plt.title("Place Reader Spikes")
        plot_spikes(
            *merge(
                *cluster(
                    *sample_by_variance(
                        sim.trange(), sim.data[p_place_reader_spikes],
                        num=500, filter_width=.02),
                    filter_width=.002),
                num=50))
        plt.show()

        




    
