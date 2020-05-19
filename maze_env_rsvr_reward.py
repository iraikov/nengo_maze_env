import numpy as np
import scipy.ndimage
from functools import partial
from nengo_maze_env.maze_env import NengoMazeEnvironment
import nengo
from nengo.utils.least_squares_solvers import LSMRScipy
import nengolib
from nengolib import RLS
from reservoir_net import NengoReservoir
from ws import LoadFrom, WeightSaver
import matplotlib.pyplot as plt

# need to install mazelab to use this maze generator
# https://github.com/zuoxingdong/mazelab


seed = 21
dt = 0.005

# parameters related to sensing and angular/linear velocity control
n_sensors = 9
n_t_sensors = 9
tau_sensory = 0.004
ang_exc = 0.2
n_motor = 50

# sensory binding network
n_sensory = 50
ndim_sensory = n_sensors + n_t_sensors + 1
sensory_time_delay = 0.5 # s

# parameters for place learning module
n_place_rsvr = 30
learning_rate_place = 1e-4
tau_place_probe = 0.05
tau_place = 0.05

# parameters for reward module
n_value = 40
tau_value_slow = 0.1
tau_value_fast = 0.01
learning_rate_value = 1e-4


T_train = 20.0

# We'll make a simple object to implement the delayed connection
class Delay:
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]
    
def sense_to_ang_vel(x, n_sensors, k=2.0):
    rotation_weights = np.linspace(-1, 1, n_sensors)
    res = k * np.dot(rotation_weights, np.array(x))
    return res

def sense_to_lin_vel(x, n_sensors, v=0.5):
    max_dist = np.max(x)
    res = 0.
    if max_dist > 0.:
        res = v * max_dist
    else:
        res = -v
    return res

rng = np.random.RandomState(seed=seed)

model = nengo.Network(seed=seed)

with model:

    map_selector = nengo.Node([1])

    environment = nengo.Node(
        NengoMazeEnvironment(
            n_sensors=n_sensors,
            height=15,
            width=15,
            fov=125,
            normalize_sensor_output=True
        ),
        size_in=4,
        size_out=n_sensors + n_t_sensors + 4,
    )

    linear_velocity = nengo.Ensemble(n_neurons=n_motor, dimensions=1)

    angular_velocity = nengo.Ensemble(n_neurons=n_motor, dimensions=2)

    nengo.Connection(map_selector, environment[3]) # dimension 4
    nengo.Connection(linear_velocity, environment[0], synapse=tau_sensory) # dimension 1
    nengo.Connection(angular_velocity[0], environment[1], synapse=tau_sensory) 
    nengo.Connection(angular_velocity[1], environment[2], synapse=tau_sensory) 

    ang_control_func = partial(sense_to_ang_vel, n_sensors = n_sensors)
    nengo.Connection(environment[3:n_sensors+3], angular_velocity,
                     function=ang_control_func,
                     transform=[[1.], [1.]], synapse=tau_sensory)

    ang_exc_const = nengo.Node([ang_exc])
    ang_exc = nengo.Ensemble(n_neurons=n_motor, dimensions=1)
    nengo.Connection(ang_exc_const, ang_exc, synapse=tau_sensory)
    nengo.Connection(ang_exc, angular_velocity, transform=[[1.], [1.]])

    lin_control_func = partial(sense_to_lin_vel, n_sensors = n_sensors)
    nengo.Connection(environment[3:n_sensors+3], linear_velocity,
                     function=lin_control_func, synapse=tau_sensory)


    node_sensory = nengo.Node(size_in=ndim_sensory, output=lambda t,v: v)

    ens_sensory_cur = nengo.Ensemble(n_neurons=n_sensory, dimensions=ndim_sensory, radius=2.0)
    ens_sensory_del = nengo.Ensemble(n_neurons=n_sensory, dimensions=ndim_sensory, radius=2.0)

    nengo.Connection(environment[2:2+ndim_sensory], node_sensory, synapse=None)

    sensory_delay = Delay(ndim_sensory, timesteps=int(sensory_time_delay / dt))
    node_sensory_delay = nengo.Node(sensory_delay.step, size_in=ndim_sensory,
                                    size_out=ndim_sensory)
    nengo.Connection(node_sensory, ens_sensory_cur, synapse=tau_sensory)
    nengo.Connection(node_sensory, node_sensory_delay, synapse=None)
    nengo.Connection(node_sensory_delay, ens_sensory_del, synapse=tau_sensory)


    # Place learning
    
    place_learning = nengo.Node(size_in=1, output=lambda t,v: True if t < T_train else False)

    rsvr_place= NengoReservoir(n_per_dim = n_place_rsvr, dimensions=ndim_sensory, 
                               learning_rate=learning_rate_place, tau=tau_place,
                               weights_path='maze_env_rsvr_place_rsvr_weights'  )
    
    nengo.Connection(ens_sensory_del, rsvr_place.input, synapse=None)
    nengo.Connection(place_learning, rsvr_place.enable_learning, synapse=None)

    place_reader = nengo.Ensemble(n_place_rsvr*ndim_sensory, dimensions=ndim_sensory)
    place_error = nengo.Node(size_in=ndim_sensory+1, size_out=ndim_sensory,
                             output=lambda t, e: e[1:] if e[0] else 0.)

    place_reader_conn = nengo.Connection(rsvr_place.ensemble.neurons, place_reader, synapse=None,
                                        transform=np.zeros((ndim_sensory, n_place_rsvr*ndim_sensory)),
                                        learning_rule_type=RLS(learning_rate=learning_rate_place,
                                                               pre_synapse=tau_place))
                                                               
    place_reader_ws = WeightSaver(place_reader_conn, 'maze_env_rsvr_place_reader_weights')
    
    # Error = actual - target = post - pre
    nengo.Connection(place_reader, place_error[1:])
    nengo.Connection(ens_sensory_cur, place_error[1:], transform=-1)

    nengo.Connection(place_learning, place_error[0])
    # Connect the error into the learning rule
    nengo.Connection(place_error, place_reader_conn.learning_rule)

    ## Reward reader
    #xy_transform = rng.randn(n_place_rsvr*ndim_sensory, n_place_rsvr*ndim_sensory, )
    #xy_in = nengo.Ensemble(n_place_rsvr*ndim_sensory, dimensions=2, radius=2.)
    #nengo.Connection(place_reader.neurons, xy_in.neurons, transform=xy_transform)

    node_reward = nengo.Node(size_in=1)
    nengo.Connection(environment[2+ndim_sensory], node_reward, synapse=None)

    ens_value = nengo.Ensemble(n_value, dimensions=1, radius=2.)

    ens_value_learn = nengo.Ensemble(n_neurons=1000, dimensions=1)

    value_conn = nengo.Connection(place_reader.neurons, ens_value,
                                  transform=np.zeros((1, n_place_rsvr*ndim_sensory)),
                                  learning_rule_type=RLS(learning_rate=learning_rate_value,
                                                         pre_synapse=tau_value_slow),
                                  synapse=tau_value_fast)
    value_ws = WeightSaver(value_conn, 'maze_env_rsvr_value_weights')

    nengo.Connection(ens_value_learn, value_conn.learning_rule, transform=-1, synapse=tau_value_fast)
    
    # this connection subtracts the reward to the value error signal
    nengo.Connection(node_reward, ens_value_learn,
                     transform=-1, synapse=tau_value_slow)
    # this connection adds the predicted value
    nengo.Connection(ens_value, ens_value_learn,
                     transform=-1, synapse=tau_value_slow)
    # this connection adds the observed value
    nengo.Connection(ens_value, ens_value_learn,
                     transform=0.9, synapse=tau_value_fast)

    
    
def on_sim_exit(sim): 
    # this will get triggered when the simulation is over
    rsvr_place.weight_saver.save(sim)
    place_reader_ws.save(sim)    
    value_ws.save(sim)    


#from nengo_extras.gexf import CollapsingGexfConverter
#CollapsingGexfConverter().convert(model).write('maze_env_rsvr_obs.gexf')

if __name__ == '__main__':
    runtime = T_train
    
    with model:
        tau_probe = 0.05
        p_place_error = nengo.Probe(place_error, synapse=tau_probe)
        p_value = nengo.Probe(ens_value, synapse=tau_probe)
        p_reward = nengo.Probe(node_reward, synapse=tau_probe)
        p_place_error = nengo.Probe(place_reader, synapse=tau_probe)


    with nengo.Simulator(model, progress_bar=True, dt=dt) as sim:
        sim.run(runtime)

    
        plt.figure(figsize=(16, 6))
        plt.title("Training Output")
        plt.plot(sim.trange(), sim.data[p_value],
                 alpha=0.8, label="Value")
        plt.plot(sim.trange(), sim.data[p_reward],
                 alpha=0.8, label="Reward")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(16, 6))
        plt.title("Training Error")
        plt.plot(sim.trange(), np.mean(sim.data[p_place_error], axis=1),
                 alpha=0.8, label="Mean place error")
        plt.xlabel("Time (s)")
        plt.ylabel("Output")
        plt.legend()
        plt.show()

    
    on_sim_exit(sim)




    
