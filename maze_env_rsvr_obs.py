import numpy as np
import scipy.ndimage
from functools import partial
from nengo_maze_env.maze_env import NengoMazeEnvironment
import nengo
from nengo.utils.least_squares_solvers import LSMRScipy
import nengolib
from nengolib import RLS, Network
from reservoir_net import NengoReservoir
from ws import LoadFrom, WeightSaver


# need to install mazelab to use this maze generator
# https://github.com/zuoxingdong/mazelab


seed = 21
dt = 0.005

# parameters related to sensing and angular/linear velocity control
n_sensors = 8
n_t_sensors = 9
tau_sensory = 0.005
ang_exc = 0.1
n_motor = 50

# sensory binding network
n_sensory = 100
ndim_sensory = n_sensors + n_t_sensors + 1
n_sensory_conv = 300
sensory_time_delay = 0.5 # s

# parameters for place learning module
n_place_rsvr = 100
learning_rate_place = 1e-4
tau_place_probe = 0.05
tau_place = 0.05
T_train = 5.0

# We'll make a simple object to implement the delayed connection
class Delay:
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]
    
def sense_to_ang_vel(x, n_sensors, k=1.):
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


model = Network(seed=seed)

with model:

    map_selector = nengo.Node([0])

    environment = nengo.Node(
        NengoMazeEnvironment(
            n_sensors=n_sensors,
            height=15,
            width=15,
            fov=125,
            normalize_sensor_output=True
        ),
        size_in=4,
        size_out=n_sensors + n_t_sensors + 3,
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

    # Circular convolution of current sensory information and sensory
    # information from the recent past

    node_sensory = nengo.Node(size_in=ndim_sensory, output=lambda t,v: v)

    ens_sensory_cur = nengo.Ensemble(n_neurons=n_sensory, dimensions=ndim_sensory, radius=2.0)
    ens_sensory_del = nengo.Ensemble(n_neurons=n_sensory, dimensions=ndim_sensory, radius=2.0)

    nengo.Connection(environment[3:], node_sensory[1:], synapse=None)

    sensory_delay = Delay(ndim_sensory, timesteps=int(sensory_time_delay / dt))
    node_sensory_delay = nengo.Node(sensory_delay.step, size_in=ndim_sensory,
                                    size_out=ndim_sensory)
    nengo.Connection(node_sensory, ens_sensory_cur, synapse=tau_sensory)
    nengo.Connection(node_sensory, node_sensory_delay, synapse=None)
    nengo.Connection(node_sensory_delay, ens_sensory_del, synapse=tau_sensory)

    ens_sensory_bound = nengo.Ensemble(n_neurons=n_sensory_conv, dimensions=ndim_sensory)
    
    bind_sensory = nengo.networks.CircularConvolution(n_neurons=n_sensory_conv,
                                                      dimensions=ndim_sensory)
    nengo.Connection(ens_sensory_del, bind_sensory.A)
    nengo.Connection(ens_sensory_cur, bind_sensory.B)
    nengo.Connection(bind_sensory.output, ens_sensory_bound) 

    # Place learning
    
    place_learning = nengo.Node(size_in=1, output=lambda t,v: True if t < T_train else False)

    rsvr_place= NengoReservoir(n_per_dim = n_place_rsvr, dimensions=ndim_sensory, 
                               learning_rate=learning_rate_place, tau=tau_place,
                               weights_path='maze_env_rsvr_place_rsvr_weights'  )
    
    nengo.Connection(ens_sensory_bound, rsvr_place.input, synapse=None)
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

    
    xy_reader = nengo.Ensemble(n_place_rsvr*ndim_sensory, dimensions=2, radius=10)
    xy_error = nengo.Node(size_in=3, size_out=2,
                          output=lambda t, e: e[1:] if e[0] else 0.)

    xy_reader_conn = nengo.Connection(rsvr_place.ensemble.neurons, xy_reader, synapse=None,
                                      transform=np.zeros((2, n_place_rsvr*ndim_sensory)),
                                      learning_rule_type=RLS(learning_rate=learning_rate_place,
                                                             pre_synapse=tau_place))
    xy_reader_ws = WeightSaver(xy_reader_conn, 'maze_env_rsvr_obs_xy_reader_weights')
    
    nengo.Connection(xy_reader, xy_error[1:])
    nengo.Connection(environment[:2], xy_error[1:], transform=-1)
    nengo.Connection(place_learning, xy_error[0])
    nengo.Connection(xy_error, xy_reader_conn.learning_rule)


    
def on_sim_exit(sim): 
    # this will get triggered when the simulation is over
    rsvr_place.weight_saver.save(sim)
    place_reader_ws.save(sim)    
    xy_reader_ws.save(sim)    


#from nengo_extras.gexf import CollapsingGexfConverter
#CollapsingGexfConverter().convert(model).write('maze_env_rsvr_obs.gexf')

if __name__ == '__main__':
    runtime = 3.0
    with nengo.Simulator(model, progress_bar=True, dt=dt) as sim:
        sim.run(runtime)
    on_sim_exit(sim)




    
