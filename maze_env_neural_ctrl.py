import nengo
import numpy as np
import scipy.ndimage
from functools import partial
from nengo_maze_env.maze_env import NengoMazeEnvironment

# need to install mazelab to use this maze generator
# https://github.com/zuoxingdong/mazelab

dt = 0.001
tau_sensory = 0.004
n_sensors = 9
n_t_sensors = 9
n_neurons = 50

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


model = nengo.Network(seed=21)

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
        size_out=n_sensors + n_t_sensors + 4,
    )

    linear_velocity = nengo.Ensemble(n_neurons=n_neurons, dimensions=1)
    angular_velocity = nengo.Ensemble(n_neurons=n_neurons, dimensions=2)

    nengo.Connection(map_selector, environment[3]) # dimension 4
    nengo.Connection(linear_velocity, environment[0], synapse=tau_sensory) # dimension 1
    nengo.Connection(angular_velocity[0], environment[1], synapse=tau_sensory) 
    nengo.Connection(angular_velocity[1], environment[2], synapse=tau_sensory) 

    ang_control_func = partial(sense_to_ang_vel, n_sensors = n_sensors)
    nengo.Connection(environment[3:n_sensors+3], angular_velocity, function=ang_control_func,
                     transform=[[1.], [1.]], synapse=tau_sensory)

    lin_control_func = partial(sense_to_lin_vel, n_sensors = n_sensors)
    nengo.Connection(environment[3:n_sensors+3], linear_velocity, function=lin_control_func,
                     synapse=tau_sensory)
    
    ang_exc_const = nengo.Node([0.1])
    ang_exc = nengo.Ensemble(n_neurons=n_neurons, dimensions=1)
    nengo.Connection(ang_exc_const, ang_exc, synapse=tau_sensory)
    nengo.Connection(ang_exc, angular_velocity, transform=[[1.], [1.]])

    
    

if __name__ == '__main__':
    runtime = 5.0
    
    with model:
        tau_probe = 0.05
        p_xy = nengo.Probe(environment, synapse=tau_probe)


    with nengo.Simulator(model, progress_bar=True, dt=dt) as sim:
        sim.run(runtime)
        
        plt.figure(figsize=(16, 6))
        plt.title("X-Y Position")
        plt.plot(sim.trange(), sim.data[p_xy],
                 alpha=0.8, label="Vel error")
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.legend()
        plt.show()

    




    
