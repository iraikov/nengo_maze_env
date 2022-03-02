# -*- coding: utf-8 -*-


import websocket
from io import BytesIO
from PIL import Image, ImageOps
import time, math
import numpy as np
import nengo
from prf_net import PRF
from rmsp import RMSP
from hsp import HSP
from nengo.utils.progress import TerminalProgressBar
from nengo_extras.vision import Gabor, Mask
from nengo_extras.gui import image_display_function


image_shape = (1, 40, 30)

class GodotEnvironment(object):
    
    def __init__(self, address, step_size=5, collision_distance=1.2):

        self.ws = websocket.WebSocket()
        self.ws.connect(address)
        self.n_actions = 4
        self.image_state_dim = (40, 30)
        self.im_dim = (30, 40)
        self.pose_dim = (6, )
        self.terrain_dim = (5, 53)
        self.t=0
        self.stepsize = step_size
        self.current_action = 0
        self.image_state = np.zeros((self.image_state_dim))
        self.pose_state = np.zeros(self.pose_dim)
        self.collision_state = None
        self.midline_level = -1
        self.collision_distance = collision_distance
        
    def __del__(self):
        self.ws.close()
        
    def take_action(self, action):
        if action == 0:
            command = "aTurnLeft"
        elif action == 1:
            command = "aTurnRight"
        elif action == 2:
            command = "aMoveForward"
        elif action == 3:
            command = "aMoveBackward"
        else:
            raise RuntimeError('Unknown action {action}')
        self.ws.send(command)
        pose_array, terrain_array, im = self.receive_data()
        im1 = ImageOps.grayscale(im)

        # *****Distance/collision arrays, the FIRST TWO ROWS are sensor ids (height, position/angle),
        # SEE SENSOR DIAGRAMS
        distance_array = np.vstack([terrain_array[0:2,:], np.linalg.norm(terrain_array[2:5,:], axis=0)])
        collision_array = distance_array[:,np.where(distance_array[2,:] < self.collision_distance)[0]]
        # data from sensors at mid-height
        midline_data = distance_array[:,np.where(distance_array[0,:] == self.midline_level)[0]]

        print(f'collision_array.shape = {collision_array.shape}')
        self.pose_state = pose_array
        self.image_state = np.array(im1.resize(self.im_dim, Image.ANTIALIAS)) / 256.0
        self.collision_state = collision_array
        
    def image_sensor(self,t):
        return self.image_state.flatten()

    def pose_sensor(self,t):
        return self.pose_state.flatten()

    def collision_sensor(self,t):
        return self.collision_state.flatten()
    
    def step(self,t,x):
        if int(t*100)%self.stepsize == 0:
            self.current_action = np.argmax(x) 
            self.take_action(self.current_action)

    def receive_data(self):
        
        pose_data_received = self.ws.recv()
        terrain_data_received = self.ws.recv()
        image_data_received = self.ws.recv()
    
        pose_data = pose_data_received.decode("utf8").split(", ")
        pose_array = np.asarray([float(i) for i in pose_data])
        ## Position data format:
        #[posX, posY, posZ, rotX, rotY, rotZ] = pose_data
        im = Image.open(BytesIO(image_data_received))

        terrain_data = terrain_data_received.decode("utf8")
        terrain_data = terrain_data.replace("(", "").replace(" ", "").replace(")]","").replace("[","").split("),")
        terrain_array = np.zeros((5,int(len(terrain_data)/2)))
        print(f'terrain_array.shape = {terrain_array.shape}')
        count = 0
        for i in range(len(terrain_data)):
            if i % 2 == 0:  # if index is even, write point ids into first two columns
                count = 0
                point_id = terrain_data[i].replace(")", "").split(",")
                for j in point_id:
                    terrain_array[count, math.floor(i/2)] = int(j)
                    count = count + 1
            else:           # if index is odd, write point coords into last three columns
                point_coords = terrain_data[i].replace(")", "").split(",")
                for j in point_coords:
                    terrain_array[count, math.floor(i/2)] = float(j)
                    count = count + 1
        # Load image from BytesIO and display:
        return pose_array, terrain_array, im


    
env_iface = GodotEnvironment("ws://localhost:9080")


tau = 0.01

fast_tau = 0
slow_tau = 0.01


n_place = 100
n_input = 200
n_inhibitory = 100

n_actor = 10
n_critic = 50

sensor_radius = 2
actor_radius = 10

seed = 19

rng = np.random.RandomState(seed)

srf_params = {
          'w_actor': 0.1,
          'w_input': 0.05,
          'w_initial_E': 0.01, 
          'w_initial_EI': 1e-3,
          'w_initial_EE': 0.001, 
          'w_initial_I': -0.05, 
          'w_EI_Ext': 1e-3,
          'p_E': 0.05, 
          'p_EE': 0.05, 
          'p_EI': 0.1,
          'p_EI_Ext': 0.007,
          'sigma_scale_E': 0.002,
          'tau_E': 0.005, 
          'tau_I': 0.020, 
          'tau_input': 0.1,
          'learning_rate_I': 0.01, 
          'learning_rate_E': 0.005,
          'learning_rate_EE': 1e-4
}

    
model=nengo.Network()

with model:
    
    image_sensor = nengo.Node(env_iface.image_sensor)
    image_sensor_net = nengo.Ensemble(n_neurons=n_input,
                                      dimensions=np.prod(env_iface.image_state_dim),
                                      radius=sensor_radius)

    gabor_size = (5, 5)  # Size of the gabor filter

    # Generate the encoders for the sensory ensemble
    image_sensor_encoders = Gabor().generate(n_input, gabor_size, rng=rng)
    image_sensor_encoders = Mask(image_shape).populate(image_sensor_encoders, rng=rng, flatten=True)
    image_sensor_net.encoders = image_sensor_encoders
    
    srf_net = PRF(n_excitatory = n_input,
                  n_inhibitory = n_inhibitory,
                  connect_exc_inh_input=True,
                  n_outputs = n_place,
                  dimensions=env_iface.n_actions,
                  label="Spatial receptive field network",
                  seed=seed, **srf_params)

    actor_net = nengo.Ensemble(n_neurons=n_actor,
                               dimensions=env_iface.n_actions,
                               radius=actor_radius)
    
    image_sensor_conn = nengo.Connection(image_sensor, image_sensor_net)
    
    image_sensor_srf_conn = nengo.Connection(image_sensor_net.neurons, srf_net.exc.neurons, 
                                             synapse=nengo.Lowpass(tau), 
                                             transform=np.eye(n_input) *srf_params['w_input'])
    
    srf_actor_conn=nengo.Connection(srf_net.output.neurons, actor_net.neurons,
                                    synapse=nengo.Lowpass(tau),
                                    transform=np.eye(n_actor, n_place) * srf_params['w_actor'],
                                    learning_rule_type=HSP(learning_rate=2e-4))
    
    step_node = nengo.Node(env_iface.step, size_in=env_iface.n_actions)
    
    nengo.Connection(actor_net, step_node, synapse=fast_tau)


    display_func = image_display_function(image_shape)
    display_node = nengo.Node(display_func, size_in=image_sensor.size_out)
    nengo.Connection(image_sensor, display_node, synapse=None)

dt = 0.01    
t_end = 10
with nengo.Simulator(model, optimize=True, dt=dt, progress_bar=TerminalProgressBar()) as sim:
    sim.run(np.max(t_end))
    
