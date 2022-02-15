# -*- coding: utf-8 -*-


import websocket
from io import BytesIO
from PIL import Image, ImageOps
import time
import numpy as np
import nengo
from prf_net import PRF
from rmsp import RMSP
from hsp import HSP
from nengo.utils.progress import TerminalProgressBar


class GodotEnvironment(object):
    
    def __init__(self, address, step_size=5):

        self.ws = websocket.WebSocket()
        self.ws.connect(address)
        self.n_actions = 3
        self.state_dim = (40, 30)
        self.t=0
        self.stepsize = step_size
        self.current_action = 0
        self.state = np.zeros((self.state_dim))
        
    def __del__(self):
        self.ws.close()
        
    def take_action(self, action):
        if action == 0:
            command = "aTurnLeft"
        elif action == 1:
            command = "aTurnRight"
        elif action == 2:
            command = "aMoveForward"
#        elif action == 3:
#            command = "aMoveBackward"
        else:
            raise RuntimeError('Unknown action {action}')
        self.ws.send(command)
        _, im = self.receive_data()
        im1 = ImageOps.grayscale(im)
        self.state = np.array(im1.resize(self.state_dim)) / 256.0
        
    def sensor(self,t):
        return self.state.flatten()
    
    def step(self,t,x):
        if int(t*100)%self.stepsize == 0:
            self.current_action = np.argmax(x) 
            self.take_action(self.current_action)

    def receive_data(self):
        pose_data_received = self.ws.recv()
        image_data_received = self.ws.recv()
    
        pose_data = pose_data_received.decode("utf8").split(", ")
        pose = [float(i) for i in pose_data]
        ## Position data format:
        #[posX, posY, posZ, rotX, rotY, rotZ] = pose_data
    
        # # Load image from BytesIO and display:
        im = Image.open(BytesIO(image_data_received))
        return pose, im


    
env_iface = GodotEnvironment("ws://localhost:9080")
#env.take_action(2)
#env.take_action(1)
#env.take_action(1)
#env.take_action(2)


tau = 0.01

fast_tau = 0
slow_tau = 0.01


n_place = 100
n_input = 1000
n_inhibitory = 200

n_actor = 10
n_critic = 50

sensor_radius = 2
actor_radius = 10

seed = 19

srf_params = {
          'w_actor': 0.1,
          'w_input': 0.1,
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
    
    sensor = nengo.Node(env_iface.sensor)
    sensor_net = nengo.Ensemble(n_neurons=n_input, dimensions=np.prod(env_iface.state_dim), radius=sensor_radius)

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
    
    sensor_conn = nengo.Connection(sensor, sensor_net)
    
    sensor_srf_conn = nengo.Connection(sensor_net.neurons, srf_net.exc.neurons, 
                                       synapse=nengo.Lowpass(tau), 
                                       transform=np.eye(n_input) *srf_params['w_input'])
    
    srf_actor_conn=nengo.Connection(srf_net.output.neurons, actor_net.neurons,
                                    synapse=nengo.Lowpass(tau),
                                    transform=np.eye(n_actor, n_place) * srf_params['w_actor'],
                                    learning_rule_type=HSP(learning_rate=2e-4))
    
    step_node = nengo.Node(env_iface.step, size_in=env_iface.n_actions)
    
    nengo.Connection(actor_net, step_node, synapse=fast_tau)


dt = 0.01    
t_end = 10
with nengo.Simulator(model, optimize=True, dt=dt, progress_bar=TerminalProgressBar()) as sim:
    sim.run(np.max(t_end))
    
