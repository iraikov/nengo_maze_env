import sys
from collections import deque
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from scipy.interpolate import Rbf, Akima1DInterpolator
from scipy.spatial import Delaunay
from rbf.pde.nodes import disperse, poisson_disc_nodes
import networkx as nx

def euclidean_distance(p, q):
    d = p - q
    return np.sqrt(np.dot(d, d))


def generate_random_trajectory(arena, max_distance=None, spacing=10.0, temporal_resolution=1., velocity=30., equilibration_duration=None, local_random=None, n_trials=1):
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

    if local_random is None:
        local_random = np.random.RandomState(0)
    
    vert, smp = arena

    # generate the nodes. `nodes` is a (N, 2) float array, `groups` is a dict
    # identifying which group each node is in.
    nodes, groups, _ = poisson_disc_nodes(spacing, (vert, smp))
    N = nodes.shape[0]

    tri = Delaunay(nodes)
    G = nx.Graph()
    for path in tri.simplices:
        nx.add_path(G, path)
    
    # Choose a starting point
    path_distance = 0.
    path_nodes = []
    p = None
    s = local_random.choice(N)
    visited = deque(maxlen=5)
    while ((max_distance is None) or (path_distance < max_distance)):

        neighbors = list(G.neighbors(s))
        visited.append(s)
        path_nodes.append(s)
        if p is not None:
            path_distance = path_distance + euclidean_distance(nodes[s], nodes[p])
        p = s
        print(path_distance)
        
        candidates = [ n for n in neighbors if n not in visited ]
        if len(candidates) == 0:
            break
        else:
            selected = local_random.choice(len(candidates))
            s = candidates[selected]

    input_trajectory = nodes[path_nodes]
            
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
    

def generate_input_rates(spatial_domain, module_field_width_dict, basis_function='gaussian', spacing_factor=1.0, peak_rate=1.):
    input_nodes_dict = {}
    input_groups_dict = {}
    input_rates_dict = {}

    n_modules = len(module_field_width_dict)
    if isinstance(spacing_factor, float):
        spacing_factors = [spacing_factor]*n_modules
    else:
        spacing_factors = spacing_factor
    vert, smp = spatial_domain
    for m in sorted(module_field_width_dict):
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
                                     epsilon=module_field_width_dict[m] * spacing_factors[m])
            input_rates_dict[m][i] = input_rate_ip
            
    return input_nodes_dict, input_groups_dict, input_rates_dict
