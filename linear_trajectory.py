
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
from scipy.interpolate import Rbf, Akima1DInterpolator
from rbf.pde.nodes import disperse, poisson_disc_nodes


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
    

def generate_input_rates(spatial_domain, module_field_width_dict, basis_function='gaussian', spacing_factor=1.0, peak_rate=1.):
    input_nodes_dict = {}
    input_groups_dict = {}
    input_rates_dict = {}

    n_modules = len(module_field_width_dict)
    if isinstance(spacing_factor, float):
        spacing_factors = [spacing_factor]*n_modules
    else:
        spacing_factors = spacing_factor
    print(f"spacing factors: {spacing_factors}")
    vert, smp = spatial_domain
    for m in sorted(module_field_width_dict):
        
        nodes, groups, _ = poisson_disc_nodes(module_field_width_dict[m], (vert, smp))
        print(f"Generated {nodes.shape[0]} nodes for field width {module_field_width_dict[m]}")
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
