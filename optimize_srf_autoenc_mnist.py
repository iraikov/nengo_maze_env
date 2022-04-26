import sys, gc
import nengo
import numpy as np
import scipy.interpolate
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo.utils.progress import TerminalProgressBar
from nengo.dists import Choice, Uniform
from nengo_extras.dists import Tile
from nengo_extras.matplotlib import tile
from mnist_data import generate_inputs
from srf_autoenc import build_network, run
from sklearn.linear_model import LogisticRegression
from input_mask import InputMask, Gabor
from dmosopt import dmosopt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mse(rates, target_rates):
    mses = []
    for i in range(rates.shape[1]):
        rates_i = rates[:, i]
        target_rates_i = target_rates[:, i]
        mean_error = np.mean(rates_i - target_rates_i)
        mse = mean_error ** 2.
        mses.append(mse)
        #logger.info(f"modulation_depth {i}: peak_pctile: {peak_pctile} med_pctile: {med_pctile} mod_depth: {mod_depth}")
    return mses
    

    
def obj_modulation_depth(rates):
    mod_depths = []
    for i in range(rates.shape[1]):
        rates_i = rates[:, i]
        peak_pctile = np.percentile(rates_i, 80)
        med_pctile = np.percentile(rates_i, 50)
        peak_idxs = np.argwhere(rates_i >= peak_pctile)
        med_idxs = np.argwhere(rates_i <= med_pctile)
        mean_peak = np.mean(rates_i[peak_idxs])
        mean_med = np.mean(rates_i[med_idxs])
        mod_depth = (mean_peak - mean_med) ** 2.
        mod_depths.append(mod_depth)
        #logger.info(f"modulation_depth {i}: peak_pctile: {peak_pctile} med_pctile: {med_pctile} mod_depth: {mod_depth}")
    res = -np.mean(mod_depths)
    return res


def obj_fraction_active(rates, target=0.1):
    n = rates.shape[1]
    bin_fraction_active = []
    for i in range(rates.shape[0]):
        rates_i = rates[i, :]
        a = len(np.argwhere(rates_i >= 0.1))
        bin_fraction_active.append(float(a) / float(n))
        #logger.info(f"fraction_active {i}: a: {a} n: {n} fraction: {float(a) / float(n)}")
    res = (np.mean(bin_fraction_active) - target)**2
    return res


def fit_labels(rates, labels, n_samples, return_model=False, rate_sample_fraction=1.0, model_kwargs={}):
    """
    Assign labels to the neurons based on highest average spiking activity.
    """
    assert (rate_sample_fraction <= 1.0) and (rate_sample_fraction > 0.)
    n_steps, n_neurons = rates.shape
    rate_samples = [x[int(round(n_samples*(1.0 - rate_sample_fraction))):,:]
                    for x in np.split(rates, n_steps//n_samples)]

    X = [np.max(rate, axis=0) for rate in rate_samples]
    y = labels
    if not 'tol' in model_kwargs:
        model_kwargs['tol'] = 0.01
    if not 'C' in model_kwargs:
        model_kwargs['C'] = 1.
    if not 'penalty' in model_kwargs:
        model_kwargs['penalty'] = 'l1'
    reg = LogisticRegression(multi_class="multinomial", solver="saga", **model_kwargs)
    reg = reg.fit(X, y)

    score = reg.score(X, y)
    if return_model:
        return score, reg
    else:
        return score


def predict_labels(model, rates, labels, n_samples, rate_sample_fraction=1.0):

    n_steps, n_neurons = rates.shape
    rate_samples = [x[int(round(n_samples*(1.0 - rate_sample_fraction))):,:]
                    for x in np.split(rates, n_steps//n_samples)]

    X = [np.max(rate, axis=0) for rate in rate_samples]
    y = labels
    score = model.score(X, y)
    prediction = model.predict(X)
    
    return prediction, score
        


            
def eval_srf_autoenc(params, input_dimensions, input_data, input_encoders,
                     train_labels, test_labels, n_train, n_test, n_frame_steps,
                     t_train, t_end, coords_dict, seed, dt=0.01):

    model_dict = build_network(params, dimensions=input_dimensions, inputs=input_data,
                               input_encoders=input_encoders, direct_input=False,
                               oob_value=0., coords=coords_dict, t_learn=t_train, dt=dt)
    
    sim, sim_output_dict = run(model_dict, t_end, dt=dt, progress_bar=False, save_results=False)

    srf_autoenc_exc_rates_train = sim_output_dict['srf_autoenc_exc_rates'][:n_train]
    srf_autoenc_exc_rates_test = sim_output_dict['srf_autoenc_exc_rates'][n_train:]
    
    srf_autoenc_decoder_rates_train = sim_output_dict['srf_autoenc_decoder_rates'][:n_train]
    srf_autoenc_decoder_rates_test = sim_output_dict['srf_autoenc_decoder_rates'][n_train:]
    
    srf_autoenc_output_rates_train = sim_output_dict['srf_autoenc_output_rates'][:n_train]
    srf_autoenc_output_rates_test = sim_output_dict['srf_autoenc_output_rates'][n_train:]

    output_r2, output_reg = fit_labels(srf_autoenc_output_rates_train[n_frame_steps:,:],
                                       train_labels[1:], n_frame_steps, return_model=True)

    _, train_score = predict_labels(output_reg, srf_autoenc_output_rates_train[n_frame_steps:,:],
                                    train_labels[1:], n_frame_steps)
    _, test_score = predict_labels(output_reg, srf_autoenc_output_rates_test,
                                   test_labels, n_frame_steps)

    return np.asarray([obj_modulation_depth(srf_autoenc_output_rates_train),
                       obj_modulation_depth(srf_autoenc_output_rates_test), 
                       obj_fraction_active(srf_autoenc_output_rates_train),
                       obj_fraction_active(srf_autoenc_output_rates_test),
                       np.mean(mse(srf_autoenc_decoder_rates_train, srf_autoenc_exc_rates_train)),
                       np.mean(mse(srf_autoenc_decoder_rates_test, srf_autoenc_exc_rates_test)),
                       (1.0 - train_score)*100, (1.0 - test_score)*100])



def init_obj_fun(worker, dt):
    """ Returns objective function to be minimized. """

    seed = 29
    n_frame_steps = 100
    rng = np.random.RandomState(seed)
    
    input_data = None
    n_x = None
    n_y = None
    n_train = None
    n_test = None
    train_labels = None
    test_labels = None
    
    if worker.worker_id == 1:
        train_image_array, train_labels = generate_inputs(plot=False, train_size=20, dataset='train', seed=seed)
        test_image_array, test_labels = generate_inputs(plot=False, test_size=10, dataset='test', seed=seed)

        n_x = train_image_array.shape[1]
        n_y = train_image_array.shape[2]

        train_image_array = np.concatenate((train_image_array[0].reshape(1,n_x,n_y), train_image_array), axis=0)
        train_labels = np.concatenate((np.asarray([train_labels[0]]), train_labels))

        normed_train_image_array = train_image_array / np.max(train_image_array)
        normed_test_image_array = test_image_array / np.max(test_image_array)
        train_data = np.repeat(normed_train_image_array, n_frame_steps, axis=0)
        test_data = np.repeat(normed_test_image_array, n_frame_steps, axis=0)

        n_train = train_data.shape[0]
        n_test = test_data.shape[0]
        
        input_data = np.concatenate((train_data, test_data), axis=0)

    input_data = worker.group_comm.bcast(input_data, root=0)
    n_x, n_y = worker.group_comm.bcast((n_x, n_y), root=0)
    n_train, n_test = worker.group_comm.bcast((n_train, n_test), root=0)
    train_labels, test_labels = worker.group_comm.bcast((train_labels, test_labels), root=0)
    n_exc=n_x*n_y

    n_outputs=1000
    n_exc=n_x*n_y
    n_inh=200
        
    srf_exc_coords = np.asarray(range(n_exc)).reshape((n_exc,1)) / n_exc
    srf_inh_coords = np.asarray(range(n_inh)).reshape((n_inh,1)) / n_inh
    srf_output_coords = np.asarray(range(n_outputs)).reshape((n_outputs,1)) / n_outputs
    
    coords_dict = { 'srf_output': srf_output_coords,
                    'srf_exc': srf_exc_coords,
                    'srf_inh': srf_inh_coords, }
    
    t_train = float(n_train) * dt
    t_test = float(n_test) * dt
    t_end = t_train + t_test
    
    gabor_size = (7, 7) # Size of the gabor filter for image inputs
    
    # Generate the encoders for the sensory ensemble
    input_dimensions = n_x*n_y
    n_gabor = 9
    
    input_encoders = Gabor(sigma_x=Choice([0.45]),
                           sigma_y=Choice([0.45]),
                           theta=Tile(np.linspace(-np.pi,np.pi,n_gabor)),
                           freq=Tile(np.linspace(0.5,2,n_gabor)),
                           phase=Tile(np.linspace(-np.pi,np.pi,n_gabor)),).generate(n_exc, gabor_size, rng=rng)
    input_encoders = InputMask((n_x, n_y)).populate(input_encoders, rng=rng, random_positions=False, flatten=True)
    

    def obj_fun(pp, pid=None):
        res = eval_srf_autoenc(pp, input_dimensions, input_data, input_encoders,
                               train_labels, test_labels, n_train, n_test, 
                               n_frame_steps, t_train, t_end, coords_dict, pp['seed'], dt=dt)
        logger.info(f"Iter: {pid}\t pp:{pp}, result:{res}")
        return res

    return obj_fun
    
if __name__ == '__main__':

    dt = 0.01
    seed = 21
    
    problem_parameters = {'seed': seed,
                          'tau_input': 0.01,
                          'isp_target_rate': 0.1,
                          'w_DEC_E': 0.005, 
                          'w_DEC_I': 0.002,
                          'w_initial_I_DEC_fb': -0.05, 
                          'p_DEC': 0.1, 
                          'tau_E': 0.0025, 
                          'tau_I': 0.010, 
                          'tau_input': 0.005,
                          'learning_rate_D': 0.08,
                          'learning_rate_D_Exc': 0.005
                         }
        
    space = {'w_initial_E': [1e-5, 1e-1],
             'w_initial_EI': [1e-5, 1e-1],
             'w_initial_EE': [1e-5, 1e-1],
             'w_initial_I': [-1e-1, -1e-5],
             'w_EI_Ext': [1e-5, 1e-1],
             'p_E_srf': [0.001, 0.25],
             'p_I_srf': [0.001, 0.25],
             'p_EE': [0.01, 0.25],
             'p_EI': [0.01, 0.25],
             'p_EI_Ext': [0.001, 0.1],
             'tau_E': [0.005, 0.05],
             'tau_I': [0.01, 0.05],
             'learning_rate_I': [1e-4, 1e-1],
             'learning_rate_E': [1e-4, 1e-1],
             'learning_rate_EE': [1e-4, 1e-1],
            }

    objective_names = ['srf_modulation_depth_train',
                       'srf_modulation_depth_test',
                       'srf_fraction_active_train',
                       'srf_fraction_active_test',
                       'decoder_mse_train',
                       'decoder_mse_test',
                       'train_score',
                       'test_score']

    dmosopt_params = {'opt_id': 'dmosopt_srf_autoenc_mnist',
                      'obj_fun_init_name': 'init_obj_fun',
                      'obj_fun_init_args': {'dt': dt},
                      'obj_fun_init_module': 'optimize_srf_autoenc_mnist',
                      'problem_parameters': problem_parameters,
                      'optimize': 'nsga2',
                      'space': space,
                      'n_iter': 10,
                      'objective_names': objective_names,
                      'n_initial': 100,
                      'resample_fraction': 1.0,
                      'save': True,
                      'file_path': True,
                      'file_path': 'dmosopt.srf_autoenc_mnist.h5',
                  }
    
    best = dmosopt.run(dmosopt_params, spawn_workers=False, verbose=True)
            
