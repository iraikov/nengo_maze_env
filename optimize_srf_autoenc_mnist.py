import sys, gc
import nengo
import numpy as np
import scipy.interpolate
from nengo_extras.neurons import (
    rates_kernel, rates_isi, spikes2events )
from nengo.dists import Choice, Uniform
from nengo_extras.dists import Tile
from nengo_extras.matplotlib import tile
from mnist_data import generate_inputs
from srf_autoenc import build_network, run
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from input_mask import Mask, SequentialMask, Gabor
from decoding import predict_ngram, fit_ngram_decoder, predict_ngram_rates, fit_ngram_decoder_rates
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
    

def modulation_depth(rates):
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
    return mod_depths

def fraction_active(rates):
    n = rates.shape[1]
    bin_fraction_active = []
    for i in range(rates.shape[0]):
        rates_i = rates[i, :]
        a = len(np.argwhere(rates_i >= 0.1))
        bin_fraction_active.append(float(a) / float(n))
        #logger.info(f"fraction_active {i}: a: {a} n: {n} fraction: {float(a) / float(n)}")
    return bin_fraction_active
    
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


            
def eval_srf_autoenc(params, input_dimensions, input_data, input_encoders,
                     train_labels, test_labels, presentation_time, pause_time, skip_time,
                     t_train, t_end, coords_dict, seed, ngram_n=2, dt=0.01):

    model_dict = build_network(params, dimensions=input_dimensions, inputs=input_data,
                               input_encoders=input_encoders, direct_input=False,
                               presentation_time=presentation_time, pause_time=pause_time,
                               coords=coords_dict, t_learn_exc=t_train, t_learn_inh=t_train,
                               sample_weights_every=t_end // 5.0)

    sim, sim_output_dict = run(model_dict, t_end, dt=dt, progress_bar=False, save_results=False)

    train_size = len(train_labels)
    test_size = len(test_labels)
    n_steps_frame = int((presentation_time + pause_time) / dt)
    n_steps_present = int((presentation_time) / dt)
    n_steps_skip = int(skip_time / dt)
    n_steps_train = n_steps_frame * (train_size+1)
    n_steps_test = n_steps_frame * test_size
    n_labels = len(np.unique(train_labels))
    
    srf_autoenc_output_spikes_train = sim_output_dict['srf_autoenc_output_spikes'][:n_steps_train]
    srf_autoenc_output_spikes_test = sim_output_dict['srf_autoenc_output_spikes'][n_steps_train:]
    
    srf_autoenc_exc_rates_train = sim_output_dict['srf_autoenc_exc_rates'][:n_steps_train]
    srf_autoenc_exc_rates_test = sim_output_dict['srf_autoenc_exc_rates'][n_steps_train:]
    
    srf_autoenc_decoder_rates_train = sim_output_dict['srf_autoenc_decoder_rates'][:n_steps_train]
    srf_autoenc_decoder_rates_test = sim_output_dict['srf_autoenc_decoder_rates'][n_steps_train:]
    
    srf_autoenc_output_rates_train = sim_output_dict['srf_autoenc_output_rates'][:n_steps_train]
    srf_autoenc_output_rates_test = sim_output_dict['srf_autoenc_output_rates'][n_steps_train:]

    example_spikes_train = [x[n_steps_skip:n_steps_present]
                            for x in np.split(srf_autoenc_output_spikes_train[1*n_steps_frame:,:], (n_steps_train - 1*n_steps_frame)//n_steps_frame)]
    example_spikes_test = [x[n_steps_skip:n_steps_present]
                           for x in np.split(srf_autoenc_output_spikes_test, n_steps_test/n_steps_frame)]

    
    ngram_decoder = fit_ngram_decoder(example_spikes_train, train_labels, n_labels, ngram_n, {})
    output_predictions_train = predict_ngram(example_spikes_train, ngram_decoder, n_labels, ngram_n)
    output_train_score = accuracy_score(train_labels, output_predictions_train)

    output_predictions_test = predict_ngram(example_spikes_test, ngram_decoder, n_labels, ngram_n)
    output_test_score = accuracy_score(test_labels, output_predictions_test)

    logger.info(f'output modulation depth (train): {np.mean(modulation_depth(srf_autoenc_output_rates_train))}\n'
                f'output modulation depth (test): {np.mean(modulation_depth(srf_autoenc_output_rates_test))}\n'
                f'decoder modulation depth (train): {np.mean(modulation_depth(srf_autoenc_decoder_rates_train))}\n'
                f'decoder modulation depth (test): {np.mean(modulation_depth(srf_autoenc_decoder_rates_train))}\n'
                f'input fraction active: {np.mean(fraction_active(srf_autoenc_exc_rates_train))}\n'
                f"output fraction active (train): {np.mean(fraction_active(srf_autoenc_output_rates_train))}\n"
                f"output fraction active (test): {np.mean(fraction_active(srf_autoenc_output_rates_test))}\n"
                f"decoder fraction active (train): {np.mean(fraction_active(srf_autoenc_decoder_rates_train))}\n"
                f"decoder mse (train): {np.mean(mse(srf_autoenc_decoder_rates_train, srf_autoenc_exc_rates_train))}\n"
                f"output score (train): {output_train_score}\n"
                f"output score (test): {output_test_score}\n")

    return np.asarray([obj_modulation_depth(srf_autoenc_output_rates_train),
                       obj_modulation_depth(srf_autoenc_output_rates_test), 
                       obj_fraction_active(srf_autoenc_output_rates_train),
                       obj_fraction_active(srf_autoenc_output_rates_test),
                       np.mean(mse(srf_autoenc_decoder_rates_train, srf_autoenc_exc_rates_train)),
                       np.mean(mse(srf_autoenc_decoder_rates_test, srf_autoenc_exc_rates_test)),
                       (1.0 - output_train_score)*100,
                       (1.0 - output_test_score)*100])



def init_obj_fun(worker, seed, train_size, test_size, presentation_time, pause_time, skip_time, n_outputs, n_exc, n_inh, dt):
    """ Returns objective function to be minimized. """

    rng = np.random.RandomState(seed)
    
    input_data = None
    n_x = None
    n_y = None

    train_labels = None
    test_labels = None
    
    if worker is None or worker.worker_id == 1:

        train_image_array, train_labels = generate_inputs(plot=False, train_size=train_size, dataset='train', seed=seed)
        test_image_array, test_labels = generate_inputs(plot=False, test_size=test_size, dataset='test', seed=seed)

        n_x = train_image_array.shape[1]
        n_y = train_image_array.shape[2]

        normed_train_image_array = train_image_array / np.max(train_image_array)
        normed_test_image_array = test_image_array / np.max(train_image_array)

        input_data = np.concatenate((normed_train_image_array, normed_test_image_array), axis=0)

    input_data = worker.group_comm.bcast(input_data, root=0)
    n_x, n_y = worker.group_comm.bcast((n_x, n_y), root=0)
    train_labels, test_labels = worker.group_comm.bcast((train_labels, test_labels), root=0)
    input_dimensions = n_x*n_y

    srf_exc_coords = np.asarray(range(n_exc)).reshape((n_exc,1)) / n_exc - 0.5
    srf_inh_coords = np.asarray(range(n_inh)).reshape((n_inh,1)) / n_inh - 0.5
    srf_output_coords = (np.asarray(range(n_outputs)).reshape((n_outputs,1)) / n_outputs)*0.9 - 0.45

    coords_dict = { 'srf_output': srf_output_coords,
                    'srf_exc': srf_exc_coords,
                    'srf_inh': srf_inh_coords,
    }
    
    input_encoders = rng.normal(size=(n_exc, 2, 2))
    input_encoders = Mask((n_x, n_y)).populate(input_encoders, rng=rng, flatten=True)
    
    t_train = (train_size+1)*(presentation_time + pause_time)
    t_test = test_size*(presentation_time + pause_time)
    t_end = t_train + t_test

    def obj_fun(pp, pid=None):
        res = eval_srf_autoenc(pp, input_dimensions, input_data, input_encoders,
                               train_labels, test_labels, presentation_time, pause_time, skip_time,
                               t_train, t_end, coords_dict, pp['seed'], ngram_n=2, dt=dt)
        logger.info(f"Iter: {pid}\t pp:{pp}, result:{res}")
        return res

    return obj_fun
    
if __name__ == '__main__':

    dt = 0.01
    seed = 21

    train_size=600
    test_size=60
    
    n_outputs=600
    n_exc=200
    n_inh=100

    presentation_time=0.5
    pause_time=0.2
    skip_time=0.03
    
    problem_parameters = {'seed': seed,
                          'tau_input': 0.01,
                          'w_DEC_E': 0.005, 
                          'w_DEC_I': 0.002,
                          'w_initial_I_DEC_fb': -0.05, 
                          'p_DEC': 0.1, 
                          'learning_rate_D': 0.1,
                          'learning_rate_D_Exc': 0.005
                         }
        
    space = {'w_initial_E': [1e-5, 1e-1],
             'w_initial_EI': [1e-5, 1e-1],
             'w_initial_EE': [1e-5, 1e-1],
             'w_initial_I': [-1e-1, -1e-5],
             'w_EI_Ext': [1e-5, 1e-1],
             'p_E_srf': [0.1, 0.5],
             'p_I_srf': [0.1, 0.8],
             'p_EE': [0.05, 0.5],
             'p_EI': [0.1, 0.5],
             'p_EI_Ext': [0.1, 0.5],
             'tau_E': [0.005, 0.05],
             'tau_I': [0.01, 0.05],
             'learning_rate_I': [1e-5, 1e-1],
             'learning_rate_E': [1e-5, 1e-1],
             'learning_rate_EE': [1e-5, 1e-1],
             'isp_target_rate': [1.0, 10.0]
            }

    objective_names = ['srf_modulation_depth_train',
                       'srf_modulation_depth_test',
                       'srf_fraction_active_train',
                       'srf_fraction_active_test',
                       'decoder_mse_train',
                       'decoder_mse_test',
                       'train_score',
                       'test_score']

    init_args = {'seed': seed,
                 'train_size': train_size,
                 'test_size': test_size,
                 'presentation_time': presentation_time,
                 'pause_time': pause_time,
                 'skip_time': skip_time,
                 'n_outputs': n_outputs,
                 'n_exc': n_exc,
                 'n_inh': n_inh,
                 'dt': dt }
    
    dmosopt_params = {'opt_id': 'dmosopt_srf_autoenc_mnist',
                      'obj_fun_init_name': 'init_obj_fun',
                      'obj_fun_init_args': init_args,
                      'obj_fun_init_module': 'optimize_srf_autoenc_mnist',
                      'problem_parameters': problem_parameters,
                      'optimize': 'nsga2',
                      'space': space,
                      'n_iter': 10,
                      'surogate_method': 'siv',
                      'objective_names': objective_names,
                      'n_initial': 100,
                      'resample_fraction': 1.0,
                      'save': True,
                      'file_path': True,
                      'file_path': 'dmosopt.srf_autoenc_mnist.h5',
                  }
    
    best = dmosopt.run(dmosopt_params, spawn_workers=False, verbose=True)
            
