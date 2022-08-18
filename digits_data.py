import numpy as np
import math
import os, time
import struct

from sklearn.datasets import load_digits

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
def show(images):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    from nengo_extras.matplotlib import tile
    image_array = np.stack(images)
    show_params = {'interpolation': 'nearest'}
    tile(image_array, rows=4, cols=5, **show_params)
    #ax.xaxis.set_ticks_position('top')
    #ax.yaxis.set_ticks_position('left')
    pyplot.show()

    


def generate_inputs(train_size=1437, test_size=360, plot=False, seed=None, dataset='train', n_class=10):

    if seed == None:
        np.random.seed(int((time.time() * 1000000000 ) % (2**32 - 1)))
    else:
        np.random.seed(seed)

    num_imgs = None
    data = load_digits(n_class=n_class)
    
    imgs, lbls = data.images, data.target
    num_imgs = imgs.shape[0]
        
    if dataset == 'train':
        stim_idx = np.random.randint(0, high=train_size, size=train_size)
    elif dataset == 'test':
        stim_idx = np.random.randint(train_size, high=num_imgs, size=test_size)

    if dataset == 'train':
        inp = imgs[stim_idx]
        lbl = lbls[stim_idx]
    elif dataset == 'test':
        inp = imgs[stim_idx]
        lbl = lbls[stim_idx]
    else:
        print("Please select 'train' or 'test' for dataset")

    if plot:
        show(inp[0:20])

    return inp, lbl
