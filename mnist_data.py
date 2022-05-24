import numpy as np
import math
import os, time
import struct
from nengo_extras import data

cwd = (os.path.dirname(os.path.realpath(__file__)))

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

    
def load_mnist(dataset="train", digits=np.arange(10), path=".", size = 60000):

    train_set, test_set = data.load_mnist(f'{path}/mnist.pkl.gz')
    if dataset == "train":
        img, lbl = train_set
        num, n_pixels = img.shape
        n_dim = int(math.sqrt(n_pixels))
        rows, cols = n_dim, n_dim
    elif dataset == "test":
        img, lbl = test_set
        num, n_pixels = img.shape
        n_dim = int(math.sqrt(n_pixels))
        rows, cols = n_dim, n_dim
    else:
        raise ValueError("dataset must be 'test' or 'train'")
    
    N = size * len(digits)

    images = np.zeros((N, rows, cols), dtype='float32')
    labels = np.zeros((N, 1), dtype='int8')
    
    for i, label in enumerate(digits):
        ind = []
        for j, l in enumerate(lbl):
            if len(ind) >= size:
                break
        
            if l == label:
                ind.append(j)
        
        for j in range(size): #int(len(ind) * size/100.)):
            images[i * size + j] = np.array(img[ind[j]].reshape((rows, cols)))
            labels[i * size + j] = lbl[ind[j]]

    labels = np.array([label[0] for label in labels])

    rand = np.random.permutation(N)
    labels = labels[rand]
    images = images[rand]

    return images, labels
    



def generate_inputs(train_size=5000, test_size=500, plot=False, seed=None, dataset='train', digits=np.arange(0, 10)):

    if seed == None:
        np.random.seed(int((time.time() * 1000000000 ) % (2**32 - 1)))
    else:
        np.random.seed(seed)

    num_imgs = None
    if dataset == 'train':
        imgs, lbls = load_mnist(path=cwd + '/data/mnist', digits=digits, size=train_size, dataset='train')
        num_imgs = imgs.shape[0]
    elif dataset == 'test':    
        test_imgs, test_lbls = load_mnist(path=cwd + '/data/mnist', digits=digits, size=test_size, dataset='test')
        num_imgs = test_imgs.shape[0]
        
    if dataset == 'train':
        stim_idx = np.random.randint(0, high=num_imgs, size=train_size)
    elif dataset == 'test':
        stim_idx = np.random.randint(0, high=num_imgs, size=test_size)

    if dataset == 'train':
        inp = imgs[stim_idx]
        lbl = lbls[stim_idx]
    elif dataset == 'test':
        inp = test_imgs[stim_idx]
        lbl = test_lbls[stim_idx]
    else:
        print("Please select 'train' or 'test' for dataset")


    if plot:
        show(inp[0:20])

    return inp, lbl
    

