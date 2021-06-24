import numpy as np
import os, time
import struct

cwd = (os.path.dirname(os.path.realpath(__file__)))

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

    
def load_mnist(dataset="training", digits=np.arange(10), path=".", size = 60000):

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    
    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    
    
    N = size * len(digits)

    images = np.zeros((N, rows, cols), dtype='uint8')
    labels = np.zeros((N, 1), dtype='int8')
    
    for i, label in enumerate(digits):
        ind = []
        for j, l in enumerate(lbl):
            if len(ind) >= size:
                break
        
            if l == label:
                ind.append(j)
        
        for j in range(size): #int(len(ind) * size/100.)):
            images[i * size + j] = np.array(img[ind[j]])
            labels[i * size + j] = lbl[ind[j]]

    labels = np.array([label[0] for label in labels])

    rand = np.random.permutation(N)
    labels = labels[rand]
    images = images[rand]

    return images, labels



def generate_inputs(train_size=5000, test_size=500, plot=False, seed=None, dataset='training', digits=np.arange(0, 10)):

    if seed == None:
        np.random.seed(int((time.time() * 1000000000 ) % (2**32 - 1)))
    else:
        np.random.seed(seed)

    num_imgs = None
    if dataset == 'training':
        imgs, lbls = load_mnist(path=cwd + '/data/mnist', digits=digits, size=train_size)
        num_imgs = imgs.shape[0]
    elif dataset == 'testing':    
        test_imgs, test_lbls = load_mnist(path=cwd + '/data/mnist', digits=digits, size=test_size, dataset='testing')
        num_imgs = test_imgs.shape[0]
        
    if dataset == 'training':
        stim_idx = np.random.randint(0, high=num_imgs, size=train_size)
    elif dataset == 'testing':
        stim_idx = np.random.randint(0, high=num_imgs, size=test_size)

    if dataset == 'training':
        inp = imgs[stim_idx]
        lbl = lbls[stim_idx]
    elif dataset == 'testing':
        inp = test_imgs[stim_idx]
        lbl = test_lbls[stim_idx]
    else:
        print("Please select 'training' or 'testing' for dataset")


    if plot:
        show(inp[0])

    return inp, lbl
    

