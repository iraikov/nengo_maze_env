import nengo.utils.numpy as npext
import numpy as np
from nengo.dists import Choice, DistributionParam, Uniform
from nengo.params import TupleParam
from nengo_extras.vision import Mask, Gabor


class SequentialMask(Mask):
    def __init__(self, image_shape, strides=(1, 1)):
        super().__init__(image_shape)
        self.strides = strides

    def _positions(self, n, shape, rng):
        si, sj = np.asarray(self.image_shape[1:]) - np.asarray(shape) + 1
        sti, stj = self.strides

        # pick sequential positions, moving left->right then top->bottomw
        nj = len(np.arange(0, sj, stj))

        i = ((np.arange(0, n) // nj) * sti) % si
        j = (np.arange(0, n) * stj) % sj
        return i, j
    
