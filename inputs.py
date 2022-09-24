import nengo.utils.numpy as npext
import numpy as np
from nengo import Process
from nengo.params import FrozenObject, TupleParam, NdarrayParam, NumberParam
from nengo.dists import Choice, DistributionParam, Uniform
from nengo_extras.vision import Mask, Gabor

class PresentInputWithPause(Process):
    """Present a series of inputs, each for the same fixed length of time.

    Parameters
    ----------
    inputs : array_like
        Inputs to present, where each row is an input. Rows will be flattened.
    presentation_time : float
        Show each input for this amount of time (in seconds).
    pause_time : float
        Pause time after each input (in seconds).
    """

    inputs = NdarrayParam("inputs", shape=("...",))
    presentation_time = NumberParam("presentation_time", low=0, low_open=True)
    pause_time = NumberParam("pause_time", low=0, low_open=True)

    def __init__(self, inputs, presentation_time, pause_time, reduce_op=None, **kwargs):
        self.inputs = inputs
        self.presentation_time = presentation_time
        self.pause_time = pause_time
        self.localT = 0
        self.reduce_op = reduce_op
        input0 = self.inputs[0]
        self.size_out = reduce_op(input0).size if reduce_op is not None else input0.size
        super().__init__(
            default_size_in=0, default_size_out=self.size_out, **kwargs
        )
        
    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)
        assert shape_out == (self.size_out,)
        n = len(self.inputs)
        inputs = self.inputs.reshape(n, -1)
        presentation_time = float(self.presentation_time)
        pause_time = float(self.pause_time)
        reduce_op = self.reduce_op
        
        def step_presentinput(t):
            t = round(t,6)
            # Pause
            total_time = presentation_time + pause_time
            i = int((t - dt) / total_time + 1e-7)
            ti = t % total_time
            output = np.zeros_like(inputs[0])
            if ti <= presentation_time:
                output = inputs[i % n]
                if reduce_op is not None:
                    output = reduce_op(output)
            return output
        
        return step_presentinput


class SequentialMask(Mask):
    def __init__(self, image_shape, strides=(1, 1)):
        super().__init__(image_shape)
        self.strides = strides

    def _positions(self, n, shape, rng):
        si, sj = np.asarray(self.image_shape[1:]) - np.asarray(shape) + 1
        sti, stj = self.strides

        # pick sequential positions, moving left->right then top->bottom
        nj = len(np.arange(0, sj, stj))

        i = ((np.arange(0, n) // nj) * sti) % si
        j = (np.arange(0, n) * stj) % sj
        return i, j

    def populate(self, filters, rng=np.random, flatten=False, return_positions=False):
        filters = np.asarray(filters)
        assert filters.ndim in [3, 4]
        n, shape = filters.shape[0], filters.shape[-2:]
        channels = 1 if filters.ndim == 3 else filters.shape[1]
        assert channels == self.image_shape[0]

        i, j = self._positions(n, shape, rng)
        output = np.zeros((n,) + self.image_shape, dtype=filters.dtype)
        for k in range(n):
            output[k, :, i[k] : i[k] + shape[0], j[k] : j[k] + shape[1]] = filters[k]

        if return_positions:
            return output.reshape((n, -1)) if flatten else output, (i, j)
        else:
            return output.reshape((n, -1)) if flatten else output

    
class OOCS(FrozenObject):
    """
    On-Off Center-Surround Receptive Fields.
    https://github.com/ranaa-b/OOCS
    Compute weight at location (x, y) in the OOCS kernel with given parameters
        Parameters:
            x, y : position of the current weight
            center : position of the kernel center
            gamma : center to surround ratio
            radius : center radius
        Returns:
            filters : 
    """

    x = DistributionParam("y")
    y = DistributionParam("y")
    gamma = DistributionParam("gamma")
    radius = DistributionParam("radius")

    def __init__(
            self,
            x=Choice([0.0]),
            y=Choice([0.0]),
            gamma=Choice([0.1]),
            radius=Uniform(0.1, 1.0)
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.gamma = gamma
        self.radius = radius


    def generate(self, n, shape, rng=np.random, on_off_frac=0.5, norm=2.0):

        assert isinstance(shape, tuple) and len(shape) == 2
        radii = self.radius.sample(n, rng=rng)[:, None, None]
        gammas = self.gamma.sample(n, rng=rng)[:, None, None]
        center_xs = self.x.sample(n, rng=rng)[:, None, None]
        center_ys = self.y.sample(n, rng=rng)[:, None, None]

        x, y = np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0])
        X, Y = np.meshgrid(x, y)

        # compute sigma from radius of the center and gamma (center to surround ratio)
        sigmas = (radii / (2.0 * gammas)) * (np.sqrt((1. - gammas ** 2) / (-np.log(gammas))))
        excitation = (1.0 / (gammas ** 2)) * np.exp(-1.0 * ((X - center_xs) ** 2 + (Y - center_ys) ** 2) / (2 * ((gammas * sigmas) ** 2)))
        inhibition = np.exp(-1 * ((X - center_xs) ** 2 + (Y - center_ys) ** 2) / (2.0 * (sigmas ** 2)))

        filters = excitation + inhibition

        if norm is not None:
            filters *= norm / np.sqrt(
                (filters ** 2).sum(axis=(1, 2), keepdims=True)
            ).clip(1e-5, np.inf)

        filters *= rng.choice([1.0, -1.0], size=n, p=[on_off_frac, 1.0 - on_off_frac]).reshape((-1,1,1))
        
        return filters
