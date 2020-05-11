import nengo
import numpy as np

# Loads a Nengo decoder from a file, defaulting to zero if it doesn't exist
class LoadFrom(nengo.solvers.Solver):
    def __init__(self, filename, weights=False):
        super(LoadFrom, self).__init__(weights=weights)
        self.filename = filename
            
    def __call__(self, A, Y, rng=None, E=None):
        if self.weights:
            shape = (A.shape[1], E.shape[1])
        else:
            shape = (A.shape[1], Y.shape[1])

        try:
            value = np.load(self.filename)
            assert value.shape == shape
        except IOError:
            value = np.zeros(shape)
        return value, {}

# helper to create the LoadFrom solver and the needed probe and do the saving
class WeightSaver(object):
    def __init__(self, connection, filename, sample_every=1.0, weights=False):
        assert (isinstance(connection.pre, nengo.Ensemble) or isinstance(connection.pre, nengo.ensemble.Neurons) )
        if not filename.endswith('.npy'):
            filename = filename + '.npy'
        self.filename = filename
        connection.solver = LoadFrom(self.filename, weights=weights)
        self.probe = nengo.Probe(connection, 'weights', sample_every=sample_every)
        self.connection = connection
    def save(self, sim):
        np.save(self.filename, sim.data[self.probe][-1].T)
