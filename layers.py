import theano
import theano.tensor as T
import numpy as np


class InputLayer(object):
    """
    Input layer to theano autoencoder
    """

    def __init__(self, n_vis):
        self.n_vis = n_vis
        self.output_shape = self.n_vis
        self.input = T.matrix('input')
        self.params = []
        self.output = self.input

    def setup(self, *args, **kwargs):
        pass

class HiddenLayer(object):
    """
    Hidden layer for theano autoencoder
    """

    def __init__(self, n_outputs, activation):
        self.activation = activation
        self.output_shape = n_outputs

    def setup(self, prev_layer):
        self.input_layer = prev_layer
        self.input = prev_layer.output
        self.W = theano.shared(np.random.random((self.input_layer.output_shape, self.output_shape)).astype(theano.config.floatX)*.01)
        self.b = theano.shared(np.zeros(self.output_shape,dtype=theano.config.floatX))
        self.params = (self.W, self.b)
        self.output = self.activation(T.dot(self.input, self.W) + self.b.dimshuffle('x', 0))
