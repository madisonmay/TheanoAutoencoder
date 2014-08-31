from itertools import chain
from itertools import izip
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from  theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from load import load_mnist


def mse(targets, preds):
    return np.mean(np.square(targets - preds))

def theano_mse(targets, preds):
    return T.mean(T.sqr(targets - preds))

class AutoEncoder(object):
    """
    A generic theano autoencoder
    """

    def __init__(self, n_vis, n_hidden, activation, loss, lr, batch_size, n_batches, n_epochs):
        self.batch_size = batch_size
        self.lr = lr
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.epoch = 0

        self.batch_X = theano.shared(np.zeros((n_batches * batch_size, n_vis), dtype=theano.config.floatX), borrow=True)
        self.batch_y = theano.shared(np.zeros((n_batches * batch_size, n_vis), dtype=theano.config.floatX), borrow=True)

        self.input_layer = InputLayer(n_vis)
        self.hidden_layer = HiddenLayer(self.input_layer, n_hidden, activation)
        self.output_layer = HiddenLayer(self.hidden_layer, n_vis, activation)
        self.layers = [self.input_layer, self.hidden_layer, self.output_layer]

        self.idx = T.lscalar('idx')

        # self.batch_start = self.idx * batch_size
        # self.batch_end = (self.idx + 1) * batch_size

        self.givens = {
            self.input_layer.input: self.batch_X[self.idx*batch_size:(self.idx+1)*batch_size],
            self.input_layer.input: self.batch_y[self.idx*batch_size:(self.idx+1)*batch_size]
        }

        self._loss = loss
        self.loss = T.mean(self._loss(self.output_layer.output, self.input_layer.input))
        self.params = list(chain.from_iterable((layer.params for layer in self.layers)))
        self.grads = [T.grad(self.loss, param) for param in self.params]
        self.updates = [(param, param - self.lr * grad) for param, grad in izip(self.params, self.grads)]

        self.train = theano.function([self.idx], self.loss, givens=self.givens, updates=self.updates)
        self.fprop = theano.function([self.idx], self.output_layer.output, givens=self.givens)


    def iter_data(self, X):
        chunk_size = self.batch_size * self.n_batches
        n_chunks = X.shape[0] / chunk_size
        for chunk in range(n_chunks):
            start_idx = chunk * chunk_size
            end_idx = (chunk + 1) * chunk_size
            self.batch_X.set_value(X[start_idx:end_idx])
            self.batch_y.set_value(X[start_idx:end_idx])
            for batch in range(self.n_batches):
                yield batch

    def monitor(self, X):
        print "Epoch: %s" % self.epoch
        preds = self.predict(X)
        loss = mse(X[:len(preds)], preds)
        print "Loss: %s" % loss

    def fit(self, trX, teX):
        for epoch in range(self.n_epochs):
            self.monitor(teX)
            for batch in self.iter_data(trX):
                self.train(batch)
            self.epoch += 1

    def predict(self, X):
        """
        Currently clips the last few rows of X
        and requires a minimum of batch_size * n_batches examples
        """
        predictions = []
        for batch in self.iter_data(X):
            predictions.append(self.fprop(batch))

        return np.vstack(predictions)

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

class HiddenLayer(object):
    """
    Hidden layer for theano autoencoder
    """

    def __init__(self, input_layer, n_outputs, activation):
        self.activation = activation
        self.input_layer = input_layer
        self.input = input_layer.output
        self.output_shape = n_outputs
        self.W = theano.shared(np.random.random((self.input_layer.output_shape, n_outputs))*.01)
        self.b = theano.shared(np.zeros(n_outputs))
        self.params = (self.W, self.b)
        self.output = self.activation(T.dot(self.input, self.W) + self.b.dimshuffle('x', 0))


if __name__ == "__main__":
    data_dir = '/home/mmay/data/mnist'
    trX, _, teX, _ = load_mnist(data_dir)
    bce = T.nnet.binary_crossentropy
    model = AutoEncoder(n_vis=784, n_hidden=100, activation=T.nnet.sigmoid, loss=bce, lr=0.01, batch_size=128, n_batches=64, n_epochs=5)
    model.fit(trX, teX)
