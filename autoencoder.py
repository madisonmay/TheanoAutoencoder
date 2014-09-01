from itertools import chain
from itertools import izip
from collections import OrderedDict
from time import time

import numpy as np
import theano
import theano.tensor as T
from  theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from load import load_mnist

from vis import unit_scale, grayscale_grid_vis
from trainers import Momentum
from augmenters import SaltAndPepper
from layers import InputLayer, HiddenLayer
from schedulers import ExponentialDecay

from matplotlib import pyplot as plt 

def mse(targets, preds):
    return np.mean(np.square(targets - preds))

def theano_mse(targets, preds):
    return T.mean(T.sqr(targets - preds))

class AutoEncoder(object):
    """
    A generic theano autoencoder
    """

    def __init__(self, n_vis, layers, trainer, loss, batch_size, n_batches, n_epochs, lr_decay):
        self.batch_size = batch_size
        self.trainer = trainer
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.lr_decay = lr_decay
        self.epoch = 0
        self.examples = 0

        self.batch_X = theano.shared(np.zeros((n_batches * batch_size, n_vis), dtype=theano.config.floatX), borrow=True)
        self.batch_y = theano.shared(np.zeros((n_batches * batch_size, n_vis), dtype=theano.config.floatX), borrow=True)

        self.layers = layers
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        self.setup_layers()

        self.idx = T.lscalar('idx')

        # check that we can move these in
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
        self.updates = self.trainer.get_updates(self.params,self.grads)

        self.train = theano.function([self.idx], self.loss, givens=self.givens, updates=self.updates)
        self.fprop = theano.function([self.idx], self.output_layer.output, givens=self.givens)
        self.eval_loss = theano.function([self.idx], self.loss, givens=self.givens)

    def setup_layers(self):
        prev_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.setup(prev_layer)
            prev_layer = layer

    def iter_data(self, X):
        # Should take an input and target pair
        # Fix data clipping for predicting
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
        loss = np.mean([self.eval_loss(batch) for batch in self.iter_data(X)])
        print "%.3f epoch" % self.epoch
        print "%.3f loss" % loss
        print "%.3f learning rate"%self.trainer.lr.value.get_value()
        print "%.3f examples per second"%(self.examples/(time()-self.t))

    def fit(self, trX, teX):
        self.t = time()
        for epoch in range(self.n_epochs):
            self.monitor(teX)
            for batch in self.iter_data(trX):
                self.train(batch)
                self.examples += self.batch_size
            self.epoch += 1
            self.trainer.scheduler_updates()

    def predict(self, X):
        """
        Currently clips the last few rows of X
        and requires a minimum of batch_size * n_batches examples
        """
        predictions = []
        for batch in self.iter_data(X):
            predictions.append(self.fprop(batch))

        return np.vstack(predictions)

if __name__ == "__main__":
    data_dir = '/media/datasets/mnist'
    trX, _, teX, _ = load_mnist(data_dir)

    augmenter = SaltAndPepper(low=0.,high=1.,p_corrupt=0.5)

    bce = T.nnet.binary_crossentropy
    # Factor out trainer
    # Generalize to multiple layers
    n_vis=784
    n_hidden=2000
    batch_size = 128
    activation = T.nnet.sigmoid
    layers = [
        InputLayer(n_vis,batch_size=batch_size,augmenter=augmenter),
        HiddenLayer(n_hidden, activation),
        HiddenLayer(n_vis, activation)
    ]

    
    lr_scheduler = ExponentialDecay(value=0.1, decay=0.99)
    trainer = Momentum(lr=lr_scheduler, m=0.9)

    model = AutoEncoder(n_vis=n_vis, layers=layers, trainer=trainer, loss=bce, batch_size=batch_size, n_batches=32, n_epochs=100, lr_decay=0.99)
    model.fit(trX, teX)

    w1 = model.layers[1].W.get_value().T
    w2 = model.layers[2].W.get_value()
    pred = model.predict(teX)

    grayscale_grid_vis(pred[:100],transform=lambda x:unit_scale(x.reshape(28,28)),show=True)

    img1 = grayscale_grid_vis(w1,transform=lambda x:unit_scale(x.reshape(28,28)),show=False)
    img2 = grayscale_grid_vis(w2,transform=lambda x:unit_scale(x.reshape(28,28)),show=False)
    plt.subplot(1,2,1)
    plt.imshow(img1,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(img2,cmap='gray')
    plt.show()