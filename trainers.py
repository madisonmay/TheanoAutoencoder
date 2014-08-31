from itertools import chain
from itertools import izip
from collections import OrderedDict
from time import time

import numpy as np
import theano
import theano.tensor as T
from  theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class SGD(object):

    def __init__(self, lr):
        self.lr = theano.shared(np.asarray(lr,dtype=theano.config.floatX))

    def get_updates(self, params, grads):
	   return [(param, param - self.lr * grad) for param, grad in izip(params, grads)]

class Momentum(object):

    def __init__(self, lr, m, lr_decay):
        self.lr = theano.shared(np.asarray(lr,dtype=theano.config.floatX))
        self.m = m
        self.lr_decay = lr_decay

    def get_updates(self, params, grads):
    	updates = []
    	for param,grad in zip(params,grads):
    		mp = theano.shared(param.get_value()*0.)
    		v = self.m*mp - self.lr*grad
    		w = param + v
    		updates.append((mp,v))
    		updates.append((param,w))

        # learning rate decay
        self.lr.set_value((self.lr.get_value()*self.lr_decay).astype(theano.config.floatX))

        return updates