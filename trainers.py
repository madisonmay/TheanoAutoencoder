from itertools import chain
from itertools import izip
from collections import OrderedDict
from time import time

import numpy as np
import theano
import theano.tensor as T
from  theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Initalize constanst to constant schedulers
class SGD(object):

    def __init__(self, lr):
        self.lr = lr
        self.params = [self.lr]

    def get_updates(self, params, grads):
	   return [(param, param - self.lr.value * grad) for param, grad in izip(params, grads)]

    def scheduler_updates(self):
        """
        Called once every epoch
        """
        for param in self.params:
            param.update()

class Momentum(SGD):

    def __init__(self, lr, m):
        self.lr = lr
        self.m = m
        self.params = [self.lr]

    def get_updates(self, params, grads):
    	updates = []
    	for param, grad in zip(params,grads):
    		mp = theano.shared(param.get_value()*0.)
    		v = self.m * mp - self.lr.value * grad
    		w = param + v
    		updates.append((mp,v))
    		updates.append((param,w))

        return updates