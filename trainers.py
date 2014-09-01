from itertools import izip

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
from  theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from schedulers import Constant

class Trainer(object):

    def __init__(self, **kwargs):
        self.params = []
        for k, v in kwargs.items():
            if not isinstance(v, CudaNdarraySharedVariable):
                v = Constant(v)
            setattr(self, k, v)
            self.params.append(v)

    def scheduler_updates(self):
        """
        Called once every epoch
        """
        for param in self.params:
            param.update()


class SGD(Trainer):

    def get_updates(self, params, grads):
        return [(param, param - self.lr * grad) for param, grad in izip(params, grads)]


class Momentum(Trainer):

    def get_updates(self, params, grads):
    	updates = []
    	for param, grad in zip(params,grads):
    		mp = theano.shared(param.get_value()*0.)
    		v = self.m * mp - self.lr * grad
    		w = param + v
    		updates.append((mp,v))
    		updates.append((param,w))

        return updates