from itertools import chain
from itertools import izip
from collections import OrderedDict
from time import time

import numpy as np
import theano
import theano.tensor as T
from  theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def sgd(params,grads,lr):
	return [(param, param - lr * grad) for param, grad in izip(params, grads)]

def momentum(params,grads,lr,m):
	updates = []
	for param,grad in zip(params,grads):
		mp = theano.shared(param.get_value()*0.)
		v = m*mp - lr*grad
		w = param + v
		updates.append((mp,v))
		updates.append((param,w))
	return updates