import theano
import numpy as np

def floatX(val):
	return np.asarray(val, dtype=theano.config.floatX)

def sharedX(val):
	return theano.shared(floatX(val))