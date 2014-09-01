from  theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
from vis import *

srng = RandomStreams()

class SaltAndPepper(object):

	def __init__(self,low,high,p_corrupt):
		self.low = low
		self.high = high
		self.p_corrupt = p_corrupt

	def augment(self,X,shape):
		corrupt_mask = srng.binomial(shape, p=self.p_corrupt, dtype='int32').astype(theano.config.floatX)
		salt_and_pepper_mask = srng.binomial(shape, p=self.p_corrupt, dtype='int32').astype(theano.config.floatX)
		X = (X*(corrupt_mask < 1))+(salt_and_pepper_mask*corrupt_mask)
		return X
