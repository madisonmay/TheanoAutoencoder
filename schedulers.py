import theano
import numpy as np
from utils import floatX, sharedX

class ExponentialDecay(object):

	def __init__(self, value, decay):
		self.decay = decay
		self.value = sharedX(value)

	def update(self):
		current = self.value.get_value()
		updated = current * self.decay
		self.value.set_value(floatX(updated))

# constant scheduler
# convert to floatX, sharedX all current instances
# cleaner syntax for lr.value
