from types import MethodType

import theano
import numpy as np

from utils import floatX


class Scheduler(object):

    def __new__(self, value, **kwargs):
        variable = theano.shared(floatX(value))
        for k, v in kwargs.items():
            setattr(variable, k, v)
        variable.update = MethodType(self.update, variable)
        return variable

    @staticmethod
    def update(self):
        raise NotImplementedError()


class ExponentialDecay(Scheduler):

    @staticmethod
    def update(self):
        current = self.get_value()
        updated = current * self.decay
        self.set_value(floatX(updated))
