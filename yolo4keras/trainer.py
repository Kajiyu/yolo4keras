# -*- coding: utf-8 -*-

import numpy as np
from keras.optimizers import SGD
import os
from model import YoloModel


class Trainer():
    def __init__(self, config):
        _model = YoloModel(config)
        self.model = _model.model

    def train(self):
        model.fit()
