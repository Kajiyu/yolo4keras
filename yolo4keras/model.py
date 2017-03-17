# -*- coding: utf-8 -*-

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D

class YoloModel():
    def __init__(self, config):
        self.build_network()

    def build_network(self):
        self.input_tensor = Input(shape=(448, 448, 3))
        self.x = Convolution2D(64, 7, 7, activation='relu', border_mode='same', name='conv1', \
                               input_shape=(448, 448, 3), subsample=(2, 2))(self.input_tensor)
        self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(self.x)
        self.x = Convolution2D(192, 3, 3, activation='relu', border_mode='same', name='conv3')(self.x)
        self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(self.x)
        self.x = Convolution2D(128, 1, 1, activation='relu', border_mode='same', name='conv5')(self.x)
        self.x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv6')(self.x)
        self.x = Convolution2D(256, 1, 1, activation='relu', border_mode='same', name='conv7')(self.x)
        self.x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv8')(self.x)
        self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool9')(self.x)
        self.x = Convolution2D(256, 1, 1, activation='relu', border_mode='same', name='conv10')(self.x)
        self.x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv11')(self.x)
        self.x = Convolution2D(256, 1, 1, activation='relu', border_mode='same', name='conv12')(self.x)
        self.x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv13')(self.x)
        self.x = Convolution2D(256, 1, 1, activation='relu', border_mode='same', name='conv14')(self.x)
        self.x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv15')(self.x)
        self.x = Convolution2D(256, 1, 1, activation='relu', border_mode='same', name='conv16')(self.x)
        self.x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv17')(self.x)
        self.x = Convolution2D(512, 1, 1, activation='relu', border_mode='same', name='conv18')(self.x)
        self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool19')(self.x)
        self.x = Convolution2D(512, 1, 1, activation='relu', border_mode='same', name='conv20')(self.x)
        self.x = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv21')(self.x)
        self.x = Convolution2D(512, 1, 1, activation='relu', border_mode='same', name='conv22')(self.x)
        self.x = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv23')(self.x)
        self.x = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv24')(self.x)
        self.x = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', subsample=(2, 2), name='conv25')(self.x)
        self.x = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv26')(self.x)
        self.x = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv27')(self.x)
        self.x = Flatten(name='flatten')(self.x)
        self.x = Dense(512, name='fc1')(self.x)
        self.x = Dense(4096, name='fc2')(self.x)
        self.x = Dense(1470, activation='linear', name='fc3')(self.x)
        # Create model.
        self.model = Model(input_tensor, x, name='yolo')
        print self.model.summary()
