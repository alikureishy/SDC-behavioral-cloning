'''
Created on Dec 11, 2016

@author: safdar
'''
from basetrainer import BaseTrainer
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.contrib.learn.python.learn.learn_io import pandas_io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam

class VGG16Trainer(object):
    def __init__(self, model_name, overwrite=False):
        BaseTrainer.__init__(self, model_name, overwrite=overwrite)

    def get_image_shape(self):
        return (224, 224)

    def __create_model__(self):
        # Then step through the training stages
        vggmodel = VGG16(include_top=False, weights='imagenet')
        
        x = vggmodel.output
        x = Flatten(input_shape=x.shape[1:])(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        custom = Model(vggmodel.input, x)        


        #...
        #...

    def __compile_model__(self):
        self.__model__.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
