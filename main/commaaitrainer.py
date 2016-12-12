'''
Created on Dec 11, 2016

@author: safdar
'''
from basetrainer import BaseTrainer
from keras.layers.core import Flatten, Dropout, Dense, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

class CommaAITrainer(BaseTrainer):
    def __init__(self, model_name, overwrite):
        BaseTrainer.__init__(self, model_name, overwrite=overwrite)

    def get_image_shape(self):
        return (160, 320)

    def __create_model__(self):
        row, col, ch = 160, 320, 3 # camera format

        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1., \
                         input_shape=(row, col, ch), \
                         output_shape=(row, col, ch)))
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))
        
        return model

    def __compile_model__(self):
        self.__model__.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
