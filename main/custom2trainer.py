'''
Created on Dec 11, 2016

@author: safdar
'''
from basetrainer import BaseTrainer
from keras.layers.core import Flatten, Dropout, Dense, Lambda, Activation
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers.pooling import MaxPooling2D

class Custom2Trainer(BaseTrainer):
    
    def __init__(self, model_name, overwrite):
        BaseTrainer.__init__(self, model_name, overwrite=overwrite)

    def __create_model__(self):
        row, col, ch = 32, 32, 3 # camera format

        model = Sequential()

        model.add(Convolution2D(16, 3, 3, border_mode="same", input_shape=(row, col, ch)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ELU())

        model.add(Convolution2D(32, 3, 3, border_mode="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.50))
        model.add(ELU())

        model.add(Flatten())
        
        model.add(Dense(200))
#         model.add(Dropout(.2))
        model.add(ELU())

        model.add(Dense(100))
        model.add(Dropout(0.5))
        model.add(ELU())

        model.add(Dense(1))
#         model.add(Activation('tanh')) # To limit the output to between -1 and 1
        
        return model

    def __compile_model__(self):
#         opt = SGD(lr=0.1, momentum=0.1, decay=0.1, nesterov=True)
        opt = Adam(lr=0.001)
        self.__model__.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
