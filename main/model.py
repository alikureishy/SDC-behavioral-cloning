# This is the module that builds the model. All model architecture details are encapsulated here
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

def create_model(params):
    nb_filters = 16
    kernel_size = (3, 3)
    pool_size = (2, 2)
    
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=params.image_shape))
    model.add(Activation('relu'))
    #model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten()) # batch_input_shape=(None,32,32,3)
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    #model.add(Activation())
    
    return model

def predict(model, params, Xs, Ys):
    model.predict(Xs, Ys, batch_size=params.test_batch_size, verbose=1)

def write_model(model, filename):
    jsonfile = filename+'.json'
    print ("Writing model to file: ", jsonfile)
    json = model.to_json()
    hd5file = filename+'.hd5'
    with open(jsonfile, 'w') as jfile:
        jfile.write(json)
    print ("Writing mweights to file: ", hd5file)
    model.save_weights(hd5file)
    return jsonfile, hd5file

def read_model(filename):
    jsonfile = filename+'.json'
    print ("Reading model from file: ", jsonfile)
    with open(jsonfile, 'r') as jfile:
        model = model_from_json(jfile.read())
    hd5file = filename+'.hd5'
    print ("Reading weights from file: ", hd5file)
    model.load_weights(hd5file)
    return model