# This is the module that builds the __model__. All __model__ architecture details are encapsulated here
from os.path import isfile, isdir
import os
from keras.models import model_from_json
from os import makedirs

class BaseTrainer(object):
    def __init__(self, model_name, overwrite=False):
        self.__model_name__ = model_name
        self.__overwrite__ = overwrite
        # Obtain the model
        model = self.__read_model__()
        if not model:
            print ("No previous checkpoints found for basetrainer: ", model_name)
            print ("Creating basetrainer from scratch.")
            model = self.__create_model__()
        self.__model__ = model
        self.__compile_model__()
        return self

    @classmethod
    def get_image_shape(self):
        # TODO: I don't think this will work
        return self.__model__.input.shape()

    def checkpoint(self):
        (json, hd5) = self.__write_model__()
        if json == None:
            print ("Model {} could not be saved.".format(self.__model_name__))
        else:
            print ("Model saved to files: {}, {}".format(json, hd5))

    def __compile_model__(self):
        raise "Not Implemented"

    def train_with(self, trainbatchgen, samples_per_epoch, validation_data=None, nb_val_samples=0, num_epochs=1):
        self.__model__.fit_generator(trainbatchgen, samples_per_epoch=samples_per_epoch, nb_epoch = num_epochs, \
                                     nb_val_samples = nb_val_samples, verbose=1, validation_data=validation_data)

    def evaluate_with(self, testbatchgen, nb_val_samples):
        [loss, accuracy] = self.__model__.evaluate_generator(testbatchgen, nb_val_samples)
        return (loss, accuracy)

    def predict(self, imagearray, batchsize=32):
        return self.__model__.predict(imagearray, batchsize, verbose=1)

    def __create_model__(self):
        raise "Virtual create_model not implemented"

    def __getmodelpath__(self, iteration=None):
        if iteration==None or iteration==0:
            return os.path.join(self.__model_name__, 'model.json')
        else:
            return os.path.join(self.__model_name__, 'model_{}.json'.format(iteration))

    def __getweightspath__(self, iteration=None):
        if iteration==None or iteration==0:
            return os.path.join(self.__model_name__, 'weights.hd5')
        else:
            return os.path.join(self.__model_name__, 'weights_{}.hd5'.format(iteration))

    def __write_model__(self):
        # If no folder exists, no point looking
        if not isdir(self.__model_name__):
            makedirs(self.__model_name__)
        
        idx = None
        if self.__overwrite__:
            idx = self.__id__
        else:
            for idx in range(0, 1000):
                if not isfile(self.__getmodelpath__(idx)):
                    break

        jsonfile = self.__getmodelpath__(idx)
        print ("Writing model to file: ", jsonfile)
        json = self.__model__.to_json()
        with open(jsonfile, 'w') as jfile:
            jfile.write(json)

        hd5file = self.__getweightspath__(idx)
        print ("Writing mweights to file: ", hd5file)
        self.__model__.save_weights(hd5file)
    
        return jsonfile, hd5file

    def __read_model__(self):
        # If no folder exists, no point looking
        if not isdir(self.__model_name__):
            return None

        # find the last model file in folder
        idx = None
        for idx in range(0, 1000):
            if not isfile(self.__getmodelpath__(idx)):
                    break
        if idx==0:
            return None # No model exists if it couldn't even find id = 0
        else:
            idx = idx - 1

        jsonfile = self.__getmodelpath__(idx)
        print ("Reading model from file: ", jsonfile)
        with open(jsonfile, 'r') as jfile:
            model = model_from_json(jfile.read())

        hd5file = self.__getweightspath__(idx)
        print ("Reading weights from file: ", hd5file)
        model.load_weights(hd5file)
        
        self.__id__ = idx
        
        return model
