'''
Created on Dec 7, 2016

@author: safdar
'''
import argparse
from keras.models import save_model
from random import randint
import os
import csv
import pandas as pd
from common import Params
from collections import namedtuple
from matplotlib import image as mpimg
import cv2
import itertools
from os.path import isfile
from model import read_model, create_model, write_model
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class XYGenerator():
    
    ColumnNames = namedtuple('ColumnNames', ['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle', 'Throttle', 'Break', 'Speed'])
    Columns = ColumnNames(CenterImage=0, LeftImage=1, RightImage=2, SteeringAngle=3, Throttle=4, Break=5, Speed=6)
    
#     def __init__(self, params):
#         iter_csv = pd.read_csv(params.training_log_file, iterator=True, chunksize=1000)
#         df = pd.concat([chunk[chunk['field'] > constant] for chunk in iter_csv])        
    def __init__(self, drivinglogfile):
        rows = []
        with open(drivinglogfile, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                centerimage = row[XYGenerator.Columns.CenterImage].strip()
                leftimage = row[XYGenerator.Columns.LeftImage].strip()
                rightimage = row[XYGenerator.Columns.RightImage].strip()
                steeringangle = float(row[XYGenerator.Columns.SteeringAngle])
                brk = float(row[XYGenerator.Columns.Break])
                throttle = float(row[XYGenerator.Columns.Throttle])
                speed = float(row[XYGenerator.Columns.Speed])
                toadd = [centerimage, leftimage, rightimage, steeringangle, brk, throttle, speed]
                rows.append(toadd)
        self.rows = rows
        print ("Read {} rows of training data".format(len(self.rows)))

    def len(self):
        return self.count()
    
    def count(self):
        return len(self.rows)
    
    def __filter__(self):        
        # Filter out rows with zero steering
        tmp = self.rows
        self.rows = [row for row in self.rows if not row[XYGenerator.Columns.SteeringAngle] == 0 ]
        del tmp
        return self
        
    def __smoothen__(self):
        # Smoothen steering angles
        raise "Not implemented"
        return self
        
    def splitter(self):
        # Split a row into its component rows
        # and adjust the motion parameters
        # accordingly. Generate more than 1
        # row from given row.
        correction = 0.12
        splitrows = []
        for row in self.rows:
            centerimage = row[XYGenerator.Columns.CenterImage]
            leftimage = row[XYGenerator.Columns.LeftImage]
            rightimage = row[XYGenerator.Columns.RightImage]
            steeringangle = float(row[XYGenerator.Columns.SteeringAngle])
            brk = float(row[XYGenerator.Columns.Break])
            throttle = float(row[XYGenerator.Columns.Throttle])
            speed = float(row[XYGenerator.Columns.Speed])
            
            # Generate 3 rows from this:
            row_center = [centerimage, None, None, steeringangle, brk, throttle, speed]
            row_left = [leftimage, None, None, steeringangle+correction, brk, throttle, speed]
            row_right = [rightimage, None, None, steeringangle-correction, brk, throttle, speed]
            
            splitrows.append(row_center)
            splitrows.append(row_left)
            splitrows.append(row_right)
            
        tmp = self.rows
        self.rows = splitrows
        del tmp # release the memory
        return self

    def shuffle(self):        
        # Shuffle
        #self.rows = self.rows.sample(frac=1).reset_index(drop=True) # Shuffle
        self.rows = shuffle(self.rows)
        return self
        
    # This is the method that generates the actual (X, Y) tuple
    def generator(self):
        while 1:
            for row in self.rows:
                # First look for center. If not found, look left, and if not found, look right.
                if not row[XYGenerator.Columns.CenterImage]==None:
                    image = mpimg.imread(row[XYGenerator.Columns.CenterImage])
                else:
                    raise "Found record without any center image."
                
                steeringangle = row[XYGenerator.Columns.SteeringAngle]
                yield (image, steeringangle)

# A class of decorator functions
# The final method to be invoked is batcher().
# For example:
#        get_generator ( batcher ( 
class XYBatcher():
    def __init__(self, desiredshape, batchsize, xygenerator):
        self.desiredshape = desiredshape
        self.batchsize = batchsize
        self.xygenerator = xygenerator
        
    def len(self):
        return self.count()
    
    def count(self):
        return self.xygenerator.count()
    
    # This receives a generator 'iter' that generates
    # (X, Y) tuples. This routine performs a computation
    # on the (X, Y) tuple, and returns an (X', Y') tuple,
    # or possibly multiple (X', Y') tuples
    def __shifter__(self, iterable):
        for item in iterable:
            yield item

    def __resizer__(self, iterable):
        for (image, steering) in iterable:
            image = cv2.resize(image, (self.desiredshape[0], self.desiredshape[1]))
            yield (image, steering)

    def __cropper__(self, iterable):
        for item in iterable:
            yield item

    def __rotator__(self, iterable):
        for item in iterable:
            yield item

    # This routine iterates over the output
    # of the other (X, Y) generator routines
    # in this class, and generates aggregates
    # of those tuples, in the form (X[], Y[])
    # which is then served as input to the trainer
    def __batcher__(self, iterable):
        it = iter(iterable)
        counter = 0
        while True:
            chunk = itertools.islice(it, self.batchsize)
            if not chunk:
                return
            # We split the chunk list into 2 lists of chunks
            xys = list(zip(*chunk)) # This is a list of tuples: [(images), (steerings)]
            counter += len(xys[0])
            toyield = (np.array(list(xys[0])), np.array(list(xys[1])))
            yield toyield


    def get_generator(self):
        return iter(self.__batcher__(\
                                self.__rotator__(\
                                                 self.__cropper__(\
                                                                  self.__shifter__(\
                                                                                   self.__resizer__(\
                                                                                                    self.xygenerator.generator()))))))
    # ...
    # ...
    # ...



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('-mn', '--model_name', dest='model_name', required=True, type=str, help='Name of model. (Without ''.json'' and ''.hd5'')')
    parser.add_argument('-tdf', '--training_data_folder', dest='training_data_folder', default='.', type=str, help='Path to folder with training data')
    parser.add_argument('-ne', '--num_epochs', dest='num_epochs', default=10, type=int, help='Number of epochs to train on')
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('-trbs', '--train_batch_size', dest='train_batch_size', default=100, type=str, help='Batch size for training')
    parser.add_argument('-vbs', '--validation_batch_size', dest='validation_batch_size', default=100, type=int, help='Validation batch size')
    #parser.add_argument('-tebs', '--test_batch_size', dest='test_batch_size', default=100, type=str, help='Batch size for testing')
    parser.add_argument('-tat', '--training_accuracy_threshold', dest='training_accuracy_threshold', default=0.99, type=float, help='Accuracy threshold after which to stop training.')
    parser.add_argument('-o', '--override', dest='override', action='store_true', help='Override model if it already exists (default: false)')
    parser.add_argument('-t', '--trial', dest='trial', action='store_true', help='Trial run. Will not attempt to save anything.')

    args = parser.parse_args()
    params = Params(args.model_name,
                    training_data_folder=args.training_data_folder,
                    image_shape=(32, 32, 3),
                    num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate,
                    train_batch_size=args.train_batch_size,
                    validation_batch_size=args.validation_batch_size,
#                    test_batch_size=args.test_batch_size,
                    training_accuracy_threshold=args.training_accuracy_threshold
                    )

    # Obtain the model
    model = None
    if isfile(params.model_json_file):
        model = read_model(params.model_name)
    else:
        print ("No previous checkpoints found for model: ", params.model_name)
        print ("Creating model from scratch.")
        model = create_model(params)
    
    # Create train/validation data generators
    traingenerator = XYGenerator(params.training_log_file).splitter().shuffle()
    trainbatcher = XYBatcher(params.image_shape, params.train_batch_size, traingenerator)
    validationgenerator = None
    validationbatcher = None
    
    #validationgenerator = XYGenerator(params.training_log_file).splitter().shuffle()
    #validationbatcher = XYBatcher(params.image_shape, params.validation_batch_size, validationgenerator)
        # validation_data = validationbatcher.get_generator()
        # nb_val_samples = validationbatcher.count()

    # Train the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit_generator(trainbatcher.get_generator(), samples_per_epoch=trainbatcher.count(), nb_epoch = params.num_epochs, nb_val_samples = 0, verbose=1, validation_data=None)
    
    # Finally, determine what to do with the model
    if (args.override):
        write_model(model, params.model_name)
    elif not args.trial:
        i = 1
        filename = params.model_json_file
        while isfile(filename):
            filename = params.model_name+'_'+str(i)+'.json'
            i += 1
        write_model(model, filename)
    else:
        print ("Dry run. Nothing will be saved.")
