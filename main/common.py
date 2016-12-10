import os
from collections import namedtuple
import csv
from sklearn.utils import shuffle
from matplotlib import image as mpimg
import cv2
import numpy as np
import itertools
from os.path import isfile

Driving_Log = 'driving_log.csv'
Json_Ext = '.json'
Hd5_Ext = '.hd5'

class Params:
    def __init__(self, \
                 model_name, \
                 training_data_folder='./train', \
                 test_data_folder='./test', \
                 validation_data_folder='./valid', \
                 image_shape=(32, 32, 3), \
                 num_epochs=1, \
                 learning_rate=1e-3, \
                 train_batch_size=128, \
                 test_batch_size=128, \
                 validation_batch_size=128, \
                 training_accuracy_threshold=99.9):
        
        # Model location:
        assert model_name is not None
        self.model_name = model_name
        self.model_json_file = model_name + Json_Ext
        self.model_hd5_file = model_name + Hd5_Ext

        # Data location
        self.training_data_folder=training_data_folder
        self.training_log_file=os.path.join(training_data_folder, Driving_Log)
        self.test_data_folder=test_data_folder
        self.test_log_file=os.path.join(test_data_folder, Driving_Log)
        self.validation_data_folder=validation_data_folder
        self.validation_log_file=os.path.join(validation_data_folder, Driving_Log)
        
        # Train settings
        self.image_shape=image_shape
        self.num_epochs=num_epochs
        self.learning_rate=learning_rate
        self.train_batch_size=train_batch_size
        self.test_batch_size=test_batch_size
        self.validation_batch_size=validation_batch_size
        self.training_accuracy_threshold=training_accuracy_threshold
        
        # Image manipulation settings
        
class XYGenerator():
    
    ColumnNames = namedtuple('ColumnNames', ['CenterImage', 'LeftImage', 'RightImage', 'SteeringAngle', 'Throttle', 'Break', 'Speed'])
    Columns = ColumnNames(CenterImage=0, LeftImage=1, RightImage=2, SteeringAngle=3, Throttle=4, Break=5, Speed=6)
    
#     def __init__(self, params):
#         iter_csv = pd.read_csv(params.training_log_file, iterator=True, chunksize=1000)
#         df = pd.concat([chunk[chunk['field'] > constant] for chunk in iter_csv])        
    def __init__(self, drivinglogfile):
        rows = []
        if not isfile(drivinglogfile):
            raise "File not found: {}".format(drivinglogfile)
        
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
        print ("Read {} samples from file {}".format(len(self.rows), drivinglogfile))

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

