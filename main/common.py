import os
import csv
from sklearn.utils import shuffle
from matplotlib import image as mpimg
import cv2
import numpy as np
import itertools
from os.path import isfile
import random
from custom1trainer import Custom1Trainer
from custom2trainer import Custom2Trainer

(Center, Left, Right, Steer, Break, Throttle, Speed) = (0, 1, 2, 3, 4, 5, 6)

def get_driving_logs(*folders):
    return [os.path.join(folder, 'driving_log.csv') for folder in list(folders)]

def read_csv(*drivinglogfiles):
    rows = []
    for drivinglogfile in list(drivinglogfiles):
        if not isfile(drivinglogfile):
            raise "File not found: {}".format(drivinglogfile)
        with open(drivinglogfile, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            newrows = [[row[Center].strip(), \
                         row[Left].strip(), \
                         row[Right].strip(), \
                         float(row[Steer])] for row in csvreader]
        print ("Read {} samples from file {}".format(len(newrows), drivinglogfile))
        rows = rows + newrows
    return rows

def extract(rows, imagecol, steeringcol):
    return ([row[imagecol] for row in rows], [row[steeringcol] for row in rows])

def resize_image(image, targetshape):
    assert len(targetshape) == 2, "Shape was supposed to be (x, y) but was {}".format(targetshape)
    image = cv2.resize(image, (targetshape[1], targetshape[0]), interpolation = cv2.INTER_AREA)
    return image

# changes left to right and vice-versa (yes, it's called "Vertical" Flip)
def verticalflip(image):
    return cv2.flip(image, 1)

def normalize_image(image):
    return cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# This is the method that generates the actual (X, Y) tuple
def datagen(imagefiles, steerings, targetshape, flipprob=0.0):
    while 1:
        for (imagefile, steering) in zip(imagefiles, steerings):
            # First look for center. If not found, look left, and if not found, look right.
            image = mpimg.imread(imagefile)
            image = resize_image(image, targetshape)
            image = normalize_image(image)
            flip = random.random()
            if flip<flipprob:
                image = verticalflip(image)
                steering *= -1 # Flip the steering as well
#             print (image[0][0][0], steering)
            yield (image, steering)

# This routine iterates over  [(x,y)]
# and generates batches of ([x], [y])
def batchgen(iterable, batchsize):
    it = iter(iterable)
    counter = 0
    while True:
        chunk = itertools.islice(it, batchsize)
        if not chunk:
            return
        xys = list(zip(*chunk)) # This is a list of tuples: [(images), (steerings)]
        counter += len(xys[0])
        toyield = (np.array(list(xys[0])), np.array(list(xys[1])))
        yield toyield

def get_trainer(arch, model_name, overwrite=False):
    trainer = None
    if arch == 'custom1':
        trainer = Custom1Trainer(model_name, overwrite)
    elif arch=='custom2':
        trainer = Custom2Trainer(model_name, overwrite)
    else:
        raise "Not implemented"
    return trainer

