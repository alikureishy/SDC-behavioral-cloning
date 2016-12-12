import os
import csv
from sklearn.utils import shuffle
from matplotlib import image as mpimg
import cv2
import numpy as np
import itertools
from os.path import isfile
from vgg16trainer import VGG16Trainer
from commaaitrainer import CommaAITrainer

(Center, Left, Right, Steer, Break, Throttle, Speed) = (0, 1, 2, 3, 4, 5, 6)

def get_driving_log(folder):
    return os.path.join(folder, 'driving_log.csv')

def read_csv(drivinglogfile):
    if not isfile(drivinglogfile):
        raise "File not found: {}".format(drivinglogfile)
    with open(drivinglogfile, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = [[row[Center].strip(), \
                 row[Left].strip(), \
                 row[Right].strip(), \
                 float(row[Steer]), \
                 float(row[Break]), \
                 float(row[Throttle]), \
                 float(row[Speed])] for row in csvreader]
    print ("Read {} samples from file {}".format(len(rows), drivinglogfile))
    return rows

def extract(rows, imagecol, steeringcol):
    return ([row[imagecol] for row in rows], [row[steeringcol] for row in rows])

def process_image(image, targetshape):
    assert len(targetshape) == 2, "Shape was supposed to be (x, y) but was {}".format(targetshape)
    image = cv2.resize(image, (targetshape[1], targetshape[0]), interpolation = cv2.INTER_AREA)
    return image

# This is the method that generates the actual (X, Y) tuple
def datagen(imagefiles, steerings, targetshape):
    while 1:
        for (imagefile, steering) in zip(imagefiles, steerings):
            # First look for center. If not found, look left, and if not found, look right.
            image = mpimg.imread(imagefile)
            image = process_image(image, targetshape)
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
    if arch == 'vgg16':
        trainer = VGG16Trainer(model_name, overwrite)
    elif arch=='commaai':
        trainer = CommaAITrainer(model_name, overwrite)
    else:
        raise "Not implemented"
    return trainer

