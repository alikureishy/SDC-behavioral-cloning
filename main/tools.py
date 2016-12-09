import csv
import cv2
import numpy as np
import matplotlib.image as mpimg

# Read input from CSV and Img folder:
def generate_batches(data):
    batches = []
    num_batches = int(np.ceil(data.count / data.batch_size))
    for batch in range(num_batches):
        i = batch * data.batch_size
        j = i + min(data.batch_size, data.count - i)
        batches.append((i, j))
    return batches

# Read in from CSV file
LOG_FILE = '/Users/safdar/Documents/self-driving-car/behavioral-cloning/driving_log.csv'

Fields = [
    'CenterCam',
    'LeftCam',
    'RightCam',
    'SteeringAngle',
    'Throttle',
    'Break',
    'Speed']

LEFT_CAM_COUMN = 1
CENTER_CAM_COLUMN = 0
RIGHT_CAM_COLUMN = 2
STEERING_COLUMN = 3

img_rows = 160
img_cols = 320

def batcher(file, cameracol, steeringcol, batch_size):
    with open(file) as csvfile:
        images = None
        steerings = None
        batchcount = 0
        idx = 0
        reader = csv.DictReader(csvfile, Fields)
        for row in reader:
            if idx % batch_size == 0:
                if not images is None:
                    batchcount += 1
                    yield images, steerings
                images = []
                steerings = []
            
            imgfile = row[cameracol]
            steering = row[steeringcol]
            img = mpimg.imread(imgfile)
            
            # Pre-process image
            img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
            img = cv2.resize(img, (img_rows, img_cols))
            img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            images.append(img)
            steerings.append(steering)
        
            idx += 1

def load_training_data(file, cameracol, steeringcol):
    with open(file) as csvfile:
        images = []
        steerings = []
        reader = csv.DictReader(csvfile, Fields)
        for row in reader:
            imgfile = row[cameracol]
            steering = row[steeringcol]
            img = mpimg.imread(imgfile)
            
            # Pre-process image
            #img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
            img = cv2.resize(img, (img_rows, img_cols))
            img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
            images.append(img)
            steerings.append(steering)
        
        return (np.array(images), np.array(steerings))

def load_center_training_data():
    return load_training_data(LOG_FILE, 'CenterCam', 'SteeringAngle')

def load_left_training_data():
    return load_training_data(LOG_FILE, 'LeftCam', 'SteeringAngle')

def load_right_training_data():
    return load_training_data(LOG_FILE, 'RightCam', 'SteeringAngle')
        
def centerbatcher(batch_size):
    return batcher(LOG_FILE, 'CenterCam', 'SteeringAngle', batch_size)

def leftbatcher(batch_size):
    return batcher(LOG_FILE, 'LeftCam', 'SteeringAngle', batch_size)

def rightbatcher(batch_size):
    return batcher(LOG_FILE, 'RightCam', 'SteeringAngle', batch_size)

def printall():
    print ("Checking driver log batches...")
    batch_size = 128
    batch = 0
    for (images, steerings) in batcher(LOG_FILE, 'CenterCam', 'SteeringAngle', batch_size):
        print ("Batch #: ", batch, len(images), len(steerings))
        for i in range (len (images)):
            print ("\t", i, images[i].shape, '-->', steerings[i])
        batch += 1
