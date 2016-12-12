'''
Created on Dec 7, 2016

@author: safdar
'''
import argparse
from common import get_driving_log, get_trainer, read_csv, extract, datagen, batchgen, Center, Left, Right, Steer
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.utils import shuffle

if __name__ == '__main__':
    print ("###############################################")
    print ("#                   TRAINER                   #")
    print ("###############################################")

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('-n', '--model_name', dest='model_name', required=True, type=str, help='Name of model folder')
    parser.add_argument('-a', '--arch', dest='arch', required=True, type=str, help='Architecture of model. [vgg16, googlenet, commaai, none]')
    parser.add_argument('-t', '--training_data_folder', dest='training_data_folder', default='.', type=str)
    parser.add_argument('-v', '--validation_data_folder', dest='validation_data_folder', default='.')
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=10, type=int)
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=100, type=int)
    parser.add_argument('-o', '--overwrite', dest='overwrite', action='store_true', help='Overwrite model after training (default: false)')
    parser.add_argument('-r', '--dry_run', dest='dry_run', action='store_true', help='Dry run. Will not attempt to save anything.')
    args = parser.parse_args()

    # Obtain the trainer
    trainer = get_trainer(args.arch, args.model_name, args.overwrite)
    
    # Read training data
    driving_log = get_driving_log(args.training_data_folder)
    rows = read_csv(driving_log)
    (centerfiles, centersteerings) = extract(rows, Center, Steer)
    
    # Some training data pre-processing:
    (leftfiles, leftsteerings) = extract(rows, Left, Steer)
    (leftfiles, leftsteerings) = (leftfiles, [steering+0.5 for steering in leftsteerings])
    (rightfiles, rightsteerings) = extract(rows, Right, Steer)
    (rightfiles, rightsteerings) = (rightfiles, [steering-0.5 for steering in rightsteerings])
    (imagefiles, steerings) = (np.concatenate((centerfiles, leftfiles, rightfiles)), np.concatenate((centersteerings, leftsteerings, rightsteerings)))
    (imagefiles, steerings) = shuffle(imagefiles, steerings)
    x_train, x_val, y_train, y_val = train_test_split(imagefiles, steerings)
#     x_train, x_val, y_train, y_val = x_train[0:200], x_val[0:200], y_train[0:200], y_val[0:200]
    

    # Memory-efficient generators
    traindata = datagen(x_train, y_train, trainer.get_image_shape())
    validationdata = datagen(x_val, y_val, trainer.get_image_shape())
    
    trainbatcher = batchgen(traindata, args.batch_size)
    validationbatcher = batchgen(validationdata, args.batch_size)

    # Train the basetrainer
    trainer.train_with(trainbatcher, len(x_train), validation_data=validationbatcher, nb_val_samples=len(x_val), num_epochs=args.num_epochs)
    
    # Finally, determine what to do with the basetrainer
    if args.dry_run:
        print ("Dry run. Nothing will be saved.")
    else:
        trainer.checkpoint()

    print ("Thank you! Come again!")