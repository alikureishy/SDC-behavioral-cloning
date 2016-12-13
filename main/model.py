'''
Created on Dec 7, 2016

@author: safdar
'''
import argparse
from common import get_driving_logs, get_trainer, read_csv, extract, datagen, batchgen, Center, Left, Right, Steer
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection._split import train_test_split
from random import randint

if __name__ == '__main__':
    print ("###############################################")
    print ("#                   TRAINER                   #")
    print ("###############################################")

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('-n', '--model_name', dest='model_name', required=True, type=str, help='Name of model folder')
    parser.add_argument('-a', '--arch', dest='arch', required=True, type=str, help='Architecture of model. [vgg16, googlenet, commaai, none]')
    parser.add_argument('-t', '--training_data_folders', required=True, nargs='*', dest='training_data_folders', type=str, help="Space-separated list of 1 or more folders.")
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=10, type=int)
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=100, type=int)
    parser.add_argument('-o', '--overwrite', dest='overwrite', action='store_true', help='Overwrite model after training (default: false)')
    parser.add_argument('-r', '--dry_run', dest='dry_run', action='store_true', help='Dry run. Will not attempt to save anything.')
    args = parser.parse_args()

    # Obtain the trainer
    trainer = get_trainer(args.arch, args.model_name, args.overwrite)
    
    # Read training data
    driving_logs = get_driving_logs(*args.training_data_folders)
    rows = read_csv(*driving_logs)
    (centerfiles, centersteerings) = extract(rows, Center, Steer)
    (imagefiles, steerings) = shuffle(centerfiles, centersteerings, random_state=randint(0,100))
    print ("Steering values: min {} / max {} / average {} ".format(min(steerings), max(steerings), np.mean(steerings)))
    
    # Doing train/validation/test split:
    x_train, x_val, y_train, y_val = train_test_split(imagefiles, steerings, test_size=0.30, random_state=randint(0,100))
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=randint(0,100))
    print ("Splits [Total {}]: Training = {}, Validation = {}, Test = {}".format(len(imagefiles), len(x_train), len(x_val), len(x_test)))

    # Memory-efficient generators
    traindata = datagen(x_train, y_train, trainer.get_image_shape(), flipprob=0.5)
    validationdata = datagen(x_val, y_val, trainer.get_image_shape(), flipprob=0.5)
    testdata = datagen(x_test, y_test, trainer.get_image_shape(), flipprob=0.5)

    # Generators that perform batching
    trainbatcher = batchgen(traindata, args.batch_size)
    validationbatcher = batchgen(validationdata, args.batch_size)
    testbatcher = batchgen(testdata, args.batch_size)

    # Train the basetrainer
    print ("Starting to train...")
    trainer.train_with(trainbatcher, len(x_train), validation_data=validationbatcher, nb_val_samples=len(x_val), num_epochs=args.num_epochs)
    
    print ("Running against the test split...")
    [loss, accuracy] = trainer.evaluate_with(testbatcher, len(x_test))

    print ("Evaluation results:")
    print ("Loss: ", loss)
    print ("Accuracy: ", accuracy)
    
    # Finally, determine what to do with the basetrainer
    if args.dry_run:
        print ("Dry run. Nothing will be saved.")
    else:
        trainer.checkpoint()
        trainer.log_command_args(args)

    print ("Thank you! Come again!")