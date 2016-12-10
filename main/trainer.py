'''
Created on Dec 7, 2016

@author: safdar
'''
import argparse
import os
from common import Params, XYGenerator, XYBatcher
from os.path import isfile
from model import read_model, create_model, write_model

if __name__ == '__main__':
    print ("###############################################")
    print ("#                   TRAINER                   #")
    print ("###############################################")

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
        i = 0
        filename = params.model_json_file
        while isfile(filename):
            i += 1
            filename = params.model_name+'_'+str(i)+'.json'
        write_model(model, params.model_name+'_'+str(i))
    else:
        print ("Dry run. Nothing will be saved.")

    print ("Thank you! Come again!")