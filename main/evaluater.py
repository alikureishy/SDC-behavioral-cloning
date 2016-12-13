'''
Created on Dec 7, 2016

@author: safdar
'''
import argparse
from common import get_driving_logs, read_csv, extract, datagen, batchgen,\
    get_trainer, Center, Steer

if __name__ == '__main__':
    print ("###############################################")
    print ("#                  EVALUATOR                  #")
    print ("###############################################")
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('-n', '--model_name', dest='model_name', required=True, type=str, help='Name of basetrainer. (Without ''.json'' and ''.hd5'')')
    parser.add_argument('-a', '--arch', dest='arch', required=True, type=str, help='Architecture of model. [vgg16, googlenet, commaai, none]')
    parser.add_argument('-t', '--test_data_folders', dest='test_data_folders', required=True, nargs='*', type=str, help='Path to folder with test data')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=128, type=int, help='Batch size for test predictions')
    args = parser.parse_args()

    # Obtain the trainer
    trainer = get_trainer(args.arch, args.model_name)

    # Read test data
    driving_logs = get_driving_logs(args.test_data_folders)
    rows = read_csv(driving_logs)
    (imagefiles, steerings) = extract(rows, Center, Steer)

    # Memory-efficient generator for test data
    testdata = datagen(imagefiles, steerings, trainer.get_image_shape())
    testbatcher = batchgen(testdata, args.batch_size)

    # Invoke basetrainer on some data
    [loss, accuracy] = trainer.evaluate_with(testbatcher, len(imagefiles))

    print ("Evaluation results:")
    print ("Loss: ", loss)
    print ("Accuracy: ", accuracy)
    print ("Thank you! Come again!")