'''
Created on Dec 7, 2016

@author: safdar
'''
import argparse
from common import Params, XYGenerator, XYBatcher
from model import read_model

if __name__ == '__main__':
    print ("###############################################")
    print ("#                  EVALUATOR                  #")
    print ("###############################################")
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('-mn', '--model_name', dest='model_name', required=True, type=str, help='Name of model. (Without ''.json'' and ''.hd5'')')
    parser.add_argument('-tdf', '--test_data_folder', dest='test_data_folder', default='.', type=str, help='Path to folder with test data')
    parser.add_argument('-tbs', '--test_batch_size', dest='test_batch_size', default=128, type=int, help='Batch size for test predictions')
    args = parser.parse_args()
    
    params = Params(args.model_name,
                    test_data_folder=args.test_data_folder,
                    test_batch_size=args.test_batch_size)
    
    testgenerator = XYGenerator(params.test_log_file)
    testbatcher = XYBatcher(params.image_shape, params.test_batch_size, testgenerator)

    # Load the model
    model = read_model(params.model_name)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    # Invoke model on some data
    [loss, accuracy] = model.evaluate_generator(testbatcher.get_generator(), testbatcher.count())
    
    print ("Evaluation results:")
    print ("Loss: ", loss)
    print ("Accuracy: ", accuracy)
    print ("Thank you! Come again!")