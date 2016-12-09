'''
Created on Dec 7, 2016

@author: safdar
'''
import argparse
from common import Params
from model import read_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model_name', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    
    # Read the data
    # It will look for a driving_log.csv file in this folder, and it will train on that data.
    params = Params(args.model_name)
    
    # Train the model
    model = read_model(params.model_name)
    
    # Invoke model on some data
