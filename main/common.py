import os
from collections import namedtuple

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
        