<!--- Adding links to various images used in this document --->
[Simulator-Straight]: https://github.com/safdark/behavioral-cloning/blob/master/doc/images/illustration1.png "Straight"
[Simulator-LeftTurn]: https://github.com/safdark/behavioral-cloning/blob/master/doc/images/illustration2.png "Left Turn"
[Simulator-SharpLeftTurn]: https://github.com/safdark/behavioral-cloning/blob/master/doc/images/illustration3.png "Sharp Left Turn"
[Simulator-SharpRightTurn]: https://github.com/safdark/behavioral-cloning/blob/master/doc/images/illustration4.png "Sharp Right Turn"

# Behavioral Cloning - Racetrack Driving

![Simulator-SharpRightTurn]

## Overview

This application provides utilities to train a model of a car to steer along a simulated race traack, and to then allow that trained model to drive the same car autonomously on the same simulated track.

Note: This was one of my first forays into Python. Please excuse any non-pythonic code.

## Installation

This is a python utility requiring the following libraries to be installed prior to use:
* python (>= 3)
* numpy
* keras
* PIL
* scikit-learn
* OpenCV3
* matplotlib

## Execution

### Trainer (model.py)

This is the trainer that given a network architecture (either in code, or via an existing model.json), iterates over the weights in order to minimze the cost function. Each invocation would specify the number of epochs to train over. Since the model is committed to disk each time, and relaoded from disk on subsequent iterations, it is feasible to run the training utility iteratively until a satisfactory performance is achieved.

It is a command line utility, with a sample invocation as follows:

```
/Users/safdar/git/behavioral-cloning/main> python3.5 model.py -n custom2/ -t ../data/track2_forward_turns/ -e 1 -b 100 -a custom2
```

Here's the help menu:

```
usage: model.py [-h] -n MODEL_NAME -a ARCH -t
                [TRAINING_DATA_FOLDERS [TRAINING_DATA_FOLDERS ...]]
                [-e NUM_EPOCHS] [-b BATCH_SIZE] [-o] [-r]

Remote Driving

optional arguments:
  -h, --help            show this help message and exit
  -n MODEL_NAME, --model_name MODEL_NAME
                        Name of FOLDER where the model files reside.
  -a ARCH, --arch ARCH  Architecture of model. [custom1, custom2]
  -t [TRAINING_DATA_FOLDERS [TRAINING_DATA_FOLDERS ...]], --training_data_folders [TRAINING_DATA_FOLDERS [TRAINING_DATA_FOLDERS ...]]
                        Space-separated list of 1 or more folders containing
                        training data (driving_log and IMG folders).
  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of epochs to train on prior to checkpointing
                        the updated model.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Size of the training, validation and test batches.
  -o, --overwrite       Overwrite model files after training if they already
                        exist (default: false)
  -r, --dry_run         Dry run. Will not attempt to save anything.
```

The '-' option, when specified, causes the trainer to write back a trained model over any existing model/weight file that it was previously read from. If not specified, subsequently trained generations of the model are written out to new model/weight files suffixed with an incremental integer value.


### Driver (drive.py)

This is the server that 'drives' the car when the simulation is in autonomous mode. The simulation acts as the client, requesting navigation instructions for each image. The server receives each image from the simulation (client), and then feeds it through the trained neural network model to determine the steering angle required at that point, which it then returns to the client for enforcement.

It is a command line utility, with a sample invocation as follows:

```
/Users/safdar/git/behavioral-cloning/main> python3.5 drive.py -n custom2 -a custom2
```

Here's the help menu:

```
usage: drive.py [-h] -n MODEL_NAME -a ARCH

Remote Driving

optional arguments:
  -h, --help            show this help message and exit
  -n MODEL_NAME, --model_name MODEL_NAME
                        Path to basetrainer definition json. Model weights
                        should be on the same path. If they don't exist, a new model will
			be built using a glorot_uniform weight distribution and cannot be
			expected to perform until trained over some iterations.
  -a ARCH, --arch ARCH  Architecture of model to use, ['custom1', or 'custom2'],
  			if one isn't found on disk.
```

## Design

I have tried to design the utility in a way that allowed me to:
* Support a plug-and-play sort of mechanism to easily build and test different kinds of model architectures and associated weights.
* Re-load existing trained models/weights if they were already saved to disk.
* Read training batches from disk (so as to limit memory consumption). Batch size depends on how much is read from disk at a time.

### Training Data:
This is a work in progress. Due to the absence of a the requisite gaming control, I had to generate training data
using the keyboard, which causes several problems:
	- The driving can be jittery at times
	- Recovery from road sides back towards the road are occasionally sudden, excessive, or insufficient.
I did not have access to a gaming control (such as for PS3) to generate the training data, which is a crucial step
for training this model. Keyboard data has the disadvantage of having a majority of 0s, which can cause the model
to settle into a suboptimal local minima during training. This manifests itself as a constant valued steering angle
that gets returned to the simulator. Generating more recovery data (of the car returning to the center after veering
off to the sides) helps avoid, or get out of, these local minima. The importance of good data cannot be overstated!

### Data Manipulation

Training data is generated using the simulator in 'training' mode. The training output from a given run is a folder containing:
- IMG: 			This is a folder containing all the training images, in groups of 3, for the left, center and right çamera images obtained at each instant of the simulation.
- driving_log.csv: 	CSV file containing the full paths to each group of images under IMG/, and the corresponding throttle, steering angle etc values corresponding to the 'correct' course(s) of action.

Before being used for training, the labeled images are passed through a 'training prep' pipeline, which includes:
* resizing the images to a fixed (configurable) size based on the network architecture: This allows flexibility for training the same model using training data of varying resolutions.
* normalize the pixels in each image read from disk: To achieve a better training outcome.
* randomly flipping images around the vertical axis (to achieve a 50/50 distribution): This renders it unnecessary to training data on the race course in the reverse direction.

### Data Preparation

A train/test/validation split (with a 70/15/15 ratio) is achieved using sklearn's train_test_split() API. The validation split is used to check the accuracy of the model at the end of each training epoch. The test set returns the final test output, the goal being to not overtrain data using the test set, but instead, to rely on just the validation data.

Shuffling is possible because the actual driving log does not contain images, and can therefore be held in memory in its entirety (at least for the needs of this course project), since each row of this log would only hold a few bytes of data. It is also trivial at this point to determine the size of an epoch of data.

Batching is performed using the 'Generator' functionality in python. By invoking keras' fit_generator() API, this generator can be used to generate batches of training data, only reading the images into memory when the fit_generator() retrieves it. At this point, the images are put through the 'training prep' pipeline, before being fed into the network.

### Different model architectures

For playing with different model architectures, I used inheritance:
- basetrainer.py
	- custom1trainer.py
	- custom2trainer.py
	- ...more coming...
	
BaseTrainer provides basic utilities like reading the model/weights from disk, and writing out to disk, and training.

Custom1 and Custom2 (at the moment) are the only two model options. I wanted to be able to switch between models on the fly, so as to try out their performance without much bookkeeping overhead. Adding more models requires creating a new subclass of BaseTrainer and overriding one or two methods to build the model pipeline and choose the compile options. This was probably an overkill for this particular project, but should come in handy for later, I'm hoping.

## Limitations

### Throttle

A future version of this utility could not only return just steering angles, but also throttle values. At present the throttle is hard-coded and the same value is returned each time to the simulation client. Implementing this functionality will require training the corresponding model to output not just the steering angle, but also the throttle. Intuitively, this would be quite straightforward, since the throttle value would be inversely correlated to the steering angle being returned -- navigating a curved road would require a higher steering angle, and correspondngly a lower throttle value, and similarly, a straight road would require a lower steering angle, but a correspondingly higher throttle value.

A possible challenge, in a more advanced training scenario, would be to train the network not just on input images, but also with the present throttle value. This would achieve a more accurate and responsive network than one that would be trained just with input images. It is left as a future enhancement.


