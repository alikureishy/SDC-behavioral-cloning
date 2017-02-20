<!--- Adding links to various images used in this document --->
[Left]: https://github.com/safdark/behavioral-cloning/blob/master/doc/images/bc_illustration1.png "Left"
[Straight]: https://github.com/safdark/behavioral-cloning/blob/master/doc/images/bc_illustration2.png "Straight"
[LeftSharp]: https://github.com/safdark/behavioral-cloning/blob/master/doc/images/bc_illustration3.png "Left Sharp"
[CrossLineLeft]: https://github.com/safdark/behavioral-cloning/blob/master/doc/images/bc_illustration4.png "Cross Line Left"
[RightSharp]: https://github.com/safdark/behavioral-cloning/blob/master/doc/images/bc_illustration5.png "Right Sharp"

# Behavioral Cloning - Racetrack Driving

![LeftSharp]

## Overview

This application provides utilities to train a model of a car to steer along a simulated race traack, and to then allow that trained model to drive the same car autonomously on the same simulated track.

Note: This was one of my first forays into Python. Please excuse any non-pythonic code.

## Installation

This is a python utility requiring the following libraries to be installed prior to use:
* python (>= 3)
* numpy
* keras
* tensorflow
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


### Autonomous Driver (drive.py)

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

This section will go over various aspects of the design, including the training data, the data preparation, model architecture, and some miscellaneous other features.

### Different model architectures

For playing with different model architectures, I used inheritance:
- basetrainer.py
	- custom1trainer.py
	- custom2trainer.py
	- ...more coming...
	
BaseTrainer provides basic utilities like reading the model/weights from disk, and writing out to disk, and training.

Custom1 and Custom2 (at the moment) are the only two model options used, with each performing relatively the same as the other. I wanted to be able to switch between models on the fly, so as to try out their performance without much bookkeeping overhead. Adding more models requires creating a new subclass of BaseTrainer and overriding one or two methods to build the model pipeline and choose the compile options. This was probably an overkill for this particular project, but should come in handy for later, I'm hoping.

#### 'Custom1'


#### 'Custom2'


### Training Data:

Training data is generated using the simulator in 'training' mode. The training output from a given run is a folder containing:
- IMG: 			This is a folder containing all the training images, in groups of 3, for the left, center and right çamera images obtained at each instant of the simulation.
- driving_log.csv: 	CSV file containing the full paths to each group of images under IMG/, and the corresponding throttle, steering angle etc values corresponding to the 'correct' course(s) of action.

#### Recovery Data

Driving around the track would only generate data to try and keep the car in the middle of the road from losing its center position. However, data is also needed to train the model to return the car back to the center, in case of steering off too far to the left or right, which has a high likelihood of happening.

There were two options to achieve this:
* Training explicit recovery data
* Using the left and right camera images (with an adjustment to the steering value)

After attempting the second option for some time, I chose to instead implement the first option, since the former would require further investigation of the steering adjustments, for which I did not have the time. This is an area of further improvement.

I trained recovery data in training mode by taking the car closer to the edge, or over the edge, and then capturing the data when returning the car to the center. Here is an illustration:

![CrossLineLeft]

Recovery data is required for returning from both sides of the road, not just the left, and for sharper road curvatures as well, in which case the manouvering required is more extreme. Here is an illustration of a sharp right turn recorded on one of the sharpest points of the road, to avoid the comman problem of the car falling into the lake:

![RightSharp]

With the random vertical flip happening during training, such recovery data would also train the car to return to the center on a road sharply curving towards the left. Care has to be taken to record enough such data for the model to be able to generalize, but also not too much to cause the car to saw-tooth left-to-right in autonomous mode.

#### Absence of a Gaming Controller

Using a gaming controller (such as for PS3) to generate the training data would have yielded far better/smoother data for training this model. Keyboard data has the problem of having a majority of 0s, mixed with some choppy non-zero steering values. When used for training, this data causes the model to settle into a suboptimal minima (perhaps even a bad global minima), which manifests itself as a constant valued steering angle that gets returned to the simulator by the server. (For example, 0.0833246152). This is my conclusion because the value generated was awfully close to the mean of the training dataset. Since a majority of training data was zero, it actually makes sense because in that case, settling into a value close to the mean would likely achieve the lowest cost across the space of most reachable model weights closest to the randomly selected initial weight values (based on the glorot_uniform weight distribution algorithm used by default). Any other minima, in hindsight, would be inappropriate. A higher proportion of non-zero (smoothed) values is necessary to yield a better gradient landscape.

I did not have a gaming controller (and was reluctant to feed consumerism by purchasing one just for this project). So, I had to look for workarounds. One remedy for this was to smoothen the data to readjust zero values into a more uniform distribution within the steering range. However, this effort would require more investigation, for which I did not have sufficient time. This is an area of further improvement. In hindsight, buying the controller might have saved me even more time!

Instead of the above, I achieved a reasonable outcome by:
* *dropping* a majority of data that contained zero steering values,
* recording a wider range of recovery data than would otherwise have been required, and
* recording a lot more data overall, since dropping zeros significantly reduced the amount of training data available.

Still, absence of smoothing and/or a gaming controller and dropping of zeros meant that the training data was typically skewed further away from zero than it should be for a smoother ride. As a result:
* Autonomous driving was jittery
* Recovery actions were generally more sudden and occasionally excessive, causing the car to overshoot the center (saw-tooth).

The importance of good data cannot be overstated!

#### Data Augmentation/Adjustment

Before being used for training, the labeled images are passed through a 'training prep' pipeline, which includes:
* resizing the images to a fixed (configurable) size based on the network architecture: This allows flexibility for training the same model using training data of varying resolutions.
* normalize the pixels in each image read from disk: To achieve a better training outcome.
* randomly flipping images around the vertical axis (to achieve a 50/50 distribution): This renders it unnecessary to training data on the race course in the reverse direction.

Note: When processing real data through the trained model in autonomous mode, the last step (randomly flipping images) was not performed, for obvious reasons.

#### Data Preparation

A train/test/validation split (with a 70/15/15 ratio) is achieved using sklearn's train_test_split() API. The validation split is used to check the accuracy of the model at the end of each training epoch. The test set returns the final test output, the goal being to not overtrain data using the test set, but instead, to rely on just the validation data.

Shuffling is possible because the actual driving log does not contain images, and can therefore be held in memory in its entirety (at least for the needs of this course project), since each row of this log would only hold a few bytes of data. It is also trivial at this point to determine the size of an epoch of data.

Batching is performed using the 'Generator' functionality in python. By invoking keras' fit_generator() API, this generator can be used to generate batches of training data, only reading the images into memory when the fit_generator() retrieves it. At this point, the images are put through the 'training prep' pipeline, before being fed into the network.

### Other features
Here are a few desired outcomes toward which I targeted the design:
* Support a plug-and-play sort of mechanism to test different kinds of model architectures and associated weights without the administrative overhead of keeping track of numerous hyperparams.
* Continue training of an aborted iteration by saving models/weights after each epoch, and re-loading the latest saved trained models/weights on each launch of trainer.py.
* Read training batches from disk (so as to limit memory consumption). Batch size depends on how much is read from disk at a time.

## Limitations

### Throttle

A future version of this utility could not only return just steering angles, but also throttle values. At present the throttle is hard-coded and the same value is returned each time to the simulation client. Implementing this functionality will require training the corresponding model to output not just the steering angle, but also the throttle. Intuitively, this would be quite straightforward, since the throttle value would be inversely correlated to the steering angle being returned -- navigating a curved road would require a higher steering angle, and correspondngly a lower throttle value, and similarly, a straight road would require a lower steering angle, but a correspondingly higher throttle value.

A possible challenge, in a more advanced training scenario, would be to train the network not just on input images, but also with the present throttle value. This would achieve a more accurate and responsive network than one that would be trained just with input images. It is left as a future enhancement.

### Using left and right camera images for automatic recovery training

As discussed above, the left and right camera images could theoretically be used as training data by adjusting the steering angle corresponding to that row of data in driving_log.csv. The specific amount of this adjustment required investigation, for which I did not have the time, since training the model itself took quite a while (generally >10 mins). Furthermore, the absence of smooth training data would have made this hyperparam search even more exhausting and time consuming. There is precedent however, for successfully using these images, so this remains an item for future enhancement.

### Data Smoothing

Smoothing of data would be the closest approximation to using a gaming controller, which was the right way of obtaining trainign data. However, the overhead of investigating the right smoothing algorithm was ground for punting this feature.

I envision the smoothing to have occurred only for data within the 1 standard deviation of the mean, so as to not lose important data points outside that range. This is important since all the data outside that range would be accurate already. Only the data centered on zero was to be distributed, and 1 standard deviation (or perhaps even just a fixed range from -1 to 1), in my view, would be the range to stick within, for any kind of smoothing algorithm to not overstep its utility.
