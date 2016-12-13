import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2
from os.path import isfile
from common import get_trainer, resize_image, normalize_image

sio = socketio.Server()
app = Flask(__name__)
trainer = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image = np.asarray(image)
    image = resize_image(image, trainer.get_image_shape())
    image = normalize_image(image)
    transformed_image_array = image[None, :, :, :]

    # This basetrainer currently assumes that the features of the basetrainer are just the images. Feel free to change this.
    steering_angle = float(trainer.predict(transformed_image_array, batchsize=1))
    # The driving basetrainer currently just outputs a constant throttle. Feel free to edit this.
    throttle = 20
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    print ("###############################################")
    print ("#                   SERVER                    #")
    print ("###############################################")
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('-n', '--model_name', type=str, required=True, help='Path to basetrainer definition json. Model weights should be on the same path.')
    parser.add_argument('-a', '--arch', dest='arch', required=True, type=str, help='Architecture of model. [vgg16, googlenet, commaai, none]')
    args = parser.parse_args()

    # Get hold of the trainer
    trainer = get_trainer(args.arch, args.model_name)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)