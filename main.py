from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from imutils import paths

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    encode_a_p = tf.reduce_sum(tf.square(anchor - positive), axis=-1)

    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    encode_a_n = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = encode_a_p - encode_a_n + alpha

    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    ### END CODE HERE ###

    return loss

def verify(image_path, identity, database, model):
    """
    Face verification: 1:1 Problem

    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    encoding = img_to_encoding(image_path, model)

    dist = np.linalg.norm(encoding - database[identity])

    if dist < 0.7:
        print("[INFO] Welcome back, "+ identity + "sama")
        door_open=True
    else:
        print("[INFO] Please go away :(")
        door_open=False

    return dist, door_open

def who_dis(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    dist_dict = {} #Contains (encoded) distance btwn target & database
    for (person, encoding) in database.items():
        dist = np.linalg.norm(encoding - database[person])
        dist_dict[person].append([dist])

    dist_dict
    if dist < 0.7:
        print("[INFO] Welcome back, "+ person + "sama")
        door_open=True
    else:
        print("[INFO] Please go away :(")
        door_open=False

def encode_dataset(image_path):
    imagePaths = paths.list_images(image_path)
    for img_path in imagePaths:
        labels = img_path.split("/")[1]
        database[labels] = img_to_encoding(img_path, FRmodel)

FRmodel = faceRecoModel(input_shape=(3, 96, 96)) #Loading Inception block
print("Total Params: ", FRmodel.count_params())
FRmodel.compile(optimizer="adam", loss=triplet_loss, metrics=['accuracy'])
print("[INFO] Loading weights...")
load_weights_from_FaceNet(FRmodel) #You could load your own weights
print("[INFO] Weights loaded...")

database = {}

encode_dataset("datasets")
who_dis("datasets/riho_2.jpg", database, FRmodel)
