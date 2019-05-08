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
from keras.models import load_model
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
import imutils
import pickle
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
	# grab the paths to the input images in our dataset
	print("[INFO] quantifying faces...")
	imagePaths = list(paths.list_images(image_path))
	total = 0
	for (i, img_path) in enumerate(imagePaths):
		# extract the person name from the image path
		name = img_path.split(os.path.sep)[-2]
		print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))

		image = cv2.imread(img_path)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]
		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# ensure at least one face was found
		if len(detections) > 0:
			# we're making the assumption that each image has only ONE
			# face, so find the bounding box with the largest probability
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			# ensure that the detection with the largest probability also
			# means our minimum probability test (thus helping filter out
			# weak detections)
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI and grab the ROI dimensions
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				cv2.imshow("crop", face)
				cv2.waitKey()
				vec = img_to_encoding(face, FRmodel)

				# add the name of the person + corresponding face
				# embedding to their respective lists
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total+=1
	# dump the facial embeddings + names to disk
	print("[INFO] serializing {} encodings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	f = open("output/embeddings.pickle", "wb")
	f.write(pickle.dumps(data))
	f.close()
	#database[name] = img_to_encoding(img_path, FRmodel)
"""
FRmodel = faceRecoModel(input_shape=(3, 96, 96)) #Loading Inception block
print("Total Params: ", FRmodel.count_params())
print("[INFO] Loading weights...")
load_weights_from_FaceNet(FRmodel) #You could load your own weights
print("[INFO] Weights loaded...")

print("[INFO] Saving model...")

save_model(FRmodel, "FaceNet")
"""

print("[INFO] Loading model...")
FRmodel = load_model("FaceNet.h5")
print("[INFO] Model loaded...")

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detector", "deploy.prototxt.txt"])
modelPath = os.path.sep.join(["face_detector",
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

knownEmbeddings = []
knownNames = []
# initialize the total number of faces processed
total = 0

encode_dataset("datasets")
#who_dis("datasets/riho_2.jpg", database, FRmodel)
