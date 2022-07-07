#!pip install tf-nightly-gpu

import os
import sys
import PIL
import numpy as np
import math
import urllib
import glob
import shutil
from matplotlib import pyplot
from distutils.dir_util import copy_tree
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, MaxPool2D, MaxPooling2D, Flatten ,Dropout 
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

#pip install -U scikit-learn
from sklearn.utils import class_weight

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

import tensorflow as tf


print("starting")


#if we dont want to use the gpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#width=224
#height=224
width=90
height=90
batch_size=32

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024*8)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
else:
  batch_size=8
  print("no gpus")



datadir="./sorted/"


print("Gathering data")

# create data generator
datagen = ImageDataGenerator(rescale=1.0/255.0)

# prepare iterators  was class_mode='binary' class_mode='sparse' class_mode='categorical'
train_it = datagen.flow_from_directory(datadir, classes=None, class_mode='categorical', batch_size=64, target_size=(width, height), color_mode="rgb")
test_it = datagen.flow_from_directory(datadir, classes=None, class_mode='categorical', batch_size=64, target_size=(width, height), color_mode="rgb")



## Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(width, height, 3))
base_model.trainable = False ## Not trainable weights

#base_model.summary()


flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
#number of outputs
prediction_layer = layers.Dense(5, activation='softmax')


model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()


es = EarlyStopping(monitor='val_accuracy', mode='max', patience=15,  restore_best_weights=True)

print("Training")
#model.fit(train_it, train_labels, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])
accuracy = 0
while accuracy <= 90:
  history = model.fit(train_it, steps_per_epoch=len(train_it), batch_size=batch_size, epochs=80, validation_data=test_it, validation_steps=len(test_it), verbose=2, shuffle=True, callbacks=[es])
  accuracy = (history.history['accuracy'][-1] * 100)
  print("accuracy")
  print(accuracy)
  print('{: <{padding}}'.format("", padding=75), end='\r', flush=True) #flush output maybe?


# evaluate model
print("Evaluating")
_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
print('accuracy > %.3f' % (acc * 100.0))

# learning curves
#summarize_diagnostics(history)


# save model
print("Saving")
model.save('final_model.h5')
model.save('final_model', save_format='tf')


print("Done")






