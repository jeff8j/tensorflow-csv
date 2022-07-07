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
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

import tensorflow as tf

import pandas as pd
import re    # To match regular expression for extracting labels



batch_size=32



print("Gathering data")
files = glob.glob("training_data/**/*.csv", recursive=True)
print( "Files:",len(files) )


class CustomSequence(tf.keras.utils.Sequence):  # It inherits from `tf.keras.utils.Sequence` class
  def __init__(self, filenames, batch_size):  # Two input arguments to the class.
        self.filenames= filenames
        self.batch_size = batch_size

  def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

  def __getitem__(self, idx):  # idx is index that runs from 0 to length of sequence
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size] # Select a chunk of file names
        data = []
        labels = []
        label_classes = ["bad", "good"]

        for file in batch_x:   # In this loop read the files in the chunk that was selected previously
            temp = pd.read_csv(open(file,'r')) # Change this line to read any other type of file
            #print(temp.values.shape)
            data.append(temp.values.reshape(500,4,-1)) # Convert column data to matrix like data with one channel
            #print(temp.values.shape)
            #pattern = "/.*?/" + eval("file[14:21]")      # Pattern extracted from file_name
            #print(pattern)
            for j in range(len(label_classes)):
                if "/"+label_classes[j]+"/" in file:
                    labels.append(j)  
        #original #data = np.asarray(data).reshape(-1,32,32,1)
        #data = np.asarray(data).reshape(-1,500,4)
        data = np.asarray(data).reshape(-1,32,32,4)
        labels = np.asarray(labels)
        print("Labels: ", labels)
        return data, labels


print("Training")

train_sequence = CustomSequence(filenames = files, batch_size = batch_size)
val_sequence = CustomSequence(filenames = files, batch_size = batch_size)
test_sequence = CustomSequence(filenames = files, batch_size = batch_size)

model = tf.keras.Sequential([
    layers.Conv2D(16, 3, activation = "relu", input_shape = (32,32,1)),
    layers.MaxPool2D(2),
    layers.Conv2D(32, 3, activation = "relu"),
    layers.MaxPool2D(2),
    layers.Flatten(),
    layers.Dense(16, activation = "relu"),
    layers.Dense(5, activation = "softmax")
])
model.summary()

print("Training - Compile")
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

print("Training - Fit")
model.fit(train_sequence, validation_data = val_sequence, epochs = 10)



# save model
print("Saving")
#model.save('final_model.h5')
#model.save('final_model', save_format='tf')


print("Done")


