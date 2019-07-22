# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:08:16 2019

@author: hfuji
"""

from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import os
import glob
import cv2

image_dir = 'D:/Develop/Halcon/TrainDefectImages'
#image_path = get_files(neu_raw, '*.png')
image_path = glob.glob(image_dir + '/**/*.jpg', recursive=True)
N = 1800 # all images

seed = 2
np.random.seed(seed) # Use the same seed for shuffling if you want to load data from file later on
np.random.shuffle(image_path)
train_image_paths = image_path[0:N]

# Load the images
images = [cv2.imread(path) for path in train_image_paths]
images=[cv2.resize(image,(224,224)) for image in images ]
images = np.asarray(images)

# Get image size
image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
print(image_size)

# Scale
x_train = images / 255
# Read the labels from the filenames
n_images = images.shape[0]
y_train =[]
class_names = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
for i in range(n_images):
#    filename = os.path.basename(train_image_paths[i])[0]
    filename = os.path.basename(train_image_paths[i])
    idx = class_names.index(filename[:2])
        
    y_train.append(int(idx))
#

# parameters for architecture
input_shape = (224, 224, 3)
num_classes = 6
conv_size = 32

# parameters for training
batch_size = 32
num_epochs = 20

# load MobileNet from Keras
MobileNet_model = MobileNet(include_top=False, input_shape=input_shape)

# add custom Layers
x = MobileNet_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
Custom_Output = Dense(num_classes, activation='softmax')(x)

# define the input and output of the model
model = Model(inputs = MobileNet_model.input, outputs = Custom_Output)
        
# compile the model
model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

model.summary()

# train the model 
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_split=0.1)