# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:11:41 2019
https://github.com/arkitahara/neu_vgg16/blob/master/notebooks/neu_features.ipynb
https://gist.github.com/f-rumblefish/64787ad40f3f0f7feeca70db941eea99
@author: hfuji
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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

# build the model
model = Sequential()

model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same', input_shape=input_shape)) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
    
model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
    
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

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