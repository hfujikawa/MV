# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:59:56 2019

@author: hfuji
"""

import keras
print(keras.__version__)
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Dense(1000, activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(200, activation='sigmoid'))
model.add(layers.Dense(20, activation='sigmoid'))
model.add(layers.Dense(1))

model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['mean_absolute_error', 'mean_squared_error'])
