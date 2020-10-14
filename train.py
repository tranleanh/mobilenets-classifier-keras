import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import cv2
print(tf.__version__)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mobilenetv2 import MobileNetv2

batch_size = 32
image_size = 28

data_folder = "data_folder_name"

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1./255)

# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        f'/home/mju-125/test-server/{data_folder}/train',  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to image_size x image_size
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        f'/home/mju-125/test-server/{data_folder}/val',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary')

IMG_SHAPE = (image_size, image_size, 3)
model = MobileNetv2(input_shape = IMG_SHAPE, k = 1, alpha=1.0)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

# model.load_weights('pretrained-weights.h5')

model.fit_generator(
        train_generator,
        steps_per_epoch=200000 // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=10000 // batch_size)

model.save_weights('./models/model.h5')

