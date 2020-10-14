import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import cv2
import numpy as np
import glob
from random import randint

from mobilenetv2 import MobileNetv2

class_names = ["true_positive", "false_positive"]

IMG_SHAPE = (32, 32, 3)
model = MobileNetv2(input_shape = IMG_SHAPE, k = 2, alpha=1.0)
model.summary()

model.load_weights('./verification_mobilenetv2_model_try.h5')
src = glob.glob("/home/mju-125/test-server/data/val/false_positive/*.jpg")

test_images = []

num_test_imgs = 10

for i in range(num_test_imgs):
	index = randint(0, len(src)-1)

	img = cv2.imread(src[index])
	img = cv2.resize(img, (32,32))

	img=img/255.0

	test_images.append(img)

# img1 = img.reshape(1,32,32,3)
test_images = np.array(test_images)
print(test_images.shape)
prediction = model.predict(test_images)
print(prediction)
