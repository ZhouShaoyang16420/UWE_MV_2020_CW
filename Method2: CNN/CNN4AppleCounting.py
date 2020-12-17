import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

IMAGE_SIZE = 50

DATADIR = "C:/Users/Amon/Desktop/study/MV/dotaset/counting/train/images"

CATEGORIES = ["0apple", "1apple", "2apple", "3apple", "4apple", "5apple", "6apple"]

# for category in CATEGORIES:
#     path = os.path.join(DATADIR,category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path,img))
#         plt.imshow(img_array)
#         plt.show()
#         break
#     break
# print(img_array)

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([new_array,class_num])
            except Exception:
                pass
create_training_data()

print(len(training_data))

import random

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])
X = []
y = []

for feature,label in training_data:
    X.append(feature)
    y.append(label)
print(X[0].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1))
X = np.array(X)
y = np.array(y)
print(len(X))
print('-------------------------------------------------------------------------------------------------------------------------------')
print(len(y))
print(y)
import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

import tensorflow as tf
from tensorflow.keras import datasets,layers,models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)

