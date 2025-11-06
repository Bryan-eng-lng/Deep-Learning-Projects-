import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import  datasets , layers , models
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense , Flatten , Conv2D , MaxPooling2D , BatchNormalization


(X_train , y_train) , (X_test , y_test) = datasets.cifar10.load_data()


print(X_train.shape)
print(y_train.shape)

plt.imshow(X_train[4])
plt.show()


X_train = X_train / 255
X_test = X_test / 255

model = Sequential()

model.add(Conv2D(32,(3,3),padding = "valid",activation="relu",input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding = "valid",activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding = "valid",activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(10,activation="softmax"))


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(X_train , y_train , epochs=10)




