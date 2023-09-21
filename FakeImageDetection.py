""" Author: Colin Antony"""

#standard imports
import keras.utils
from keras import layers , models, optimizers, applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPool2D
from keras.metrics import Precision, Recall

data_src= "C:/Colin/MoonRaft/data" # Replace with actual directory

batch_size = 32
image_x = 32
image_y = 32

# Load training data with label
train_ds = keras.utils.image_dataset_from_directory(
    data_src +"/train",
    seed=3,
    image_size=(image_x, image_y),
    batch_size=batch_size,
)

# load testing data with label
val_ds = keras.utils.image_dataset_from_directory(
    data_src +"/test",
    seed=3,
    image_size=(image_x, image_y),
    batch_size=batch_size
)

# Verify labels of training and validation data
print("Training Classes:")
class_names = train_ds.class_names
print(class_names)

print("Testing Classes:")
class_names = val_ds.class_names
print(class_names)

#Building the model
model = Sequential()

model.add(keras.layers.Rescaling(1./255.))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (image_x,image_y,3)))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation = "sigmoid")) # Dense layer with 1 output neuron as it is a binary classifier


model.build(input_shape=(None,image_x,image_y,3))

model.summary()

model.compile(optimizer="rmsprop", loss='binary_crossentropy',metrics=['accuracy', keras.metrics.Precision(),keras.metrics.Recall()])

# Fitting the model
model.fit(train_ds,
          epochs=5,
          validation_data=val_ds)

model.save("FakeImageDetector.h5") # saving the model for later use