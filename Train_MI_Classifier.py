import os

import matplotlib.pyplot as plt
import numpy.random
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np

# Read left data
MI_left_psd_dir = "psd_data/MI_left/"
MI_left_psd_files = os.listdir(MI_left_psd_dir)
MI_left_psd = []
for filename in MI_left_psd_files:
    MI_left_psd.append([np.load(MI_left_psd_dir + filename), [1,0,0]])

# Read normal data
normal_psd_dir = "psd_data/MI_none/"
normal_psd_files = os.listdir(normal_psd_dir)
normal_psd = []
for filename in normal_psd_files:
    normal_psd.append([np.load(normal_psd_dir + filename), [0,1,0]])

# Read right data
MI_right_psd_dir = "psd_data/MI_right/"
MI_right_psd_files = os.listdir(MI_right_psd_dir)
MI_right_psd = []
for filename in MI_right_psd_files:
    MI_right_psd.append([np.load(MI_right_psd_dir + filename), [0,0,1]])

# create Validation set and Training set
val = []
num_val = 100
val.extend(normal_psd[:num_val])
val.extend(MI_left_psd[:num_val])
val.extend(MI_right_psd[:num_val])
train = []
train.extend(normal_psd[num_val:])
train.extend(MI_left_psd[num_val:])
train.extend(MI_right_psd[num_val:])

# Shuffle
np.random.shuffle(val)
np.random.shuffle(train)

train_data = []
train_label =[]
for data, label in train:
    train_data.append(data)
    train_label.append(label)

val_data = []
val_label = []
for data, label in val:
    val_data.append(data)
    val_label.append(label)

train_data = np.array(train_data)
train_label = np.array(train_label)
val_data = np.array(val_data)
val_label = np.array(val_label)

inputs = keras.Input(shape=MI_left_psd[0][0].shape) 

x = layers.Dense(8, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
x = layers.Dense(8, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01))(x)

#x = layers.Dense(4, activation="relu")(x)
x = layers.Flatten()(x)
#x = layers.Dropout(0.5)(x)
outputs = layers.Dense(3, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer="rmsprop", loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
callbacks = [
    keras.callbacks.ModelCheckpoint("MI_Classifier_model.keras", save_best_only=True)
]
history = model.fit(train_data, train_label, batch_size=32, epochs=100, validation_data=(val_data,val_label), callbacks=callbacks)


epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
plt.figure()
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.show()
