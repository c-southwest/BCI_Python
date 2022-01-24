import os

import matplotlib.pyplot as plt
import numpy.random
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np

ssvep_7_psd_dir = "psd_data/SSVEP_7_2/"
ssvep_7_psd_files = os.listdir(ssvep_7_psd_dir)
ssvep_7_psd = []
for filename in ssvep_7_psd_files:
    ssvep_7_psd.append([np.load(ssvep_7_psd_dir + filename), [1, 0, 0, 0, 0]])

ssvep_8_psd_dir = "psd_data/SSVEP_8_2/"
ssvep_8_psd_files = os.listdir(ssvep_8_psd_dir)
ssvep_8_psd = []
for filename in ssvep_8_psd_files:
    ssvep_8_psd.append([np.load(ssvep_8_psd_dir + filename), [0, 1, 0, 0, 0]])

ssvep_9_psd_dir = "psd_data/SSVEP_9_2/"
ssvep_9_psd_files = os.listdir(ssvep_9_psd_dir)
ssvep_9_psd = []
for filename in ssvep_9_psd_files:
    ssvep_9_psd.append([np.load(ssvep_9_psd_dir + filename), [0, 0, 1, 0, 0]])

ssvep_10_psd_dir = "psd_data/SSVEP_10_2/"
ssvep_10_psd_files = os.listdir(ssvep_10_psd_dir)
ssvep_10_psd = []
for filename in ssvep_10_psd_files:
    ssvep_10_psd.append([np.load(ssvep_10_psd_dir + filename), [0, 0, 0, 1, 0]])

ssvep_none_psd_dir = "psd_data/SSVEP_None_2/"
ssvep_none_psd_files = os.listdir(ssvep_none_psd_dir)
ssvep_none_psd = []
for filename in ssvep_none_psd_files:
    ssvep_none_psd.append([np.load(ssvep_none_psd_dir + filename), [0, 0, 0, 0, 1]])

# 分配训练集 和 验证集
val = []
num_val = 60
val.extend(ssvep_7_psd[:num_val])
val.extend(ssvep_8_psd[:num_val])
val.extend(ssvep_9_psd[:num_val])
val.extend(ssvep_10_psd[:num_val])
#val.extend(ssvep_none_psd[:num_val])

train = []
train.extend(ssvep_7_psd[num_val:])
train.extend(ssvep_8_psd[num_val:])
train.extend(ssvep_9_psd[num_val:])
train.extend(ssvep_10_psd[num_val:])
#train.extend(ssvep_none_psd[num_val:])

# 打乱顺序
np.random.shuffle(val)
np.random.shuffle(train)

train_data = []
train_label = []
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

inputs = keras.Input(shape=ssvep_7_psd[0][0].shape)  # 最后一个0代表数据本身，index=1为标签
x = layers.Dense(8, activation="sigmoid")(inputs)
# x = layers.Dense(8, activation="sigmoid")(x)
# x = layers.Dense(16, activation="relu")(x)
x = layers.Flatten()(x)
# x = layers.Dropout(0.5)(x)
outputs = layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer="rmsprop", loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
callbacks = [
    keras.callbacks.ModelCheckpoint("ssvep_detect_model.keras", save_best_only=True)
]
history = model.fit(train_data, train_label, batch_size=32, epochs=200, validation_data=(val_data, val_label),
                    callbacks=callbacks)

epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()
