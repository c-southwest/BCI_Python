import os

import matplotlib.pyplot as plt
import numpy.random
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np

# 读取静息态数据
eye_closed_psd_dir = "psd_data/eye_closed_1s_3/"
eye_closed_psd_files = os.listdir(eye_closed_psd_dir)
eye_closed_psd = []
for filename in eye_closed_psd_files:
    eye_closed_psd.append([np.load(eye_closed_psd_dir + filename), [1,0]])

# 读取标准状态数据
normal_psd_dir = "psd_data/normal_1s_3/"
normal_psd_files = os.listdir(normal_psd_dir)
normal_psd = []
for filename in normal_psd_files:
    normal_psd.append([np.load(normal_psd_dir + filename), [0,1]])

# 分配训练集 和 验证集
val = []
num_val = 100
val.extend(normal_psd[:num_val])
val.extend(eye_closed_psd[:num_val])
train = []
train.extend(normal_psd[num_val:])
train.extend(eye_closed_psd[num_val:])

# 打乱顺序
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

inputs = keras.Input(shape=eye_closed_psd[0][0].shape)  # 最后一个0代表数据本身，index=1为标签
x = layers.Dense(4, activation="sigmoid")(inputs)
#x = layers.Dense(8, activation="relu")(x)
#x = layers.Dense(16, activation="relu")(x)
x = layers.Flatten()(x)
#x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer="rmsprop", loss=keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
callbacks = [
    keras.callbacks.ModelCheckpoint("alpha_detect_model_1s_3.keras", save_best_only=True)
]
history = model.fit(train_data, train_label, batch_size=32, epochs=100, validation_data=(val_data,val_label), callbacks=callbacks)

# plot loss
epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# plot accuracy
epochs = range(1, len(history.history["accuracy"]) + 1)
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
plt.figure()
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.show()
