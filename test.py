# import socket  # 导入 socket 模块
#
# s = socket.socket()  # 创建 socket 对象
# host = "127.0.0.1"  # 获取本地主机名
# port = 8080  # 设置端口号
#
# s.connect((host, port))
# print(s.recv(1024).decode("utf-8"))
#
# s.close()
# import time
# import winsound
#
#
# winsound.Beep(500,500)
# print(time.time())
#
#
#
# import tensorflow.keras as keras
#
#
# model = keras.models.load_model("alpha_detect_model_1s_3.keras")
#
# print(model.summary())
from brainflow import DataFilter
from utils import Filters
import matplotlib.pyplot as plt
from scipy.signal import welch

filename = f"./data/eye_closed_1s_3/2022-01-21-21_42_43.csv"
data = DataFilter.read_file(filename)
fs = 250
nfft = 1024
order = 5
plot_range = 250
# raw data
f = data[1][1:fs]
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title("原始信号")
plt.plot(f)
# raw psd data
psd = welch(f, fs, nfft=nfft)
plt.subplot(1, 2, 2)
plt.title("对应PSD")
plt.plot(psd[0][:plot_range], psd[1][:plot_range])
plt.show()

# 48-52Hz BandStop data
f = Filters.BandStop(data=f, low=48, high=52, order=order, fs=fs)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title("用48-52Hz带阻滤波器去除50Hz工频干扰")
plt.plot(f)
# psd data
psd = welch(f, fs, nfft=nfft)
plt.subplot(1, 2, 2)
plt.title("对应PSD")
plt.plot(psd[0][:plot_range], psd[1][:plot_range])
plt.show()

# 12Hz LowPass data
f = Filters.LowPass(data=f, freq=12, order=order, fs=fs)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title("用12Hz低通滤波器去除高频干扰")
plt.plot(f)
# psd data
psd = welch(f, fs, nfft=nfft)
plt.subplot(1, 2, 2)
plt.title("对应PSD")
plt.plot(psd[0][:plot_range], psd[1][:plot_range])
plt.show()

# 8Hz HighPass data
f = Filters.HighPass(data=f, freq=8, order=order, fs=fs)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title("用8Hz高通滤波器去除低频干扰")
plt.plot(f)
# psd data
psd = welch(f, fs, nfft=nfft)
plt.subplot(1, 2, 2)
plt.title("对应PSD")
plt.plot(psd[0][:plot_range], psd[1][:plot_range])
plt.show()