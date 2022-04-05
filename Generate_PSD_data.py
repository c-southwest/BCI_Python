import datetime
import time
import winsound
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
from scipy.signal import welch
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowFunctions, FilterTypes
import matplotlib.pyplot as plt
from utils import Filters
import os

# 信号处理设定
fs = 250
nfft = 1024
# Alpha Wave parameters
# freq_left = 8
# freq_right = 12
freq_left = 4
freq_right = 23
order = 5

# 数据读写设定
data_dir = "data/SSVEP_7_wet/"
# Alpha Wave parameters
# channels = [1, 2, 3, 4, 5, 6, 7, 8]
# channels = [7, 8]
channels = [8]
files = os.listdir(data_dir)
time_window = 3
slide_window = 0.5
out_dir = "psd_data/SSVEP_7_wet/"

for f in files:
    # 读取一条10s的数据
    fpath = data_dir + f
    print(fpath)
    data = DataFilter.read_file(fpath)
    # 初始化时间窗口
    start = 1
    end = start + time_window * fs
    count = 0
    while start + time_window * fs * 0.8 < len(data[1]):
        psd_data = []
        # 在给定时间段内，组合通道产生的psd数据
        for index, channel in enumerate(channels):
            ch = data[channel][start:end]
            filtedData = Filters.BandStop(data=ch, low=48, high=52, order=order, fs=fs)
            filtedData2 = Filters.LowPass(data=filtedData, freq=freq_right, order=order, fs=fs)
            filtedData3 = Filters.HighPass(data=filtedData2, freq=freq_left, order=order, fs=fs)
            psd = welch(filtedData3, fs, nfft=nfft)
            freq_end = 40  # 只关心40hz之前的PSD
            points = int((freq_end / (fs / 2)) * len(psd[0]))
            psd_data.append(psd[1][:points])

        np_psd = np.array(psd_data)
        np.save(out_dir + os.path.splitext(f)[0] + "_"+str(count), np_psd)
        # 推进时间窗口
        start = start + int(fs*slide_window)
        end = start + time_window * fs
        count += 1
    print("写入:", str(count + 1))
