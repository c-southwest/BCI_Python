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

# Signal processing settings
fs = 250
nfft = 1024
# Alpha Wave parameters
# freq_left = 8
# freq_right = 12
freq_left = 4
freq_right = 23
order = 5

# Data r/w settings
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
    # Read data
    fpath = data_dir + f
    print(fpath)
    data = DataFilter.read_file(fpath)
    # Initialize time window
    start = 1
    end = start + time_window * fs
    count = 0
    while start + time_window * fs * 0.8 < len(data[1]):
        psd_data = []
        # Combine PSD data
        for index, channel in enumerate(channels):
            ch = data[channel][start:end]
            filtedData = Filters.BandStop(data=ch, low=48, high=52, order=order, fs=fs)
            filtedData2 = Filters.LowPass(data=filtedData, freq=freq_right, order=order, fs=fs)
            filtedData3 = Filters.HighPass(data=filtedData2, freq=freq_left, order=order, fs=fs)
            psd = welch(filtedData3, fs, nfft=nfft)
            freq_end = 40  # Only care PSD data before 40hz
            points = int((freq_end / (fs / 2)) * len(psd[0]))
            psd_data.append(psd[1][:points])

        np_psd = np.array(psd_data)
        np.save(out_dir + os.path.splitext(f)[0] + "_"+str(count), np_psd)
        # Update time window
        start = start + int(fs*slide_window)
        end = start + time_window * fs
        count += 1
    print("Write: ", str(count + 1))
