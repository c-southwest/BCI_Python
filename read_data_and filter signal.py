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

# f"./data/eye_closed/2021-12-13-22_52_42.csv"
mode = "normal"
filename = f"./data/SSVEP_7_wet/2022-01-26-00_46_22.csv"
data = DataFilter.read_file(filename)

eeg_channels = BoardShim.get_eeg_channels(0)

print(len(data[8]))

#exit()
fs = 250
nfft = 1024
f = data[8][1:3*fs]
f = f - np.mean(f)

filtedData = Filters.BandStop(data=f, low=48, high=52, order=5, fs=fs)
filtedData2 = Filters.LowPass(data=filtedData, freq=12, order=5, fs=fs)
filtedData3= Filters.HighPass(data=filtedData2, freq=8, order=5, fs=fs)

plt.subplot(4,2,1)
plt.plot(f)
plt.subplot(4,2,2)
psd = welch(f, fs, nfft=nfft)
plt.title(f"Raw psd {len(psd[1][:])}")
plt.plot(psd[0][:], psd[1][:])


plt.subplot(4,2,3)
plt.plot(filtedData)
plt.subplot(4,2,4)
psd = welch(filtedData, fs, nfft=nfft)
plt.title(f"48-50Hz removed psd {len(psd[1][:])}")
plt.plot(psd[0][:], psd[1][:])


plt.subplot(4,2,5)
plt.plot(filtedData2)
plt.subplot(4,2,6)
psd = welch(filtedData2, fs, nfft=nfft)
plt.title(f"40hz Low Pass Filter psd {len(psd[1][:])}")
plt.plot(psd[0][:], psd[1][:])

plt.subplot(4,2,7)
plt.plot(filtedData3)
plt.subplot(4,2,8)
psd = welch(filtedData3, fs, nfft=nfft)
plt.title(f"3hz High Pass Filter psd {len(psd[1][:])}")
plt.plot(psd[0][:], psd[1][:])
plt.tight_layout()
plt.show()



f = filtedData3
seg = 40
points = int((seg/(fs/2))*len(psd[0]))
psd = welch(f, fs, nfft=nfft)
plt.title(f"scipy psd welch {len(psd[1][:points])}")
plt.plot(psd[0][:points], psd[1][:points])
plt.show()

arr = np.array(psd)
print(arr.shape)
print(arr.dtype)
print(arr)