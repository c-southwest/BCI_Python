from brainflow.data_filter import DataFilter, WindowFunctions

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

dt = 1/250
t = np.arange(0,3, dt)
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
f_clean = f
while True:
    f = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    f = f + 2.5*np.random.randn(len(t))
# plt.plot(t,f, label="noisy signal")
# plt.plot(t,f_clean, label="clean signal")
# plt.legend()
# plt.show()

    # FFT
    n = 1024
    fhat = np.fft.fft(f[:512],n)
    PSD = fhat * np.conj(fhat) / n
    max= np.max(PSD)
    PSD/=max
    freq = (1/(dt*n)) * np.arange(n)
    print("freq", len(freq))
    L = np.arange(1, np.floor(n/2),dtype='int')
    plt.subplot(2,2,1)
    plt.title(f"numpy {len(freq[L])}")
    plt.plot(freq[L],PSD[L])

    plt.subplot(2, 2, 2)
    psd = DataFilter.get_psd_welch(f[:512], 512, 0, 250, WindowFunctions.NO_WINDOW)
    max = np.max(psd[0])
    print("max",np.max(psd[1]))
    plt.title(f"brainflow get_psd_welch {len(psd[1])}")
    plt.plot(psd[1],psd[0]/max)

    plt.subplot(2,2,3)
    psd2 = DataFilter.get_psd(f[:512], 250, WindowFunctions.NO_WINDOW)
    max = np.max(psd2[0])
    plt.title(f"brainflow get_psd {len(psd2[1])}")
    plt.plot(psd2[1],psd2[0]/max)

    plt.subplot(2, 2, 4)
    psd3 = welch(f[:512], 1/dt, nfft=1024)
    max = np.max(psd3[1])
    plt.title(f"scipy psd welch {len(psd3[1])}")
    plt.plot(psd3[0], psd3[1]/max)
    plt.show()
