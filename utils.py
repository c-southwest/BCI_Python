
from scipy import signal


class Filters():
    @classmethod
    def BandStop(cls, data, low=48, high=52, order=8, fs=250):
        # BandStop 48-52 hz
        notch_low = low*2/fs
        notch_high = high*2/fs
        b, a = signal.butter(order, [notch_low, notch_high], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, data)  # f为要过滤的信号
        return filtedData

    @classmethod
    def LowPass(cls, data, freq, order=8, fs=250):
        # LowPass
        low_pass_param = freq * 2 / fs
        b, a = signal.butter(order, low_pass_param, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, data)
        return filtedData

    @classmethod
    def HighPass(cls, data, freq, order=8, fs=250):
        high_pass_param = freq * 2 / fs
        b, a = signal.butter(8, high_pass_param, 'highpass')
        filtedData = signal.filtfilt(b, a, data)
        return filtedData
    # notch_low = 48*2/fs
    # notch_high = 52*2/fs
    # low_pass = 40 *2/fs
    # high_pass = 3 *2/fs

    # # BandStop 48-52 hz
    # b, a = signal.butter(8, [notch_low, notch_high], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
    # filtedData = signal.filtfilt(b, a, f)  # f为要过滤的信号
    #
    # # LowPass
    # b, a = signal.butter(8, low_pass, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
    # filtedData2 = signal.filtfilt(b, a, filtedData)
    #
    # # HighPass
    # b, a = signal.butter(8, high_pass, 'highpass')
    # filtedData3 = signal.filtfilt(b, a, filtedData2)


