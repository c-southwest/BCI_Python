import socket
import time
import winsound
import tensorflow.keras as keras
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from utils import Filters
from scipy.signal import welch

model = keras.models.load_model("alpha_detect_model_1s_3.keras")

# 参数定义 Cyton板的id=0，我机器上的串口号为COM3
params = BrainFlowInputParams()
params.serial_port = "COM3"
board_id = BoardIds.CYTON_BOARD  # 合成版id=-1，cyton板id=0
board = BoardShim(board_id, params)

# Start stream
board.prepare_session()
board.start_stream()

channels = [1, 2, 3, 4, 5, 6, 7, 8]
time_window = 1
fs = 250
slide_window = 0.2
delay_time = time_window + slide_window
nfft = 1024
freq_left = 8
freq_right = 12
order = 5

# socket
s = socket.socket()
host = "127.0.0.1"
port = 8080
s.bind((host, port))
s.listen(5)
client, address = s.accept()

time.sleep(2)
closed_time = 0  # 闭眼时刻
while True:
    time.sleep(slide_window)
    data = board.get_current_board_data(fs * time_window)

    psd_data = []
    # 在给定时间段内，组合通道产生的psd数据
    for index, channel in enumerate(channels):
        ch = data[channel]
        filtedData = Filters.BandStop(data=ch, low=48, high=52, order=order, fs=fs)
        filtedData2 = Filters.LowPass(data=filtedData, freq=freq_right, order=order, fs=fs)
        filtedData3 = Filters.HighPass(data=filtedData2, freq=freq_left, order=order, fs=fs)
        psd = welch(filtedData3, fs, nfft=nfft)
        freq_end = 40  # 只关心40hz之前的PSD
        points = int((freq_end / (fs / 2)) * len(psd[0]))
        psd_data.append(psd[1][:points])
    psd_data_np = np.array([psd_data])
    res = model.predict(psd_data_np)
    print(res)
    option = np.argmax(res)
    if option == 0:
        if closed_time == 0:  # 第一次
            closed_time = time.time()
            client.send("1".encode("utf-8"))
            winsound.Beep(600, 500)
        if closed_time + delay_time < time.time():
            client.send("1".encode("utf-8"))
            closed_time = time.time()
            winsound.Beep(600, 500)

# End stream
board.stop_stream()
board.release_session()
