import time
import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowFunctions
import matplotlib.pyplot as plt
# 参数定义 Cyton板的id=0，我机器上的串口号为COM3
params = BrainFlowInputParams()
#params.serial_port = "COM3"
board_id = -1  # 合成版id=-1，cyton板id=0
board = BoardShim(board_id, params)

# Start stream
board.prepare_session()
board.start_stream()

print(board.get_sampling_rate(board_id))
sr = board.get_sampling_rate(board_id)

time.sleep(2)
data = board.get_board_data()
eeg_channels = board.get_eeg_channels(board_id)  # 只关心eeg channel
df = pd.DataFrame(np.transpose(data))
df[eeg_channels].plot(subplots=True)
plt.show()
DataFilter.write_file(data,"data.csv","w")
for count, channel in enumerate(eeg_channels):
    if count!=10:
        fft_data = DataFilter.perform_fft(data[channel][:128],WindowFunctions.BLACKMAN_HARRIS)
        print("fft_data.shape:",fft_data.shape)
        print(fft_data)
        plt.plot(fft_data)
        plt.show()
        psd = DataFilter.get_psd(data[channel][:128], sr, WindowFunctions.BLACKMAN_HARRIS)
        plt.plot(psd[1],psd[0]/np.max(psd[0]))
        print(len(psd[0]))
        plt.show()

# End stream
board.stop_stream()
board.release_session()
