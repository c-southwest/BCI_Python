import time
import pandas as pd
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

import matplotlib.pyplot as plt

# 参数定义 Cyton板的id=0，我机器上的串口号为COM3
params = BrainFlowInputParams()
params.serial_port = "COM3"
board_id = BoardIds.SYNTHETIC_BOARD  # 合成版id=-1，cyton板id=0
board = BoardShim(board_id, params)

# Start stream
board.prepare_session()
board.start_stream()

board.insert_marker(1)  # marker所在channel通过board.get_marker_channel(board_id)得到
time.sleep(1)
data = board.get_board_data()
eeg_channels = board.get_eeg_channels(board_id)  # 只关心eeg channel
df = pd.DataFrame(np.transpose(data))
plt.figure()
df[eeg_channels].plot(subplots=True)
plt.show()
plt.savefig

# plot eeg data
for i in eeg_channels:
    plt.subplot(16, 1, i)
    plt.plot(data[i])
plt.show()
# plot marker data

maker_channel = board.get_marker_channel(board_id)
plt.plot(data[maker_channel])
plt.title("Marker plot")
plt.show()

# End stream
board.stop_stream()
board.release_session()

DataFilter.write_file(data, "data.csv", 'w')
new = DataFilter.read_file("data.csv")
