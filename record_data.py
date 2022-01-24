import time
import winsound

import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
import matplotlib.pyplot as plt
time.sleep(2)
for i in range(5):
    # 参数定义 Cyton板的id=0，我机器上的串口号为COM3
    params = BrainFlowInputParams()
    params.serial_port = "COM3"
    board_id = BoardIds.CYTON_BOARD  # 合成版id=-1，cyton板id=0
    board = BoardShim(board_id, params)
    #time.sleep(1)
    winsound.Beep(600,1000)
    #time.sleep(1)

    # Start stream
    board.prepare_session()
    board.start_stream()

    # 收集t秒的数据
    t = 10
    mode = "SSVEP_10_2"
    time.sleep(t)
    data = board.get_board_data()


    # End stream
    board.stop_stream()
    board.release_session()

    winsound.Beep(500,600)

    # Save data
    DataFilter.write_file(data,f'data/{mode}/{time.strftime("%Y-%m-%d-%H_%M_%S")}.csv','w')