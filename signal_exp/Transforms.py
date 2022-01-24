import argparse
import time
import brainflow
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions
import matplotlib.pyplot as plt


def main():
    BoardShim.enable_dev_board_logger()

    # use synthetic board for demo
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(3)
    data = board.get_current_board_data(DataFilter.get_nearest_power_of_two(sampling_rate)*2)
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(board_id)


    # demo for transforms
    for count, channel in enumerate(eeg_channels):
        print('Original data for channel %d:' % channel)
        plt.plot(data[channel])
        plt.title("Original Channel "+ str(channel))
        plt.show()
        #print(data[channel])

        # demo for wavelet transforms
        # wavelet_coeffs format is[A(J) D(J) D(J-1) ..... D(1)] where J is decomposition level, A - app coeffs, D - detailed coeffs
        # lengths array stores lengths for each block
        wavelet_coeffs, lengths = DataFilter.perform_wavelet_transform(data[channel], 'db5', 3)
        app_coefs = wavelet_coeffs[0: lengths[0]]
        detailed_coeffs_first_block = wavelet_coeffs[lengths[0]: lengths[1]]
        # you can do smth with wavelet coeffs here, for example denoising works via thresholds
        # for wavelets coefficients
        restored_data = DataFilter.perform_inverse_wavelet_transform((wavelet_coeffs, lengths), data[channel].shape[0],
                                                                     'db5', 3)
        #print('Restored data after wavelet transform for channel %d:' % channel)
        #print(restored_data)
        # plt.plot(restored_data)
        # plt.title("Restored wavelet Channel "+ str(channel))
        # plt.show()
        # demo for fft, len of data must be a power of 2
        fft_data = DataFilter.perform_fft(data[channel], WindowFunctions.NO_WINDOW.value)
        plt.plot(fft_data)
        plt.title(" FFT data " + str(channel))
        plt.show()
        # len of fft_data is N / 2 + 1
        restored_fft_data = DataFilter.perform_ifft(fft_data)
        #print('Restored data after fft for channel %d:' % channel)
        #print(restored_fft_data)
        # plt.plot(restored_fft_data)
        # plt.title("Restored fft Channel " + str(channel))
        # plt.show()
        psd = DataFilter.get_psd_welch(data[channel][:512], 512, 0, sampling_rate, WindowFunctions.BLACKMAN_HARRIS)
        print(psd)

        plt.plot(psd[1][:40],psd[0][:40])
        plt.show()

if __name__ == "__main__":
    main()