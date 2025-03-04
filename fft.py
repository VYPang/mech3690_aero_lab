import numpy as np
import pandas as pd
import csv
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def data_preprocessing(data, conf): # data: 2D numpy array
    # time start from 0
    data[:,0] = data[:,0] - data[0,0]
    # zero drift
    mean = conf.cal_mean
    data[:,1] = data[:,1] - mean
    return data

def plot_signal(t, x, save_prefix):
    plt.plot(t, x, marker='x', markersize=1, linestyle='-', color='black', linewidth=0.7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid()
    plt.savefig(f'{save_prefix}_signal.png', dpi=300)
    plt.clf()

def fft_plot_psd(t, x, save_prefix, threshold=None):
    delta_t = (t[-1] - t[0]) / (N - 1)
    # Compute Fourier transform and extract frequency domain information
    Y = np.fft.fft(x) / (N / 2)
    Z = np.abs(Y[:N//2])
    YR = np.real(Y[:N//2])
    YI = np.imag(Y[:N//2])

    delta_f = 1 / (delta_t * N)
    f = np.arange(0, N//2) * delta_f
    f1 = np.arange(0, N) * delta_f

    # Find maximum value and corresponding frequency
    max_idx = np.argmax(Z)
    max_f = f[max_idx]
    max_Z = Z[max_idx]
    print("Maximum value of Z:", max_Z)
    print("Corresponding frequency:", max_f)

    # filter amplitude thershold
    if threshold:
        array = np.array([f, Z])
        # starting from last element, remove elements if Z < threshold, break if Z >= threshold
        for i in range(len(array[0])-1, -1, -1):
            if array[1][i] < threshold:
                array = np.delete(array, i, axis=1)
            else:
                break
        f = array[0]
        Z = array[1]

    # Plot frequency domain information
    plt.plot(f, Z, marker='x', markersize=1, linestyle='-', color='black', linewidth=0.7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig(f'{save_prefix}_fft.png', dpi=300)
    plt.clf()

    # Write frequency domain information to output file
    with open('output.txt', 'w') as file:
        for i in range(N//2):
            file.write('{:.5f} {:.5f} {:.5f}\n'.format(f1[i], YR[i], YI[i]))

if __name__ == '__main__':
    # Read data from CSV file
    file_path = 'data/lab1.xlsx'
    conf_path = 'config/config.yaml'
    save_prefix = 'ac_noise'
    num_dataPoints = 2500
    threshold = 0.00004
    # threshold = None
    sheet_name = 1

    conf = OmegaConf.load(conf_path)
    df_calibration = pd.read_excel(file_path, header=None, sheet_name=sheet_name)

    data = df_calibration[[3,4]]
    N1 = len(data)
    data = data.to_numpy()
    data = data_preprocessing(data, conf)

    if N1 > num_dataPoints:
        N = num_dataPoints
    else:
        N = N1

    # Extract time and signal values
    t = data[:N, 0] 
    x = data[:N, 1]

    # Plot signal
    plot_signal(t, x, save_prefix)
    fft_plot_psd(t, x, save_prefix, threshold)