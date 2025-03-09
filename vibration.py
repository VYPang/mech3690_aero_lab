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
    # mean, std, confidence interval 99%
    mean = np.mean(data[:,1])
    std = np.std(data[:,1])
    conf_int = 2.576 * std / np.sqrt(len(data))
    print(f"Mean: {mean} ± {conf_int} (cl99%); std: {std}")
    return data

def fft(t, x, save_prefix='fft'):
    N = len(t)
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

    plt.plot(f, Z, marker='x', markersize=1, linestyle='-', color='black', linewidth=0.7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig(f'{save_prefix}_fft.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()
    return max_f, max_Z # frequency, amplitude

def damped_period_estimation(t, x, prefix):
    f, _ = fft(t, x, prefix)
    damped_period = 1 / f # T_d
    print("Damped period T_d:", damped_period)
    actual_freq = 2 * np.pi / damped_period
    print("Actual frequency \omega_d:", actual_freq)

def plot_signal(t, x, prefix='signal'):
    plt.plot(t, x, marker='x', markersize=1, linestyle='-', color='black', linewidth=0.7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid()
    plt.savefig(f'{prefix}_signal.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()

if __name__ == '__main__':
    block_d_mass = 0.5 # kg
    block_a_mass = 0.5 # kg

    # Read data from CSV file
    coarse_path = 'lab2_data/Lab 2 LA2T1 5-25 Forced.xlsx'
    fine_path = 'lab2_data/Lab 2 LA2T1 17.6-18.4 Forced.xlsx'
    free_path = 'lab2_data/Lab 2 LA2T1 free.xlsx'
    conf_path = 'lab2_data/config.yaml'

    coarse_all = pd.read_excel(coarse_path, header=None, sheet_name=None)
    fine_all = pd.read_excel(fine_path, header=None, sheet_name=None)
    free_all = pd.read_excel(free_path, header=None, sheet_name=None)
    conf = OmegaConf.load(conf_path)

    # free vibration: Block D
    print('Block D free vibration analysis')
    d_free = free_all['D Free'][[3,4]]
    d_free = d_free.to_numpy()
    d_free = data_preprocessing(d_free, conf.ch_2)
    d_free_t = d_free[:,0]
    d_free_x = d_free[:,1]
    plot_signal(d_free_t, d_free_x, 'D_free')
    start_time_cut = 0.25 # 寫死 : according to signal plot
    d_free = d_free[d_free[:,0] > start_time_cut]
    d_free_t = d_free[:,0]
    d_free_x = d_free[:,1]
    damped_period_estimation(d_free_t, d_free_x, 'D_free')
    print(f'Mass m: {block_d_mass} kg')
    
    print('\n')

    # free vibration: Block A
    print('Block A free vibration analysis')
    a_free = free_all['A Free'][[3,4]]
    a_free = a_free.to_numpy()
    a_free = data_preprocessing(a_free, conf.ch_2)
    a_free_t = a_free[:,0]
    a_free_x = a_free[:,1]
    plot_signal(a_free_t, a_free_x, 'A_free')
    start_time_cut = 0.26 # 寫死 : according to signal plot
    a_free = a_free[a_free[:,0] > start_time_cut]
    a_free_t = a_free[:,0]
    a_free_x = a_free[:,1]
    damped_period_estimation(a_free_t, a_free_x, 'A_free')
    print(f'Mass m: {block_a_mass} kg')