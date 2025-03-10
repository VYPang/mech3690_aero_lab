import numpy as np
import pandas as pd
import csv
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import plotly.express as px
from scipy.signal import find_peaks

def plotly_scatter(t, x):
    fig = px.scatter(x=t, y=x)
    fig.show()

def data_preprocessing(data, conf, p=True): # data: 2D numpy array
    # time start from 0
    data[:,0] = data[:,0] - data[0,0]
    # zero drift
    mean = conf.cal_mean
    data[:,1] = data[:,1] - mean
    # mean, std, confidence interval 99%
    mean = np.mean(data[:,1])
    std = np.std(data[:,1])
    conf_int = 2.576 * std / np.sqrt(len(data))
    if p:
        print(f"Before normalization, Mean: {mean} ± {conf_int} (cl99%); std: {std}")
    return data

def fft(t, x, save_prefix='fft', plot=True):
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

    if plot:
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
    return actual_freq

def plot_signal(t, x, prefix='signal'):
    plt.plot(t, x, marker='x', markersize=1, linestyle='-', color='black', linewidth=0.7)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normalized Amplitude $x/x_0$')
    plt.grid()
    plt.savefig(f'{prefix}_signal.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_max_amplitude(max_record):
    plt.plot(max_record[:,0], max_record[:,1], marker='x', markersize=5, linestyle='-', color='black', linewidth=0.7)
    plt.ticklabel_format(style='sci', axis='x')
    plt.ticklabel_format(style='sci', axis='y')
    plt.xlabel(r'Normalized Frequency $\omega/\omega_n$')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig('max_amplitude_plot.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()

    max_idx = np.argmax(max_record[:,1])
    max_freq = max_record[max_idx, 0]
    max_amp = max_record[max_idx, 1]
    return max_freq, max_amp

if __name__ == '__main__':
    block_d_vol = 25.65 * 25.65 * 45.15 # mm^3
    block_a_vol = 10.00 * 23.40 * 45.00 # mm^3
    block_d_mass = 2700 * (block_d_vol / 0.001**3) # kg
    block_a_mass = 2700 * (block_a_vol / 0.001**3) # kg

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
    # plotly_scatter(d_free[:,0], d_free[:,1]) # for better visualization to compute zeta manually
    max_idx = np.argsort(d_free[:,1])[-9] # find second maximum value corresponding index (寫死)
    d_free = d_free[max_idx:]
    d_free_t = d_free[:,0]
    d_free_x = d_free[:,1]
    d_free_x = d_free_x / np.max(d_free_x) # normalization
    plot_signal(d_free_t, d_free_x, 'D_free')
    # plotly_scatter(d_free[:,0], d_free[:,1]) # for better visualization to compute zeta manually
    d_actual_freq = damped_period_estimation(d_free_t, d_free_x, 'D_free')
    print(f'Mass m: {block_d_mass} kg')
    
    print('\n')

    # free vibration: Block A
    print('Block A free vibration analysis')
    a_free = free_all['A Free'][[3,4]]
    a_free = a_free.to_numpy()
    a_free = data_preprocessing(a_free, conf.ch_2)
    # plotly_scatter(a_free[:,0], a_free[:,1]) # for better visualization to compute zeta manually
    max_idx = np.argsort(a_free[:,1])[-5] # find second maximum value corresponding index (寫死)
    a_free = a_free[max_idx:]
    a_free_t = a_free[:,0]
    a_free_x = a_free[:,1]
    a_free_x = a_free_x / np.max(a_free_x) # normalization
    plot_signal(a_free_t, a_free_x, 'A_free')
    # plotly_scatter(a_free[:,0], a_free[:,1]) # for better visualization to compute zeta manually
    a_actual_freq = damped_period_estimation(a_free_t, a_free_x, 'A_free')
    print(f'Mass m: {block_a_mass} kg')

    print('\n')

    # forced vibration: Block D
    print('Block D forced vibration analysis')
    max_record = []
    for name, df in coarse_all.items():
        if name == 'calibration':
            continue
        freq = float(name[:-9])
        ch_2 = df[[9, 10]]
        ch_2 = ch_2.to_numpy()
        ch_2 = data_preprocessing(ch_2, conf.ch_2, p=False)
        ch_2_t = ch_2[:,0]
        ch_2_x = ch_2[:,1] / 10 # /10 because of experiment setup mistake
        ch_2_x = np.abs(ch_2_x)
        max = np.max(ch_2_x)
        max_record.append([freq, max])
    for name, df in fine_all.items():
        if name == 'calibration':
            continue
        freq = float(name)
        ch_2 = df[[9, 10]]
        ch_2 = ch_2.to_numpy()
        ch_2 = data_preprocessing(ch_2, conf.ch_2, p=False)
        ch_2_t = ch_2[:,0]
        ch_2_x = ch_2[:,1]
        ch_2_x = np.abs(ch_2_x)
        max = np.max(ch_2_x)
        max_record.append([freq, max])
    max_record = np.array(max_record)
    max_record = max_record[max_record[:,0].argsort()]
    # look for resonnant frequency
    max_idx = np.argmax(max_record[:,1])
    max_freq = max_record[max_idx, 0]
    max_amp = max_record[max_idx, 1]
    print(f'Maximum amplitude: {max_amp} at {max_freq} Hz')
    # normalization with natrual frequency
    max_record[:, 0] = max_record[:, 0] / max_freq # frequency corresponding to the maximum amplitude is natrual frequency
    max_freq, max_amp = plot_max_amplitude(max_record)
    amp_thresold = 0.707 * max_amp
    # find intersection point
    # split with max_freq
    max_idx = np.argmax(max_record[:,1])
    left_curve = max_record[:max_idx]
    right_curve = max_record[max_idx:]
    # look for 1 point just lower than threshold & 1 point just higher than threshold
    for i in range(len(left_curve)):
        amplitude = left_curve[i][1]
        if amplitude > amp_thresold:
            right_amp = amplitude
            right_freq = left_curve[i][0]
            left_amp = left_curve[i-1][1]
            left_freq = left_curve[i-1][0]
            break
    # interpolation
    slope = (right_amp - left_amp) / (right_freq - left_freq)
    left_freq_intersection = left_freq + (amp_thresold - left_amp) / slope
    # look for 1 point just higher than threshold & 1 point just lower than threshold
    for i in range(len(right_curve)):
        amplitude = right_curve[i][1]
        if amplitude < amp_thresold:
            right_amp = amplitude
            right_freq = right_curve[i][0]
            left_amp = right_curve[i-1][1]
            left_freq = right_curve[i-1][0]
            break
    # interpolation
    slope = (right_amp - left_amp) / (right_freq - left_freq)
    right_freq_intersection = left_freq + (amp_thresold - left_amp) / slope
    forced_damping_ratio = (right_freq_intersection - left_freq_intersection) / 2
    print(f'Damping ratio of D under forced: {forced_damping_ratio}')

    plt.plot(max_record[:,0], max_record[:,1], marker='x', markersize=5, linestyle='-', color='black', linewidth=0.7)
    # plot vertical lines on intersection points
    plt.axvline(x=left_freq_intersection, color='red', linestyle='--', linewidth=0.7, label=r'$\omega_1/\omega_n$')
    plt.axvline(x=right_freq_intersection, color='blue', linestyle='--', linewidth=0.7, label=r'$\omega_2/\omega_n$')
    plt.ticklabel_format(style='sci', axis='x')
    plt.ticklabel_format(style='sci', axis='y')
    plt.legend()
    plt.xlabel(r'Normalized Frequency $\omega/\omega_n$')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig('max_amplitude_plot_intersect.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()