import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def calibration(data):
    max_x = np.max(data[:,1])
    min_x = np.min(data[:,1])
    mean_x = np.mean(data[:,1])
    std_x = np.std(data[:,1])
    conf_int = 2.576 * std_x / np.sqrt(len(data))
    print("Max:", max_x)
    print("Min:", min_x)
    print("Mean:", mean_x)
    print("Std:", std_x)
    print("Confidence interval 99%:", conf_int)

if __name__ == '__main__':
    # Read data from CSV file
    file_path = 'lab2_data/Lab 2 LA2T1 5-25 Forced.xlsx'
    sheet_name = 0

    df_calibration = pd.read_excel(file_path, header=None, sheet_name=sheet_name)

    data = df_calibration[[3,4]]
    data = data.to_numpy()
    data = calibration(data)

    data = df_calibration[[9,10]]
    data = data.to_numpy()
    data = calibration(data)