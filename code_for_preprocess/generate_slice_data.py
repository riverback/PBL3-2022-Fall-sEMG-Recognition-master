'''
用于从原始数据中获得切片的数据
目前手机的数据(使用购买的模块)是按照微秒的时间记录的,实际时间分辨率大概为2ms,大约500Hz
目前采取200ms的窗宽,那么对原始数据而言就是100个数据点,
考虑到实际完成一个动作得到的信号宽度大约为1s,因此两段数据之间有50%的overlap

问题: 如何判断当前slice是否代表了动作信号?
    求当前slice内 最大的通道平均值 如果大于对应通道整体的平均值 则认为是动作信号

'''

import os
import numpy as np
import pandas as pd


def generate_slice(csvfile, start, N_Channels, signal_label, interval=100, overlap=0.5):

    parent_folder, file_name = os.path.split(csvfile)
    file_name, ext = os.path.splitext(file_name)

    def isaction(sliceframe, meanframe, N_Channels)->bool:
        slice_mean = sliceframe.mean()
        for i in range(1, N_Channels+1):
            if slice_mean[i] > meanframe[i]:
                return True
        return False
    names = ['Time(/μs)'] + ['Channel_{:02d}'.format(n) for n in range(N_Channels)] 
    csv_data = pd.read_csv(csvfile, names=names).iloc[start:] # 包含start, 不包含stop
    Channels_mean = csv_data.mean()
    print(Channels_mean)

    start = 0
    rows = len(csv_data)

    signal_i, nonsignal_i = 0, 0
    while start+interval < rows:
        emg_slice = csv_data.iloc[start:start+interval]
        if isaction(emg_slice, Channels_mean, N_Channels):
            save_path = os.path.join(r'processed_data\slice_data', '{:02d}'.format(signal_label), '{}-{:03d}.csv'.format(file_name, signal_i))
            emg_slice.to_csv(save_path, header=None)
            signal_i += 1
        else:
            save_path = os.path.join(r'processed_data\slice_data', '{:02d}'.format(0), '{}-{:03d}.csv'.format(file_name, nonsignal_i))
            emg_slice.to_csv(save_path, header=None)
            nonsignal_i += 1
        start = int(start + interval*(1-overlap))

if __name__ == '__main__':
    N_channels = 4
    generate_slice(csvfile=r'processed_data\raw_data\前凹2.csv', start=75, N_Channels=N_channels, signal_label=2, interval=100, overlap=0.5)

