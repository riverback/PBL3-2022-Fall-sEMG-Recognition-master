'''
√ 1. (手工)去除数据头部---之前的数据 去除数据尾部的空行
√ 2. 去除数据刚开始可能出现的超大值数据(超过5位数)
√ 3. 将txt数据保存为csv数据
  4. 对数据进行初步的可视化:
     √ a. 时域分布
       b. 频域分布
'''

import os
import pandas as pd
import numpy as np
import scipy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def txt2csv(file, output_folder):
    '''把txt的数据转换为csv方便处理'''
    parent_folder, file_name = os.path.split(file)
    file_name, ext = os.path.splitext(file_name)
    output_path = os.path.join(output_folder, file_name+'.csv')
    with open(file, 'r') as f:
        data = f.readlines()
        data = [d.replace(' ', ',') for d in data]
    with open(output_path, 'w') as f:
        f.writelines(data)

def data_wrangling(csvfile, N_channel):
    '''使用pandas进行数据清洗'''
    parent_folder, file_name = os.path.split(csvfile)
    file_name, ext = os.path.splitext(file_name)
    names = ['Time(/μs)'] + ['Channel_{:02d}'.format(i) for i in range(N_channel)]
    df = pd.read_csv(csvfile, names=names)
    print(df.info())

    # 去除数据开始的某些异常值 (过大值)
    Channels = df.columns.values[1:]
    for channel in Channels:
        df.drop(df[df[channel]>5000].index, inplace=True) # 删除某通道某时刻值异常的行
    #df.plot(x='Time(/μs)', y=Channels)
    df.plot(y=Channels)
    plt.savefig('{}/{}.png'.format(parent_folder, file_name))
    plt.close()

if __name__ == '__main__':
    # txt2csv
    
    csv_folder = r'csv_data'
    for file in os.listdir(r'ywh_txt_data'):
        txt2csv(os.path.join(r'ywh_txt_data', file), csv_folder)
    
    for file in os.listdir(csv_folder):
        if os.path.splitext(file)[-1] == '.csv':
            data_wrangling(os.path.join(csv_folder, file), 4)

