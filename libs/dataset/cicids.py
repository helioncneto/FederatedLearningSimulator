import torch
import pandas as pd
import numpy as np
import time
import os
from torch.utils.data import Dataset
from sklearn import preprocessing


class CICIDS2017Dataset (Dataset):
    def __init__(self, data, normalize='minmax', debug=False):
        self.debug = debug
        self.data = data
        self.class_to_idx = {0: "Benign", 1: "Attack"}

        '''self.data = pd.read_csv(path)
        self.data['Flow Bytes/s'] = self.data['Flow Bytes/s'].astype(float)
        self.data[' Flow Packets/s'] = self.data[' Flow Packets/s'].astype(float)
        if dropna:
            self.data = self.data.replace(np.inf, np.NaN)
            self.data.dropna(inplace=True)

        else:
            # TODO: Implement a treatment for the NAs
            pass'''

        self.y = self.data[' Label']
        self.x = self.data.drop(' Label', axis=1)
        if normalize == 'minmax':
            x = self.x.values
            col = self.x.columns
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            self.x = pd.DataFrame(x_scaled, columns=col)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(f'X: {self.x.iloc[idx].to_numpy()} \t Y:{self.y.iloc[idx]}')
        x_tensor = torch.from_numpy(self.x.iloc[idx].to_numpy())
        y_tensor = torch.tensor(self.y.iloc[idx])
        if self.debug:
            #print(len(x_tensor))
            print(y_tensor)
        return x_tensor, y_tensor


def create_cic_ids_file():
    initial_time = time.time()
    print('Creating CICIDS2017 file...')
    files_path = os.path.join('data', 'CICIDS2017')
    cic_ids_path = os.path.join(files_path, 'cicids2017.csv')
    files = os.listdir(files_path)
    try:
        cic_file = open(cic_ids_path, 'x')
    except FileExistsError as e:
        print(e)
        print(f"The file {cic_ids_path} already exist")
    for n_file, file in enumerate(files):
        cur_file = open(os.path.join(files_path, file), 'r')
        for n, line in enumerate(cur_file.readlines()):
            if n != 0 or n_file == 0:
                if n == 0 and n_file == 0:
                    cic_file.write(line)
                else:
                    line_list = line.split(',')
                    # print(line_list[-1])
                    if line_list[-1][-1:] == '\n':
                        if line_list[-1][:-1] == 'BENIGN':
                            line_list[-1] = '0'
                        else:
                            line_list[-1] = '1'
                    else:
                        if line_list[-1] == 'BENIGN':
                            line_list[-1] = '0'
                        else:
                            line_list[-1] = '1'
                    last_line = len(line_list) - 1
                    line_list = [float(i) if n != last_line else int(i) for n, i in enumerate(line_list)]
                    cic_file.write(str(line_list)[1:-1])
                cic_file.write('\n')
        cur_file.close()
    cic_file.close()
    end_time = time.time()
    print(f'file created in {round(end_time - initial_time, 2)} seconds')
