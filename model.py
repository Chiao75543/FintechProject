import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


"""
定義資料集
"""
class StockDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback

    def __getitem__(self, index):
        x = torch.Tensor(self.data[index:index+self.lookback, 1:-1])
        y = torch.Tensor([self.data[index+self.lookback-1, -1]])
        return x, y

    def __len__(self):
        return self.data.shape[0] - self.lookback



def main():

    """
    要改的話要改資料前處理的部分
    """
    """
    資料前處理start
    """

    """
    資料前處理end
    """


    # 創建 DataLoader
    """
    根據資料去做修改
    例子：
    lookback = 15
    train_data = StockDataset('這個是原來的值:train_set_np', lookback)
    test_data = StockDataset('這個是原來的值:test_set_np', lookback)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    """
    

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    


    """
    定義我們的模型
    """
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out.view(-1)



main()