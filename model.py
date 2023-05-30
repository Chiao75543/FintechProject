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
        
    # 定義交叉驗證參數
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True)

    fold_scores = []
    best_score = 0
    # 執行交叉驗證
    for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
        # 根據fold拆分成訓練和驗證數據
        train_data_fold = torch.utils.data.Subset(train_data, train_index)
        val_data_fold = torch.utils.data.Subset(train_data, val_index)

        # 創建訓練和驗證的DataLoader
        train_loader_fold = DataLoader(train_data_fold, batch_size=32, shuffle=False)
        val_loader_fold = DataLoader(val_data_fold, batch_size=32, shuffle=False)

        """
        LSTM參數
        """
        # 初始化模型，定義損失函數和優化器
        model = LSTM(input_size=train_set_np.shape[1]-2, hidden_size=50, num_layers=2).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 訓練模型
        num_epochs = 500
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(train_loader_fold):
                inputs = inputs.to(device).float()
                labels = labels.to(device).float().view(-1)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print loss
                if (i+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader_fold)}], Loss: {loss.item():.4f}')
        
        #評估模型
        model.eval()
        with torch.no_grad():
            true_labels = []
            pred_labels = []
            for inputs, labels in val_loader_fold:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                true_labels.extend(labels.tolist())
                pred_labels.extend(preds.tolist())
        # 計算準確率
        accuracy = accuracy_score(true_labels, pred_labels)
        # 將當前fold的結果存放到列表中
        fold_scores.append(accuracy)

        # 判斷當前fold的準確率是否為最佳準確率
        if accuracy > best_score:
            best_model = model
            best_score = accuracy
    
    # print 每個fold的結果
    for fold, score in enumerate(fold_scores):
        print(f'Fold {fold+1} Validation Score: {score}')
    
    best_model.eval()
    with torch.no_grad():
        true_labels = []
        pred_labels = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = best_model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            true_labels.extend(labels.tolist())
            pred_labels.extend(preds.tolist())



main()