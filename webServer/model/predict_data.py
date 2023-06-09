import features
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from datetime import timedelta

# stock dataset
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def get_prediction(select_data):

    data = features.total_features(select_data)

    data = data[["開盤價","收盤價","最高價","最低價","成交張數","成交筆數",
        "成交金額","成交均張","漲跌幅","累計_融資_增減(張)","融資_融資今日餘額(張)","累計_買進_自營商買進股數(自行買賣+避險)",
        "累計_三大法人賣出股數","累計_賣出_自營商賣出股數(自行買賣+避險)","累計_賣出_外陸資賣出股數(不含外資自營商)",
        "累計_賣出_投信賣出股數","累計_融資_融資買進(張)","三大法人買賣超股數","累計_借券賣出_當日賣出"]]

    scaler = MinMaxScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])

    # 創建標籤
    data['Label'] = data['收盤價'].rolling(15).apply(lambda x: 1 if list(x)[-1] / list(x)[0] - 1 > 0.07 else 0, raw=True)
    data = data.dropna()



    data_np = data.values
    lookback = 15
    data_tensor = StockDataset(data_np,lookback)
    data_loader = DataLoader(data_tensor, batch_size=32, shuffle=False)

    # model 

    model = torch.load('webServer\model\my_model.pth',map_location=torch.device('cpu'))

    model.eval()

    with torch.no_grad():
        pred_labels = []
        for inputs, labels in data_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            pred_labels.extend(preds.tolist())


    pred_length = len(pred_labels)

    # 輸出結果的dataframe
    tomorrow = datetime.now() + timedelta(days=1)

    dates = pd.date_range(start=tomorrow, periods=int(pred_length * 1.5))

    business_days = dates.to_series().map(lambda x: x.weekday() < 5)

    business_days = business_days[business_days].index[:pred_length]

    df = pd.DataFrame(pred_labels, index=business_days, columns=['Prediction'])
    df.index = df.index.strftime('%Y-%m-%d')
    
    return df[:31]

select_data = "2330"
print(get_prediction(select_data))
