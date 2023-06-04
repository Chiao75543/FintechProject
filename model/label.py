import pandas as pd
import numpy as np

def label_stock_prices(df, window_size=15, upper_threshold=0.07, lower_threshold=-0.1):
    df['Label'] = np.nan

    for i in range(len(df)-window_size):

        if (df.iloc[i+1:i+1+window_size]['收盤價'].max() / df.iloc[i]['收盤價']) - 1 >= upper_threshold:
            df.loc[df.index[i], 'Label'] = 1  # 設定未來 15 天漲超過 7% 的標籤為 1

        elif (df.iloc[i+1:i+1+window_size]['收盤價'].min() / df.iloc[i]['收盤價']) - 1 <= lower_threshold:
            df.loc[df.index[i], 'Label'] = -1  # 設定未來 15 天跌超過 10% 的標籤為 -1

        else:
            df.loc[df.index[i], 'Label'] = 0  # 設定未來 15 天 沒有漲超過 7% 沒有跌超過 10% 的標籤為0

    return df


stock_id = "2330"   #台積電
total_features_name = 'data/{}_Total_features.csv'.format(stock_id)

total_features_df = pd.read_csv(total_features_name, index_col='date')

df_labeled = label_stock_prices(total_features_df)

label_features_name = 'data/{}_Label_features.csv'.format(stock_id)
df_labeled.to_csv(label_features_name, index=True)