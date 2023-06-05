import pandas as pd
import numpy as np
import math

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



def triple_barrier(df, upper_threshold = 1.07, lower_threshold = 0.9 , window_size = 15):

    # 計算結束價格/初始價格 (收益)
    def end_price(s):
        return np.append(s[(s / s[0] > upper_threshold) | (s / s[0] < lower_threshold)], s[-1])[0]/s[0]
    
    r = np.array(range(window_size))
    
    # 計算碰到障礙的時間
    def end_time(s):
        return np.append(r[(s / s[0] > upper_threshold) | (s / s[0] < lower_threshold)], window_size-1)[0]

    p = df.rolling(window_size).apply(end_price, raw=True).shift(-window_size+1)
    t = df.rolling(window_size).apply(end_time, raw=True).shift(-window_size+1)
    t = pd.Series([t.index[int(k+i)] if not math.isnan(k+i) else np.datetime64('NaT') 
                   for i, k in enumerate(t)], index=t.index).dropna()

    signal = pd.Series(0, p.index)
    signal.loc[p > upper_threshold] = 1
    signal.loc[p < lower_threshold] = -1
    ret = pd.DataFrame({'首次接觸收益':p, '持有日期':t, '標籤':signal})

    return ret




stock_id = "2330"   #台積電
total_features_name = 'data/{}_Total_features.csv'.format(stock_id)

total_features_df = pd.read_csv(total_features_name, index_col='date')

# df_labeled = label_stock_prices(total_features_df)

# label_features_name = 'data/{}_Label_features.csv'.format(stock_id)
# df_labeled.to_csv(label_features_name, index=True)