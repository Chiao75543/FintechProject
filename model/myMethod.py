import pandas as pd
import numpy as np
import math

"""
確定觸及的障礙：
觸及的障礙可以是上行障礙、下行障礙或時間障礙。
根據三重障礙法的定義，
上行障礙被觸及表示價格上升到了我們的停利價格，
下行障礙被觸及表示價格下降到了我們的止損價格，
而時間障礙被觸及則表示在我們的觀察期內，
價格沒有達到我們的目標價格或止損價格。

買賣方向：
如果上行障礙被觸及，那麼我們應該買入，因為這表示價格可能會繼續上升。
反之，如果下行障礙被觸及，那麼我們應該賣出，因為這表示價格可能會繼續下降。
如果時間障礙被觸及，那麼我們選擇保持現狀。

"""


# 三重標籤法
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


# 計算特徵重要性
def featImpMDI(fit,featNames):
# feat importance based on IS mean impurity reduction
    df0={i:tree.feature_importances_ for i,tree in enumerate(fit.estimators_)}
    df0=pd.DataFrame.from_dict(df0,orient='index')
    df0.columns=featNames
    df0=df0.replace(0,np.nan) # because max_features=1
    imp=pd.concat({'mean':df0.mean(),'std':df0.std()*df0.shape[0]**-.5},axis=1)
    imp/=imp['mean'].sum()
    return imp