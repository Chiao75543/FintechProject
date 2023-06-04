import requests
import pandas as pd
import numpy as np

stock_id = "2330"   #台積電
start_date = "2018-01-01"
end_date = "2023-01-31"

def TaiwanStockPrice(): #FinMind_台灣股價資料表
    url = "https://api.finmindtrade.com/api/v4/data"  #FinMind v4 api
    parameter = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": "", # 參考登入，獲取金鑰
    }
    resp = requests.get(url, params=parameter)
    data = resp.json()
    df1 = pd.DataFrame(data["data"])
    df1 = df1.set_index('date')
    df1 = df1.rename(columns={'Trading_Volume':'成交量(成交股數)',
                              'Trading_money':'成交金額',
                              'open':'開盤價',
                              'max':'最高價',
                              'min':'最低價',
                              'close':'收盤價',
                              'spread':'漲跌幅',
                              'Trading_turnover':'成交筆數'})
    
    df1['成交張數'] = df1['成交量(成交股數)'] // 1000
    df1['股價'] = df1['收盤價']
    df1['成交均張'] = round(df1['成交張數'] / df1['成交筆數'], 2) #取到小數第二位

    #篩選後的價格
    new_df1 = df1.loc[: ,['股價','開盤價','收盤價','最高價','最低價','成交張數','成交筆數','成交金額','成交均張','漲跌幅']]

    # print(new_df1)
    # print(new_df1.isnull().sum()) #查看缺失值
    # new_df1.to_csv("feature01.csv", index=True)

    return new_df1


def TaiwanStockInstitutionalInvestorsBuySell(): #FinMind_個股三大法人買賣表
    url = "https://api.finmindtrade.com/api/v4/data" #FinMind v4 api
    parameter = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": "", # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    df1 = pd.DataFrame(data['data'])
    df1 = df1.set_index('date')
    new_df1 = df1.pivot_table(index=['date', 'stock_id'], columns='name', values=['buy', 'sell'])
    new_df1.columns = [f"{col[1]}_{col[0]}" for col in new_df1.columns]
    new_df1 = new_df1.reset_index(drop=False)
    new_df1 = new_df1.set_index('date')
    new_df1 = new_df1.rename(columns={'Dealer_Hedging_buy':'買進_自營商買進股數(避險)',
                                      'Dealer_self_buy':'買進_自營商買進股數(自行買賣)',
                                      'Foreign_Dealer_Self_buy':'買進_外資自營商買進股數',
                                      'Foreign_Investor_buy':'買進_外陸資買進股數(不含外資自營商)',
                                      'Investment_Trust_buy':'買進_投信買進股數',
                                      'Dealer_Hedging_sell':'賣出_自營商賣出股數(避險)',
                                      'Dealer_self_sell':'賣出_自營商賣出股數(自行買賣)',
                                      'Foreign_Dealer_Self_sell':'賣出_外資自營商賣出股數',
                                      'Foreign_Investor_sell':'賣出_外陸資賣出股數(不含外資自營商)',
                                      'Investment_Trust_sell':'賣出_投信賣出股數'})
    
    new_df1['買進_自營商買進股數(自行買賣+避險)'] = new_df1['買進_自營商買進股數(自行買賣)'] + new_df1['買進_自營商買進股數(避險)']
    new_df1['賣出_自營商賣出股數(自行買賣+避險)'] = new_df1['賣出_自營商賣出股數(自行買賣)'] + new_df1['賣出_自營商賣出股數(避險)']

    new_df1['外陸資買賣超股數(不含外資自營商)'] = new_df1['買進_外陸資買進股數(不含外資自營商)'] - new_df1['賣出_外陸資賣出股數(不含外資自營商)']
    new_df1['投信買賣超股數'] = new_df1['買進_投信買進股數'] - new_df1['賣出_投信賣出股數']
    new_df1['自營商買賣超股數'] = new_df1['買進_自營商買進股數(自行買賣+避險)'] - new_df1['賣出_自營商賣出股數(自行買賣+避險)']

    new_df1['三大法人買進股數'] = new_df1['買進_外陸資買進股數(不含外資自營商)'] + new_df1['買進_投信買進股數'] + new_df1['買進_自營商買進股數(自行買賣+避險)']
    new_df1['三大法人賣出股數'] = new_df1['賣出_外陸資賣出股數(不含外資自營商)'] + new_df1['賣出_投信賣出股數'] + new_df1['賣出_自營商賣出股數(自行買賣+避險)']
    new_df1['三大法人買賣超股數'] = new_df1['自營商買賣超股數'] + new_df1['外陸資買賣超股數(不含外資自營商)'] + new_df1['投信買賣超股數']

    new_df1['累計_買進_外陸資買進股數(不含外資自營商)'] = new_df1['買進_外陸資買進股數(不含外資自營商)'].cumsum()
    new_df1['累計_賣出_外陸資賣出股數(不含外資自營商)'] = new_df1['賣出_外陸資賣出股數(不含外資自營商)'].cumsum()
    new_df1['累計_外陸資買賣超股數(不含外資自營商)'] = new_df1['外陸資買賣超股數(不含外資自營商)'].cumsum()

    new_df1['累計_買進_投信買進股數'] = new_df1['買進_投信買進股數'].cumsum()
    new_df1['累計_賣出_投信賣出股數'] = new_df1['賣出_投信賣出股數'].cumsum()
    new_df1['累計_投信買賣超股數'] = new_df1['投信買賣超股數'].cumsum()

    new_df1['累計_買進_自營商買進股數(自行買賣+避險)'] = new_df1['買進_自營商買進股數(自行買賣+避險)'].cumsum()
    new_df1['累計_賣出_自營商賣出股數(自行買賣+避險)'] = new_df1['賣出_自營商賣出股數(自行買賣+避險)'].cumsum()
    new_df1['累計_自營商買賣超股數'] = new_df1['自營商買賣超股數'].cumsum()

    new_df1['累計_三大法人買進股數'] = new_df1['三大法人買進股數'].cumsum()
    new_df1['累計_三大法人賣出股數'] = new_df1['三大法人賣出股數'].cumsum()
    new_df1['累計_三大法人買賣超股數'] = new_df1['三大法人買賣超股數'].cumsum()

    #篩選後的三大法人
    df2 = new_df1.loc[:, ['買進_外陸資買進股數(不含外資自營商)','賣出_外陸資賣出股數(不含外資自營商)','外陸資買賣超股數(不含外資自營商)',\
                          '買進_投信買進股數','賣出_投信賣出股數','投信買賣超股數',\
                          '買進_自營商買進股數(自行買賣+避險)','賣出_自營商賣出股數(自行買賣+避險)','自營商買賣超股數',\
                          '三大法人買進股數','三大法人賣出股數','三大法人買賣超股數',\
                          '累計_買進_外陸資買進股數(不含外資自營商)','累計_賣出_外陸資賣出股數(不含外資自營商)','累計_外陸資買賣超股數(不含外資自營商)',\
                          '累計_買進_投信買進股數','累計_賣出_投信賣出股數','累計_投信買賣超股數',\
                          '累計_買進_自營商買進股數(自行買賣+避險)','累計_賣出_自營商賣出股數(自行買賣+避險)','累計_自營商買賣超股數',\
                          '累計_三大法人買進股數','累計_三大法人賣出股數','累計_三大法人買賣超股數']]

    # print(df2)
    # print(df2.isnull().sum()) #查看缺失值
    # df2.to_csv("feature02.csv", index=True)

    return df2


def TaiwanStockMarginPurchaseShortSale(): #FinMind_個股融資融劵表  ＃融資融券
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": "", # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    df1 = pd.DataFrame(data['data'])
    df1 = df1.set_index('date')
    df1 = df1.rename(columns={'MarginPurchaseBuy':'融資_融資買進',
                              'MarginPurchaseCashRepayment':'融資_融資現金償還',
                              'MarginPurchaseLimit':'融資_融資限額',
                              'MarginPurchaseSell':'融資_融資賣出',
                              'MarginPurchaseTodayBalance':'融資_融資今日餘額',
                              'MarginPurchaseYesterdayBalance':'融資_融資昨日餘額',
                              'Note':'註記',
                              'OffsetLoanAndShort':'資券互抵',
                              'ShortSaleBuy':'融券_融券買進',
                              'ShortSaleCashRepayment':'融券_融券償還',
                              'ShortSaleLimit':'融券_融券限額',
                              'ShortSaleSell':'融券_融券賣出',
                              'ShortSaleTodayBalance':'融券_融券今日餘額',
                              'ShortSaleYesterdayBalance':'融券_融券昨日餘額'})
    
    #單位：張
    new_df1 = df1.drop('註記', axis=1, inplace=False) #刪除df['註記']

    df2 = new_df1.copy()
    for col in df2.columns[1:]: #從第二列開始
        df２.rename(columns={col: col + '(張)'}, inplace=True)  # 在列名後面加上(張)

    df2['融資_增減(張)'] = df2['融資_融資買進(張)'] - df2['融資_融資賣出(張)'] - df2['融資_融資現金償還(張)']
    df2['融券_增減(張)'] = df2['融券_融券賣出(張)'] - df2['融券_融券買進(張)'] - df2['融券_融券償還(張)']

    df2['累計_融資_融資買進(張)'] = df2['融資_融資買進(張)'].cumsum() #累計
    df2['累計_融資_融資賣出(張)'] = df2['融資_融資賣出(張)'].cumsum()
    df2['累計_融資_增減(張)'] = df2['融資_增減(張)'].cumsum()

    df2['累計_融券_融券買進(張)'] = df2['融券_融券買進(張)'].cumsum()
    df2['累計_融券_融券賣出(張)'] = df2['融券_融券賣出(張)'].cumsum()
    df2['累計_融券_增減(張)'] = df2['融券_增減(張)'].cumsum()


    # 篩選後的融資融券
    new_df2 = df2.loc[:,['融資_融資買進(張)','融資_融資賣出(張)','融資_融資現金償還(張)','融資_增減(張)','融資_融資今日餘額(張)',\
                         '融券_融券買進(張)','融券_融券賣出(張)','融券_融券償還(張)','融券_增減(張)','融券_融券今日餘額(張)','資券互抵(張)',\
                         '累計_融資_融資買進(張)','累計_融資_融資賣出(張)','累計_融資_增減(張)',\
                         '累計_融券_融券買進(張)','累計_融券_融券賣出(張)','累計_融券_增減(張)']]
    
    #print(new_df2)
    #print(new_df1)
    #print(df2)

    DailyShortSaleBalances_df = TaiwanDailyShortSaleBalances() #借券

    #合併 融資融券 與 借券
    merged_df = new_df2.merge(DailyShortSaleBalances_df, left_index=True, right_index=True, how='outer')

    #print(merged_df)
    #print(merged_df.isnull().sum()) #查看缺失值
    #merged_df.to_csv("融資融券借券_feature.csv", index=True)

    return merged_df


def TaiwanDailyShortSaleBalances(): #FinMind_融券借券賣出表  #借券
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanDailyShortSaleBalances",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": "", # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    df1 = pd.DataFrame(data['data'])
    df1 = df1.set_index('date')
    df1 = df1.rename(columns={'MarginShortSalesPreviousDayBalance':'融券_前日餘額',
                              'MarginShortSalesShortSales':'融券_賣出',
                              'MarginShortSalesShortCovering':'融券_買進',
                              'MarginShortSalesStockRedemption':'融券_現券',
                              'MarginShortSalesCurrentDayBalance':'融券_今日餘額',
                              'MarginShortSalesQuota':'融券_限額',  #數據好像有點奇怪
                              'SBLShortSalesPreviousDayBalance':'借券賣出_前日餘額',
                              'SBLShortSalesShortSales':'借券賣出_當日賣出',
                              'SBLShortSalesReturns':'借券賣出_當日還券',
                              'SBLShortSalesAdjustments':'借券賣出_當日調整',
                              'SBLShortSalesCurrentDayBalance':'借券賣出_當日餘額',
                              'SBLShortSalesQuota':'借券賣出_次一營業日可限額',
                              'SBLShortSalesShortCovering':'借券賣出_?'}) #不知道是哪一個數據

    #print(df1)
    #df1.to_csv("T_tmp_feature.csv", index=True)

    df1['借券賣出_增減'] = df1['借券賣出_當日賣出'] - df1['借券賣出_當日還券']

    df1['累計_借券賣出_當日賣出'] = df1['借券賣出_當日賣出'].cumsum() #累加
    df1['累計_借券賣出_當日還券'] = df1['借券賣出_當日還券'].cumsum()
    df1['累計_借券賣出_增減'] = df1['借券賣出_增減'].cumsum()

    #篩選後的借券
    new_df1 = df1.loc[:,['借券賣出_當日賣出','借券賣出_當日還券','借券賣出_當日調整',\
                         '借券賣出_增減','借券賣出_當日餘額','借券賣出_次一營業日可限額',\
                         '累計_借券賣出_當日賣出','累計_借券賣出_當日還券','累計_借券賣出_增減']]
    #print(new_df1)
    #new_df1.to_csv("T_tmp_feature.csv", index=True)

    return new_df1


def TaiwanStockHoldingSharesPer(): #FinMind_股東持股分級表
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanStockHoldingSharesPer",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": "", # 參考登入，獲取金鑰
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    df1 = pd.DataFrame(data['data'])
    new_df1 = pd.pivot_table(df1, index='date', columns='HoldingSharesLevel', values='percent')

    df2 = new_df1.loc[:,['1-999','1,000-5,000','5,001-10,000','10,001-15,000','15,001-20,000',\
                         '20,001-30,000','30,001-40,000','40,001-50,000','50,001-100,000','100,001-200,000',\
                         '200,001-400,000','400,001-600,000','600,001-800,000','800,001-1,000,000','more than 1,000,001']]
    
    df2.columns = [col + '(股)' for col in df2.columns]
    df2 = df2.rename(columns={'more than 1,000,001(股)':'1,000,001(股)以上'})

    # df2.to_csv("feature03.csv", index=True)
    # print(df2)
    return df2


def total_features(price, IIBS, MPSS, HSP):
    result = pd.merge(price, IIBS, left_index=True, right_index=True)
    result = pd.merge(result, MPSS, left_index=True, right_index=True)
    result = pd.merge(result, HSP, left_index=True, right_index=True, how='outer')

    # 使用 ffill 方法向前填充缺失值
    result = result.fillna(method='ffill')

    result_name = 'data/{}_Total_features.csv'.format(stock_id)
    result.to_csv(result_name, index=True)
    # print(result.isnull().sum())
    # print(result)

    return result



Price = TaiwanStockPrice() #股價
IIBuySell = TaiwanStockInstitutionalInvestorsBuySell() #三大法人
MPSSale = TaiwanStockMarginPurchaseShortSale() #融資融券借券
HSPer = TaiwanStockHoldingSharesPer() #張數區間持股比例

Total_features = total_features(Price,IIBuySell,MPSSale,HSPer)


