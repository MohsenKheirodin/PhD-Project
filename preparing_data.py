import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import requests
import matplotlib.pyplot as plt
import shap
import copy
from hmmlearn import hmm
import requests
from persiantools.jdatetime import JalaliDate
import warnings
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import timedelta

from joblib import dump, load
import os
from tqdm import tqdm

# from datalib.timeseries import TimeSeries
# from datalib.fundamentals import Fundamentals
# from datalib.tickers import Tickers

# from datalib.timeseries import TimeSeries, TGJUTimeSeries, TETimeSeries, IMETimeSeries
# from datalib.fundamentals import Fundamentals
# from datalib.tickers import Tickers, TGJUTickers, TETickers

# %load_ext autoreload
# %autoreload 2

# tickers = Tickers()
# ts = TimeSeries()
# fd = Fundamentals()
# tgju_tickers = TGJUTickers()
# te_tickers = TETickers()
# tgju_ts = TGJUTimeSeries()
# te_ts = TETimeSeries()
# ime_ts = IMETimeSeries()


# # Get raw Data

# def Get_Raw_Data():
#     list = pd.read_excel('get (1).xlsx')
#     list = list[['symbol','first_trading_date','sector_code']]
    
#     price_raw = pd.DataFrame()
#     income_raw = pd.DataFrame()
#     monthly_raw = pd.DataFrame()
    
#     for sym in tqdm(list['symbol'].unique().tolist(), desc="Processing symbols"):
#         try:
#             price_stock = ts.get_daily(symbols=(sym, ), cols=None, to_date=None, from_date = None)
#             price_raw = pd.concat([price_raw, price_stock])
    
#             income_stock = fd.get_income_statement(symbols=(sym, ), from_date= None, to_date= None, mode='earliest')
#             income_raw = pd.concat([income_raw, income_stock])
    
#             monthly_stock = fd.get_monthly_production(symbols=(sym, ), from_date= None, to_date= None, mode='earliest')
#             monthly_raw = pd.concat([monthly_raw, monthly_stock])
            
#         except Exception as e:
#             print('Error in Getting ' + str(sym))
    
#     price_raw.reset_index(inplace = True)
#     price_raw['date'] = pd.to_datetime(price_raw['date'])
#     price_raw = price_raw.set_index('date')
#     price_raw = price_raw.dropna(subset=['close','adj_close']).sort_index(ascending = True)     
#     price_raw['symbol'] = price_raw['symbol'].apply(lambda x: x.replace('ي','ی').replace('ك','ک'))
#     price_raw.to_parquet('price_raw_data.parquet', index=True)   

#     income_raw = income_raw.reset_index()
#     income_raw['symbol'] = income_raw['symbol'].apply(lambda x: x.replace('ي','ی').replace('ك','ک'))
#     income_raw = income_raw.set_index('period_ending_date')
#     income_raw.loc[:,~income_raw.columns.duplicated()].to_parquet('income_raw_data.parquet', index=True)
    
#     monthly_raw.loc[:,~monthly_raw.columns.duplicated()].to_parquet('monthly_raw_data.parquet', index=True)
    
#     Brent_oil = tgju_ts.get_daily(symbols=('energy-brent-oil', ), cols=None, from_date=None, to_date=None)
#     Brent_oil.to_parquet('Brent_raw_data.parquet', index=True)

#     index_raw = get_index()
#     index_raw = pd.read_parquet('index_raw_data.parquet')
#     index_raw.reset_index(inplace = True)
#     index_raw['date'] = pd.to_datetime(index_raw['date'])
#     index_raw = index_raw.set_index('date')
#     index_raw = index_raw.rename(columns={'adj_vwap': 'index_vwap', 'adj_low': 'index_low', 'adj_high': 'index_high'}).sort_index(ascending = True)   
#     index_raw.to_parquet('index_raw_data.parquet', index=True)
    
    
# Size Categorization

def size_categorization(a, L1, L2, L3):
    if a/(10**13) < L1:
        return 0
    elif a/(10**13) < L2:
        return 1
    elif a/(10**13) < L3:
        return 2
    else:
        return 3
    
    
# Categorizing the stocks based on size, industry and first date

def categorizing_Based_on_Indsry_size(list, price):

    symbol        = []
    first_date    = []
    size_category = []
    industry      = []
    is_production = []
    
    mudium_size_threshhold_hmat = 5
    huge_size_threshhold_hmat = 20
    suphug_size_threshhold_hmat = 70

    for i in range(list.shape[0]):   
        
        sym =  list['symbol'].iloc[i]
        if price[price['symbol'] == sym].shape[0] > 0:
                                    
            size_class = size_categorization(price_info['market_cap'].iloc[-1], mudium_size_threshhold_hmat, huge_size_threshhold_hmat, suphug_size_threshhold_hmat)
            indsry     = list['sector_code'].iloc[i]
            frst_date  = list['first_trading_date'].iloc[i]
            is_prod    = np.where(list['production'].iloc[i] == 'yes', 1.0, 0.0)
            
            price.loc[price['symbol'] == sym,'industry']   = indsry
            price.loc[price['symbol'] == sym,'size']       = size_class    
            price.loc[price['symbol'] == sym,'first_date'] = frst_date  
            price.loc[price['symbol'] == sym,'is_prod']    = is_prod  
            
            symbol.append(sym)
            industry.append(indsry)
            first_date.append(frst_date)
            size_category.append(size_class)
            is_production.append(is_prod)
    
    List_stock = pd.DataFrame({'symbol': symbol, 'INDUSTRY': industry, 'FIRST_DATE': first_date, 'SIZE': size_category, 'IS_PROD': is_production})
    List_stock['IS_PROD'] = List_stock['IS_PROD'].astype(int)
    return price, List_stock


# Add Three Border Labeling

def add_three_border_labeling(lists, three_border_dict, col):

    days_three_border = three_border_dict['num_days']
    
    t_p_abs = three_border_dict['take_profit_abs']
    t_p_rel = three_border_dict['take_profit_rel']
    
    s_l_abs = three_border_dict['stop_loss_abs']
    s_l_rel = three_border_dict['stop_loss_rel']

    price_info_whole = pd.DataFrame()
    
    for i in range(lists.shape[0]):   
        try:
            price_info = price[price['symbol'] == lists.iloc[i]] 
            if price_info.shape[0] > 0:
                                
                price_info = price_info.reset_index().merge(dfindex.reset_index()[['date','index_vwap']], on = 'date', how = 'left').set_index('date').sort_index(ascending = True)
                price_info.dropna(subset=['close','adj_close','index_vwap'], inplace=True)
                
                price_info = three_border_labeling(price_info, num_days = days_three_border, tp = t_p_abs, sl = s_l_abs, rel_index = False, col = col)
                price_info = three_border_labeling(price_info, num_days = days_three_border, tp = t_p_rel, sl = s_l_rel, rel_index = True,  col = col)
                
                price_info_whole = pd.concat([price_info_whole, price_info], axis = 0)
        
                print(lists['symbol'].iloc[i])       
        
        except:
            print('Error in :' + str(lists['symbol'].iloc[i]))

    price_info_whole = price_info_whole.rename(columns={'3B_label_abs': '3B_abs_'+str(days_three_border)+'_days', 
                                                        '3B_label_rel': '3B_abs_'+str(days_three_border)+'_days'})
            
    return price_info_whole
    
    
# Labeling the data

def assign_label(row, quantiles_by_date, labels, col):
    try:
        bins = quantiles_by_date.loc[row.name].values
        unique_bins = pd.unique(bins)
        return pd.cut([row[col]], bins=unique_bins, labels=labels, include_lowest=True, duplicates='drop')[0]
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return None

def Relative_Labeling_Data(price_info_whole, days):

    all_dates = price_info_whole.index.unique()
    weird_dates = []

    for date in all_dates:
        price_info_whole_date = price_info_whole[price_info_whole.index == date]
        quantiles_by_date_ret = price_info_whole_date['ret_abs_'+str(days)+'_days'].quantile([0, 0.25, 0.5, 0.75, 1])
        if ((quantiles_by_date_ret.iloc[0] == quantiles_by_date_ret.iloc[1]) or 
            (quantiles_by_date_ret.iloc[1] == quantiles_by_date_ret.iloc[2]) or 
            (quantiles_by_date_ret.iloc[2] == quantiles_by_date_ret.iloc[3]) or 
            (quantiles_by_date_ret.iloc[3] == quantiles_by_date_ret.iloc[4])):
            weird_dates.append(date)
    
    price_info_whole = price_info_whole[~price_info_whole.index.isin(weird_dates)]
    
    quantiles_by_date_Ret = price_info_whole.groupby(price_info_whole.index)['ret_abs_'+str(days)+'_days'].quantile([0, 0.25, 0.5, 0.75, 1]).unstack()    
    labels = [1, 2, 3, 4]
    
    price_info_whole['label_multi_classes_' + str(days) + '_days'] = price_info_whole.apply(lambda row: assign_label(row, quantiles_by_date_Ret, labels, col = 'ret_abs_'+str(days)+'_days'), axis=1)   
    
    print('End of Relative_Labeling_Data for ' + str(days) + ' Days')
    return price_info_whole

# three border Labeling

def three_border_labeling(df_price_daily, num_days, tp, sl, rel_index, col):
    
    if rel_index:
        df_price_daily['3B_label_rel'] = 2        
        for date in df_price_daily.index[:-num_days]:
            set = 0
            for i in range(1, num_days + 1):
                if (df_price_daily.shift(-i).loc[date,'adj_high'] / df_price_daily.loc[date,'adj_close']) - (df_price_daily.shift(-i).loc[date,'index_vwap'] / df_price_daily.loc[date,'index_vwap']) > tp / 100:
                    df_price_daily.loc[date, '3B_label_rel'] = 4
                    set = 1
                    break
                elif (df_price_daily.shift(-i).loc[date,'adj_low'] / df_price_daily.loc[date,'adj_close']) - (df_price_daily.shift(-i).loc[date,'index_vwap'] / df_price_daily.loc[date,'index_vwap']) < -sl / 100:
                    df_price_daily.loc[date, '3B_label_rel'] = 1
                    set = 1
                    break
            if set == 0:
                if df_price_daily.shift(-i).loc[date,'adj_close'] / df_price_daily.loc[date,'adj_close'] >= df_price_daily.shift(-i).loc[date,'index_vwap'] / df_price_daily.loc[date,'index_vwap']:
                    df_price_daily.loc[date, '3B_label_rel'] = 3

    else:
        df_price_daily['3B_label_abs'] = 2        
        for date in df_price_daily.index[:-num_days]:
            set = 0
            for i in range(1, num_days + 1):
                if df_price_daily.shift(-i).loc[date,'adj_high'] / df_price_daily.loc[date,'adj_close'] - 1 > tp / 100:
                    df_price_daily.loc[date, '3B_label_abs'] = 4
                    set = 1
                    break
                elif df_price_daily.shift(-i).loc[date,'adj_low'] / df_price_daily.loc[date,'adj_close'] - 1 < -sl / 100:
                    df_price_daily.loc[date, '3B_label_abs'] = 1
                    set = 1
                    break
            if set == 0:
                if df_price_daily.shift(-i).loc[date,'adj_close'] >= df_price_daily.loc[date,'adj_close']:
                    df_price_daily.loc[date, '3B_label_abs'] = 3
        
    return df_price_daily


# # Checking 3 Border
# df_price_daily = ts.get_daily(symbols=('فولاد', ), cols=None, to_date=None, from_date = first_date_rf) 
# df_price_daily = df_price_daily.xs('فولاد', level = 'symbol')
# df_index, first_date_rf, all_days = Create_Index()
# nanrows = df_price_daily['adj_vwap'].isna()
# df_price_daily = df_price_daily[~nanrows]

# df_price_daily = three_border_labeling(df_price_daily, num_days = 15, tp = 8, sl = 8, rel_index = True, dfindex = copy.deepcopy(df_index))

# df_price_daily.reset_index(inplace = True)
# df_index.reset_index(inplace = True)
# df_price_daily = df_price_daily.merge(df_index[['date','index_vwap']], on = 'date', how = 'left')
# df_price_daily['price_rel'] = df_price_daily['adj_close'] / df_price_daily['index_vwap']
# df_price_daily.set_index('date', inplace = True)

# # df_price_daily['max'] = df_price_daily['adj_close'].shift(-15).rolling(window = 16).max() / df_price_daily['adj_close'] -1
# # df_price_daily['min'] = df_price_daily['adj_close'].shift(-15).rolling(window = 16).min() / df_price_daily['adj_close'] -1

# df_price_daily['max'] = df_price_daily['price_rel'].shift(-15).rolling(window = 16).max() / df_price_daily['price_rel'] -1
# df_price_daily['min'] = df_price_daily['price_rel'].shift(-15).rolling(window = 16).min() / df_price_daily['price_rel'] -1

# df_price_daily['maxx'] = np.where(df_price_daily['max'] > (0.08), 1, 0)
# df_price_daily['minn'] = np.where(df_price_daily['min'] < (-0.08), -1, 0)

# df_price_daily[['adj_close','3B_label_rel','max','min','maxx','minn']].iloc[100:160]
# # df_price_daily[['price_rel','3B_label_rel','max','min','maxx','minn']].iloc[100:160]

# df_price_daily['diff'] = 0
# df_price_daily.loc[(df_price_daily['maxx'] == 1) & (df_price_daily['3B_label_rel']  != 4) | (df_price_daily['maxx'] != 1) & (df_price_daily['3B_label_rel'] == 4), 'diff'] = 1
# df_price_daily.loc[(df_price_daily['minn'] == -1) & (df_price_daily['3B_label_rel'] != 1) | (df_price_daily['minn'] != -1) & (df_price_daily['3B_label_rel'] == 1), 'diff'] = 1
# df_price_daily['diff'].sum()

# Getting Index data

def get_index():
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9,fa;q=0.8',
        'Connection': 'keep-alive',
        'Origin': 'https://www.tsetmc.com',
        'Referer': 'https://www.tsetmc.com/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
    }
    
    response = requests.get('https://cdn.tsetmc.com/api/Index/GetIndexB2History/32097828799138957', headers=headers)
    df = pd.DataFrame(response.json()['indexB2'])
    df = pd.DataFrame(response.json()['indexB2'])
    df.columns = ['ins_code', 'date', 'adj_vwap', 'adj_low', 'adj_high']
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['trade_symbol'] = 'شاخص کل'
    df = df.set_index(['date', 'trade_symbol']).sort_index()
    return df


# Regime Detection by HMM

def regime_detection(price, col, windw = 5, coefs_regime = [1,3,2,5]):
     
    # The code of a non-smooth regime switch model
    
    # # Less smooth
    # model = hmm.GaussianHMM(n_components=2, covariance_type="full")
    # model.fit(returns.reshape(-1, 1))
    # hidden_states = model.predict(returns.reshape(-1, 1))
    # a = pd.Series(hidden_states)
    # a.index = df_index2.index

    # 1.1. Detecting Positive Trends
    # more smooth
    # Detecting rising patterns
    price['ret_non_smth'] = price[col].pct_change()
    price['ret_smooth1']  = price['ret_non_smth'].rolling(window=windw).mean().dropna()
    price.loc[price['ret_smooth1'] > 0, 'ret_smooth1'] = price.loc[price['ret_smooth1'] > 0, 'ret_smooth1'] * coefs_regime[0]
    returns_smooth = price['ret_smooth1'].dropna().values
    
    model1 = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model1.startprob_ = np.array([0.5, 0.5])  # Initial state probabilities
    model1.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])  # High probability of staying in the same state
    model1.fit(returns_smooth.reshape(-1, 1))
    hidden_states = model1.predict(returns_smooth.reshape(-1, 1))
    a = pd.Series(hidden_states, index=price['ret_smooth1'].dropna().index)

    price['regime1'] = a

    # 1.2. Detecting Negative Trends
    
    # price_plot = price.iloc[1:-1]
    # plt.scatter(price_plot.index, price_plot['adj_close'], c=price_plot['regime'], cmap='Set1')

    # Detecting falling patterns
    price['ret_smooth2'] = price['ret_smooth1']
    price.loc[price['ret_smooth2'] < 0, 'ret_smooth2'] = price.loc[price['ret_smooth2'] < 0, 'ret_smooth2'] * coefs_regime[1]
    returns_smooth = price['ret_smooth2'].dropna().values
    
    model2 = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model2.startprob_ = np.array([0.5, 0.5])  # Initial state probabilities
    model2.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])  # High probability of staying in the same state
    model2.fit(returns_smooth.reshape(-1, 1))
    hidden_states = model2.predict(returns_smooth.reshape(-1, 1))
    a = pd.Series(hidden_states, index=price['ret_smooth2'].dropna().index)

    price['regime2'] = -a

    # 1.3. Merge Trends

    price['regime'] = price['regime1'] + price['regime2']

    # price.loc[(price['regime1'] == 1) & (price['regime2'] == -1), 'regime'] = 0

    # # 2.1. Repeat the process. Detect positives
    
    # price1 = price[price['regime'] != 0]
    # price2 = price[price['regime'] == 0]

    # price2.loc[price2['ret_smooth1'] > 0, 'ret_smooth1'] = price2.loc[price2['ret_smooth1'] > 0, 'ret_smooth1'] * coefs_regime[2]
    # returns_smooth = price2['ret_smooth1'].dropna().values
    # model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    # model.startprob_ = np.array([0.5, 0.5])  # Initial state probabilities
    # model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])  # High probability of staying in the same state
    # model.fit(returns_smooth.reshape(-1, 1))
    # hidden_states = model.predict(returns_smooth.reshape(-1, 1))
    # a = pd.Series(hidden_states, index=price2['ret_smooth1'].dropna().index)
    # price2['regime1'] = a

    # # 2.2. Repeat the process. Detect Negatives

    # price2.loc[price2['ret_smooth2'] < 0, 'ret_smooth2'] = price2.loc[price2['ret_smooth2'] < 0, 'ret_smooth2'] * coefs_regime[3]
    # returns_smooth = price2['ret_smooth2'].dropna().values
    # model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    # model.startprob_ = np.array([0.5, 0.5])  # Initial state probabilities
    # model.transmat_ = np.array([[0.9, 0.1], [0.02, 0.98]])  # High probability of staying in the same state
    # model.fit(returns_smooth.reshape(-1, 1))
    # hidden_states = model.predict(returns_smooth.reshape(-1, 1))
    # a = pd.Series(hidden_states, index=price2['ret_smooth2'].dropna().index)
    # price2['regime2'] = a

    # # 2.3. Repeat the process. Merging
    # price2['regime'] = 0
    # price2.loc[price2['regime1'] == 1, 'regime'] = 1
    # price2.loc[price2['regime2'] == -1, 'regime'] = -1

    # # 3. Final Merging
    # price = pd.concat([price1,price2]).sort_index(ascending = True)

    price = price.drop(columns = ['regime1','regime2','ret_non_smth','ret_smooth1','ret_smooth2'])
    return price, model1, model2

def since_last_regime_change(price):
    
    price['nrows'] = np.arange(1, len(price) + 1)
    price['change_regime'] = 0
    price.loc[price['regime'] != price['regime'].shift(1), 'change_regime'] = 1
    price.loc[price['nrows'] == 1, 'change_regime'] = 0
    
    price['tillnext'] = 0
    price['currnt_reg_days'] = 0
    
    changes = price.loc[price['change_regime'] == 1, 'nrows'].tolist()
    changes_diff = price.loc[price['change_regime'] == 1, 'nrows'].diff().tolist()[1:]
    
    for i in range(len(changes)-1):
        price.loc[price['nrows'] == changes[i], 'tillnext'] = changes_diff[i]

    price.loc[price['nrows'] < changes[0], 'currnt_reg_days'] = price['nrows']
    for i in range(len(changes)-1):
        for row in range(changes[i],changes[i+1]):
            price.loc[price['nrows'] == row, 'currnt_reg_days'] = row - changes[i] + 1
        
    price = price.drop(columns = ['nrows','change_regime','tillnext'])
    return price


# Adding Fundamental situation of the Market

def Add_Aggregate_fundamental_features(df_index):
    
    income_statement = pd.read_parquet('incomestatement.parquet')
    income_statement = income_statement.reset_index().set_index('period_ending_date')
    income_statement.sort_index(inplace = True, ascending = True)
    ending_dates = income_statement.index.unique().tolist()
        
    start_date = None
    for date in ending_dates:
        rows_matching_date = income_statement.loc[income_statement.index == date, ['symbol', 'publish_day']]
        if rows_matching_date.shape[0] > 150:
            start_date = pd.to_datetime(rows_matching_date['publish_day'].max())
            break
    
    Total_revenue = pd.DataFrame(index=pd.date_range(start=start_date, end = df_index.index[-1], freq='D'))
    Total_revenue['net_operating'] = income_statement.loc[income_statement.index == date, 'Operating Profit (Loss)'].sum() / (10**13)
    Total_revenue['net_income']    = income_statement.loc[income_statement.index == date, 'Net Income (Loss)'].sum() / (10**13)
    Total_revenue['sales']         = income_statement.loc[income_statement.index == date, 'Sales'].sum() / (10**13)
    
    symbols = rows_matching_date['symbol'].tolist()
    for end_date in income_statement[income_statement.index > start_date].index.unique():
        for syms in income_statement.loc[income_statement.index == end_date, 'symbol'].unique().tolist():
            if syms in symbols:
                income_stock = income_statement[income_statement['symbol'] == syms]
                pblshdt = income_stock.loc[income_stock.index == end_date, 'publish_day'].iloc[0] 
                pblshdt = pd.to_datetime(pblshdt).normalize()
    
                growth_sls = income_stock.loc[income_stock.index == end_date, 'Sales'] - income_stock.shift(1).loc[income_stock.index == end_date, 'Sales']
                growth_opr = income_stock.loc[income_stock.index == end_date, 'Operating Profit (Loss)'] - income_stock.shift(1).loc[income_stock.index == end_date, 'Operating Profit (Loss)']
                growth_net = income_stock.loc[income_stock.index == end_date, 'Net Income (Loss)'] - income_stock.shift(1).loc[income_stock.index == end_date, 'Net Income (Loss)']
                
                if pblshdt in Total_revenue.index:
                    Total_revenue.loc[pblshdt:, 'net_operating'] += growth_opr.iloc[0] / (10**13)
                    Total_revenue.loc[pblshdt:, 'net_income']    += growth_net.iloc[0] / (10**13)
                    Total_revenue.loc[pblshdt:, 'sales']         += growth_sls.iloc[0] / (10**13)
            else:
                symbols.append(syms)
    
    Total_revenue['opr_margin'] = Total_revenue['net_operating']/Total_revenue['sales']
    Total_revenue['net_margin'] = Total_revenue['net_income']/Total_revenue['sales']
    
    Total_revenue.index.name = 'date'
    return df_index.reset_index().merge(Total_revenue.reset_index(), on = 'date', how = 'left').set_index('date'), start_date


# Creating the Index

def Create_Index(Selected_features):

    df_index = pd.read_parquet('index_raw_data.parquet')
    
    # Regime Detection
    if Selected_features['Regime']:

        set1   = [1,2.5,2,8]
        price1, _ , _ = regime_detection(price = df_index, col = 'index_vwap', coefs_regime = set1)
        total1 = price1['regime'].count()
        rato1  = [price1['regime'].value_counts()[0], price1['regime'].value_counts()[1], price1['regime'].value_counts()[-1]] / total1 
        eval1  = sum([np.abs(rato1[0] - 0.5), np.abs(rato1[1] - 0.3), np.abs(rato1[2] - 0.20)])
        
        set2   = [2,4,3,5]
        price2, _ , _ = regime_detection(price = df_index, col = 'index_vwap', coefs_regime = set2)
        total2 = price2['regime'].count()
        rato2  = [price2['regime'].value_counts()[0], price2['regime'].value_counts()[1], price2['regime'].value_counts()[-1]] / total2 
        eval2  = sum([np.abs(rato2[0] - 0.5), np.abs(rato2[1] - 0.3), np.abs(rato2[2] - 0.20)])
                
        set3   = [2,7,1,8]
        price3, _ , _ = regime_detection(price = df_index, col = 'index_vwap', coefs_regime = set3) 
        total3 = price3['regime'].count()
        rato3  = [price3['regime'].value_counts()[0], price3['regime'].value_counts()[1], price3['regime'].value_counts()[-1]] / total3 
        eval3  = sum([np.abs(rato3[0] - 0.5), np.abs(rato3[1] - 0.3), np.abs(rato3[2] - 0.20)])

        if   min([eval1, eval2, eval3]) == eval1: df_index = price1            
        elif min([eval1, eval2, eval3]) == eval2: df_index = price2            
        else: df_index = price3

        df_index = since_last_regime_change(df_index)

    
        # df_index2_plot = df_index2.iloc[1100:1300]
        # plt.scatter(df_index2_plot.index, df_index2_plot['index_vwap'], c=df_index2_plot['regime'], cmap='Set1')
        # plt.savefig('scatter_plot.png')

    # Make df_index ready for Merging and setting default first date
    df_index.reset_index(inplace = True)     
    df_index['date'] = pd.to_datetime(df_index['date'])
    first_date = df_index['date'].iloc[0]

    # Exchange Rate
    if Selected_features['Dollar']:
        df_exchange =  pd.read_excel('ex.xlsx')
        df_exchange['date'] = pd.to_datetime(df_exchange['تاریخ میلادی'])
        df_exchange = df_exchange.set_index('date').drop(columns = ['تاریخ میلادی'])
        df_exchange = df_exchange.rename(columns={'مقدار': 'exchange'})
        first_date_ex = pd.to_datetime(df_exchange.index[0])
        df_exchange.reset_index(inplace = True)
        df_exchange['date'] = pd.to_datetime(df_exchange['date'])
        df_index = df_index.merge(df_exchange[['date', 'exchange']], on='date', how='left')
        df_index['exchange'] = df_index['exchange'].ffill()
        if first_date < first_date_ex:
            first_date = first_date_ex
        

    # Risk Free Rate
    if Selected_features['Risk_free']:
        df_risk_free =  pd.read_excel('rf.xlsx')
        df_risk_free['date'] = pd.to_datetime(df_risk_free['تاریخ میلادی'])
        df_risk_free = df_risk_free.set_index('date').drop(columns = ['تاریخ میلادی'])
        df_risk_free = df_risk_free.rename(columns={'مقدار': 'risk_free'})
        first_date_rf = pd.to_datetime(df_risk_free.index[0])
        df_risk_free.reset_index(inplace = True)
        df_risk_free['date'] = pd.to_datetime(df_risk_free['date'])
        df_index = df_index.merge(df_risk_free[['date', 'risk_free']], on='date', how='left')
        df_index['risk_free'] = df_index['risk_free'].ffill()
        if first_date < first_date_rf:
            first_date = first_date_rf
    
    # Value of Trades
    if Selected_features['Value']:
        price_info_whole = pd.read_parquet('all_stocks_price_info.parquet')
        List_stock = pd.read_parquet('list_stocks.parquet')
        first_date_val = pd.to_datetime('2012-01-01')
        List_stock_old = List_stock.loc[List_stock['FIRST_DATE'] < first_date_val,'symbol']
        price_info_whole.reset_index(inplace = True)  
        price_info_whole_old_enough = price_info_whole[price_info_whole['symbol'].isin(List_stock_old)]
        value_of_trades = price_info_whole_old_enough.groupby('date')['value'].sum() / (10**13)
        value_of_trades = value_of_trades.reset_index()
        first_date_value = value_of_trades['date'].iloc[0]
        df_index = df_index.merge(value_of_trades[['date','value']], on = 'date', how = 'left')
        df_index['Daily_Trd_Vlu'] = df_index['value']
        if first_date < first_date_val:
            first_date = first_date_val
    
    # Brend Oil Data
    if Selected_features['Oil']:
        Brent_oil = pd.read_parquet('Brent_raw_data.parquet')
        Brent_oil = Brent_oil.xs('energy-brent-oil', level = 'symbol')
        Brent_oil.reset_index(inplace = True)
        Brent_oil['date'] = pd.to_datetime(Brent_oil['date'], errors='coerce')
        first_date_brent =  Brent_oil['date'].iloc[0]
        full_date_range = pd.date_range(start=Brent_oil['date'].min(), end=Brent_oil['date'].max())
        Brent_oil = Brent_oil.set_index('date').reindex(full_date_range).rename_axis('date').reset_index()
        Brent_oil = Brent_oil.ffill()
        Brent_oil = Brent_oil.dropna()
        Brent_oil['oil_price'] = Brent_oil['close']
        df_index = df_index.merge(Brent_oil[['date','oil_price']], on = 'date', how = 'left')
        if first_date < first_date_brent:
            first_date = first_date_brent

    # Set index again for df_index
    df_index.set_index('date', inplace = True)

    # Aggregate Fundamental Data
    if Selected_features['Fundamental']:
        df_index, first_date_fund = Add_Aggregate_fundamental_features(df_index)
        if first_date < first_date_fund:
            first_date = first_date_fund

    # Making the output Ready
    df_index = df_index.loc[df_index.index > first_date,:]
    df_index = df_index.drop(columns=['ins_code']).drop_duplicates(keep='first')
    all_days = df_index.index.tolist()
    
    return df_index, first_date, all_days

# # Check Missing Of rf and ex

# df_risk_free =  pd.read_excel('rf.xlsx')
# df_risk_free['date'] = pd.to_datetime(df_risk_free['تاریخ میلادی'])
# df_risk_free = df_risk_free.set_index('date').drop(columns = ['تاریخ میلادی'])
# list_rf = df_risk_free.index.tolist()
# df_index = get_index()
# df_index2 = df_index.xs('شاخص کل', level = 'trade_symbol')
# df_index2 = df_index2.loc[df_index2.index > df_risk_free.index[0]]
# list_index = df_index2.index.tolist()
# list_index
# Not_rf = [item for item in list_index if item not in list_rf]

# df_exchange =  pd.read_excel('ex.xlsx')
# df_exchange['date'] = pd.to_datetime(df_exchange['تاریخ میلادی'])
# df_exchange = df_exchange.set_index('date').drop(columns = ['تاریخ میلادی'])
# list_ex = df_exchange.index.tolist()
# df_index = get_index()
# df_index2 = df_index.xs('شاخص کل', level = 'trade_symbol')
# df_index2 = df_index2.loc[df_index2.index > df_exchange.index[0]]
# list_index = df_index2.index.tolist()
# Not_ex = [item for item in list_index if item not in list_ex]

# print(Not_rf)
# print(Not_ex)


# Calculate Macro Features

def calculating_Macro_features(df_index, df_index_info, lags_return):
    
    # Index return and Candle Features        
    df_index, index_features = Calculate_high_and_low(df_index = df_index, column = 'index_vwap', lag = 0, period = ['w','m','y'])
    
    df_index['Market_Ret_L2'] = df_index['index_vwap'].shift(1) / df_index['index_vwap'].shift(2) - 1
    df_index['Market_Ret_L3'] = df_index['index_vwap'].shift(2) / df_index['index_vwap'].shift(3) - 1
    
    df_index['Market_Range_L1'] = df_index['index_high'].shift(0) / df_index['index_low'].shift(0) - 1
    df_index['Market_Range_L2'] = df_index['index_high'].shift(1) / df_index['index_low'].shift(1) - 1
    df_index['Market_Range_L3'] = df_index['index_high'].shift(2) / df_index['index_low'].shift(2) - 1

    index_features.extend(['index_high','index_low','Market_Ret_L2','Market_Ret_L3','Market_Range_L1','Market_Range_L2','Market_Range_L3'])

    # Regime Features
    regime_features = ['regime']

    # Exchange Rate Features 
    dollar_features = []
    if df_index_info['Dollar']:
        df_index, dollar_features = Calculate_high_and_low(df_index = df_index, column = 'exchange', lag = 2)

    # Fundamental Aggregate Features
    fundamental_values_features = []
    fundamental_margin_features = []
    if df_index_info['Fundamental']:
        
        df_index['ptoE_index'] = df_index['index_vwap'] / df_index['net_income']
        df_index['ptoS_index'] = df_index['index_vwap'] / df_index['sales']
        
        df_index, opr_features        = Calculate_high_and_low(df_index = df_index, column = 'net_operating', lag = 0)
        df_index, net_features        = Calculate_high_and_low(df_index = df_index, column = 'net_income'   , lag = 0)
        df_index, sales_features      = Calculate_high_and_low(df_index = df_index, column = 'sales'        , lag = 0)
        df_index, opr_margin_features = Calculate_high_and_low(df_index = df_index, column = 'opr_margin'   , lag = 0)
        df_index, net_margin_features = Calculate_high_and_low(df_index = df_index, column = 'net_margin'   , lag = 0)
        df_index, ptoE_features       = Calculate_high_and_low(df_index = df_index, column = 'ptoE_index'   , lag = 0)
        df_index, ptoS_features       = Calculate_high_and_low(df_index = df_index, column = 'ptoS_index'   , lag = 0)
        
        fundamental_values_features.extend(opr_features)
        fundamental_values_features.extend(net_features)
        fundamental_values_features.extend(sales_features)
        
        fundamental_margin_features.extend(opr_margin_features)
        fundamental_margin_features.extend(net_margin_features)
        fundamental_margin_features.extend(ptoE_features)
        fundamental_margin_features.extend(ptoS_features)
        
    
    # Risk Free Features    
    risk_free_features = []
    if df_index_info['Risk_free']:
        df_index, risk_free_features = Calculate_high_and_low(df_index = df_index, column = 'risk_free', lag = 0)

    # Oli Price Features
    oil_features = []
    if df_index_info['Oil']:
        df_index, oil_features = Calculate_high_and_low(df_index = df_index, column = 'oil_price', lag = 0)

    
    # Aggregate Trading Value of the Market
    trade_volume_features = []
    if df_index_info['Value']:
        df_index, trade_volume_features = Calculate_high_and_low(df_index = df_index, column = 'Daily_Trd_Vlu', lag = 0)

    # OutPut
    out_put_columns = []
    for lag in lags_return:
        df_index['ret_'+str(lag)+'_days']   = df_index['index_vwap'].shift(-lag) / df_index['index_vwap'] - 1
        df_index['binary_'+str(lag)+'_days'] = 0
        df_index.loc[df_index['ret_'+str(lag)+'_days'] > 0, 'binary_'+str(lag)+'_days'] = 1

        df_index['class_'+str(lag)+'_days'] = 0
        quantiles = df_index['ret_'+str(lag)+'_days'].quantile([0, 0.25, 0.5, 0.75, 1])
        
        for i in range(quantiles.shape[0]-1):
            df_index.loc[(df_index['ret_'+str(lag)+'_days'] > quantiles.iloc[i]) & (df_index['ret_'+str(lag)+'_days'] < quantiles.iloc[i+1]), 'class_'+str(lag)+'_days'] = i
        
        df_index['class_'+str(lag)+'_days'] = df_index['class_'+str(lag)+'_days'] + 1
        out_put_columns.extend(['ret_'+str(lag)+'_days', 'binary_'+str(lag)+'_days', 'class_'+str(lag)+'_days'])
        
    df_index.dropna(inplace = True)
    columns = {'INDEX':index_features, 'REGIME':regime_features, 'DOLLAR':dollar_features, 'FUND_VAL':fundamental_values_features, 'FUND_RATIO':fundamental_margin_features,
                       'RISKFREE':risk_free_features, 'OIL':oil_features, 'VALUE':trade_volume_features, 'OUTPUT':out_put_columns}

    return df_index, columns


def Calculate_high_and_low(df_index, column, lag, period = ['m','y']):

    day   = 1
    week  = 5
    month = 20
    year  = 60
    
    df_index[column + '_D'] = df_index[column].shift(lag) / df_index[column].shift(day+lag) - 1
    df_index[column + '_W'] = df_index[column].shift(lag) / df_index[column].shift(week+lag) - 1
    df_index[column + '_M'] = df_index[column].shift(lag) / df_index[column].shift(month+lag) - 1
    df_index[column + '_Y'] = df_index[column].shift(lag) / df_index[column].shift(year+lag) - 1

    featuers_added = [column, column + '_D',column + '_W',column + '_M',column + '_Y']
    
    if 'w' in period:
        df_index[column + '_Khiz_W'] = df_index[column].shift(lag) / df_index[column].shift(lag).rolling(window = week+lag).min() - 1
        df_index[column + '_Riz_W']  = df_index[column].shift(lag) / df_index[column].shift(lag).rolling(window = week+lag).max() - 1
        featuers_added.extend([column + '_Khiz_W',column + '_Riz_W'])
    
    if 'm' in period:
        df_index[column + '_Khiz_M'] = df_index[column].shift(lag) / df_index[column].shift(lag).rolling(window = month+lag).min() - 1
        df_index[column + '_Riz_M']  = df_index[column].shift(lag) / df_index[column].shift(lag).rolling(window = month+lag).max() - 1
        featuers_added.extend([column + '_Khiz_M',column + '_Riz_M'])
    
    if 'y' in period:   
        df_index[column + '_Khiz_Y'] = df_index[column].shift(lag) / df_index[column].shift(lag).rolling(window = year+lag).min() - 1    
        df_index[column + '_Riz_Y']  = df_index[column].shift(lag) / df_index[column].shift(lag).rolling(window = year+lag).max() - 1
        featuers_added.extend([column + '_Khiz_Y',column + '_Riz_Y'])

    return df_index, featuers_added



# Features Categorization for Index info

def Select_Features_Categories_index(X_train, X_test, feat_dict, columns):
    
    index_info         = columns['INDEX']
    Exchange_info      = columns['DOLLAR']
    Risk_free_info     = columns['RISKFREE']
    oil_info           = columns['OIL']
    trade_value_info   = columns['VALUE']
    Fundamental_values = columns['FUND_VAL']
    Fundamental_ratios = columns['FUND_RATIO']
    regime_info        = columns['REGIME']

    features = []
    if feat_dict['Trend']:
        features.extend(index_info)
        
    if feat_dict['Value']:
        features.extend(trade_value_info)
        
    if feat_dict['Exchange']:
        features.extend(Exchange_info)
        
    if feat_dict['risk_free']:
        features.extend(Risk_free_info)
        
    if feat_dict['Oil']:
        features.extend(oil_info)
        
    if feat_dict['fundamental_value']:
        features.extend(Fundamental_values)
        
    if feat_dict['fundamental_ratios']:
        features.extend(Fundamental_ratios)
        
    if feat_dict['regime']:
        features.extend(regime_info)

    return X_train[features], X_test[features]


# Calculating Technical Indicators


from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

def add_technical_indicators(df_price_daily, col):

    df_price_daily['volume'] = df_price_daily['value'] / df_price_daily['adj_vwap']
    df_price_daily['volume'] = df_price_daily['volume'].astype(int)
    
    macd_indicator = MACD(close=df_price_daily[col])
    df_price_daily['trend_macd'] = macd_indicator.macd()
    df_price_daily['trend_macd_signal'] = macd_indicator.macd_signal()
    df_price_daily['trend_macd_diff'] = macd_indicator.macd_diff()
    
    # ema_fast = EMAIndicator(close=df_price_daily[col], window=12)
    # ema_slow = EMAIndicator(close=df_price_daily[col], window=26)
    # df_price_daily['trend_ema_fast'] = ema_fast.ema_indicator()
    # df_price_daily['trend_ema_slow'] = ema_slow.ema_indicator()

    # Make these variables stationary
    df_price_daily['trend_macd']        = df_price_daily['trend_macd']        / df_price_daily[col]
    df_price_daily['trend_macd_signal'] = df_price_daily['trend_macd_signal'] / df_price_daily[col]
    df_price_daily['trend_macd_diff']   = df_price_daily['trend_macd_diff']   / df_price_daily[col]
    
    rsi_indicator = RSIIndicator(close=df_price_daily[col])
    df_price_daily['momentum_rsi'] = rsi_indicator.rsi()
    
    mfi_indicator = MFIIndicator(high=df_price_daily['adj_high'], low=df_price_daily['adj_low'], close=df_price_daily['adj_close'], volume=df_price_daily['volume'])
    df_price_daily['volume_mfi'] = mfi_indicator.money_flow_index()
    
    roc_indicator = ROCIndicator(close=df_price_daily[col])
    df_price_daily['momentum_roc'] = roc_indicator.roc()
    
    technical_list = ['trend_macd','trend_macd_signal','trend_macd_diff','trend_ema_fast','trend_ema_slow','momentum_rsi','volume_mfi','momentum_roc']
    
    return df_price_daily, technical_list


# Getting Pivot points

# from zigzag import peak_valley_pivots

# def find_pivots(df_price_daily, col = 'adj_close', pct_range = 0.05):

#     peak_threshold = pct_range
#     valley_threshold = -pct_range  
    
#     pivots = peak_valley_pivots(df_price_daily[col].values, peak_threshold, valley_threshold)
    
#     # Separate peaks and valleys, excluding the first and last points
#     peak_indices = np.where(pivots == 1)[0]
#     valley_indices = np.where(pivots == -1)[0]

#     peak_indices = peak_indices[(peak_indices != 0) & (peak_indices != len(df_price_daily) - 1)]
#     valley_indices = valley_indices[(valley_indices != 0) & (valley_indices != len(df_price_daily) - 1)]
    
#     # Extract prices at peaks and valleys
#     peak_prices = df_price_daily[col].iloc[peak_indices]
#     valley_prices = df_price_daily[col].iloc[valley_indices]
    
#     # # Plotting
#     # plt.figure(figsize=(14, 7))
#     # plt.plot(df_price_daily.index, df_price_daily['adj_close'], label='Adjusted Close', color='blue')
#     # plt.scatter(peak_prices.index, peak_prices, color='green', marker='^', label='Peaks', s=100)
#     # plt.scatter(valley_prices.index, valley_prices, color='red', marker='v', label='Valleys', s=100)
#     # plt.title('Adjusted Close Price with ZigZag Peaks and Valleys (Excluding First and Last Points)')
#     # plt.xlabel('Date')
#     # plt.ylabel('Adjusted Close Price')
#     # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
#     # plt.legend()
#     # plt.tight_layout()  # Adjust layout to prevent clipping of labels
#     # plt.show()
    
#     peak_columns   = ['peak4', 'peak3', 'peak2', 'peak1']
#     valley_columns = ['vlly4', 'vlly3', 'vlly2', 'vlly1']
#     pivot_columns  = ['valy4', 'pik4', 'valy3', 'pik3', 'valy2', 'pik2', 'valy1', 'pik1']
#     df_price_daily[peak_columns + valley_columns] = np.nan

#     for i in range(df_price_daily.shape[0]):
#         recent_peaks = peak_indices[peak_indices < i][-4:]
#         recent_valleys = valley_indices[valley_indices < i][-4:]
        
#         if len(recent_peaks) > 0:
#             df_price_daily.loc[df_price_daily.index[i], ['peak4', 'peak3', 'peak2', 'peak1'][-len(recent_peaks):]] = \
#                 df_price_daily[col].iloc[recent_peaks].values
    
#         if len(recent_valleys) > 0:
#             df_price_daily.loc[df_price_daily.index[i], ['vlly4', 'vlly3', 'vlly2', 'vlly1'][-len(recent_valleys):]] = \
#                 df_price_daily[col].iloc[recent_valleys].values
    
#     df_price_daily[['valy4', 'pik4', 'valy3', 'pik3', 'valy2', 'pik2', 'valy1', 'pik1']] = ((df_price_daily[['vlly4', 'peak4', 'vlly3', 'peak3', 'vlly2', 'peak2', 'vlly1', 'peak1']].div(
#             df_price_daily[col], axis=0) - 1) * 100)
        
#     return df_price_daily.drop(columns=peak_columns + valley_columns)


# Find special days of after IPO and Closed days of stocks

def Find_Closed_IPO_Dates(df_price, df_index):
    
    df_copy = copy.deepcopy(df_index)
    
    # df_price['IPO'] = 0
    # df_price['IPO'].iloc[:50] = -1
    
    df_copy['exist'] = 0   
    df_copy = df_copy[df_copy.index > df_price.index[0]]
    df_price = df_price[df_price.index > df_copy.index[0]]
    df_copy['row_number'] = np.arange(1, len(df_copy) + 1)
    full_date_range = df_copy.index
    df_price = df_price.dropna(subset=['adj_close'])
    
    # last_available_date = None
    
    for current_date in full_date_range[1:]:
        if current_date in df_price.index:
            df_copy.loc[current_date, 'exist'] = 1
    
    df_copy['exist_change'] = 0
    df_copy.loc[df_copy['exist'] == df_copy['exist'].shift(1) - 1, 'exist_change'] = -1
    df_copy.loc[df_copy['exist'] == df_copy['exist'].shift(1) + 1, 'exist_change'] = 1
    
    df_minus_ones = df_copy.loc[df_copy['exist_change'] == -1].index
    df_ones = df_copy.loc[df_copy['exist_change'] == 1].index
    df_copy['status'] = 0
    if  df_ones[0] > df_minus_ones[0]:
        df_copy.loc[df_ones[0], 'status'] = df_copy.loc[df_ones[0], 'row_number'] - df_copy.loc[df_minus_ones[0], 'row_number'] 
    for dates in df_ones[1:]:
        df_minus_ones_iter_rows = df_minus_ones  < dates
        df_minus_ones_iter = df_minus_ones[df_minus_ones_iter_rows]
        df_copy.loc[dates, 'status'] = df_copy.loc[dates, 'row_number'] - df_copy.loc[df_minus_ones_iter[-1], 'row_number'] 
    
    df_copy.reset_index(inplace = True)
    df_price.reset_index(inplace = True)
    df_price_out = df_price.merge(df_copy[['date','status']], on = 'date', how = 'left')
    df_price_out.set_index('date',inplace = True)
    
    # df_price_out.loc[df_price_out['IPO'] == 1, 'status'] = -1
    
    return df_price_out

# Correcting price ranges and detecting ques and price modifications

def Clean_ranges_and_Detect_ques(price):
    
    price['ret1'] = price['vwap'].pct_change() 
    price['ret2'] = price['adj_vwap'].pct_change()    
    price['diff'] = np.abs(price['ret1'] - price['ret2'])    
    
    price['min_range'] = 100 * (price['day_range_min'] / price['vwap'].shift(1) - 1)
    price['max_range'] = 100 * (price['day_range_max'] / price['vwap'].shift(1) - 1)
    
    for i in price[price.index > '2019-11-05'].index:
        if pd.isna(price.loc[i, 'min_range']):
            for j in range(1,10):
                if pd.notna(price.shift(-j).loc[i, 'min_range']):
                    break
            if np.abs(price.shift(-j).loc[i, 'min_range'] - price.shift(1).loc[i, 'min_range']) < 1:
                price.loc[i, 'min_range'] = (price.shift(-j).loc[i, 'min_range'] + price.shift(1).loc[i, 'min_range']) / 2
                price.loc[i, 'max_range'] = (price.shift(-j).loc[i, 'max_range'] + price.shift(1).loc[i, 'max_range']) / 2
            else:
                price.loc[i, 'min_range'] = (price.shift(-j-1).loc[i, 'min_range'] + price.shift(-j-2).loc[i, 'min_range']) / 2
                price.loc[i, 'max_range'] = (price.shift(-j-1).loc[i, 'max_range'] + price.shift(-j-2).loc[i, 'max_range']) / 2
    
            price.loc[i, 'day_range_min'] = np.floor(price.shift(1).loc[i, 'vwap'] * (1 + price.loc[i, 'min_range']/100))
            price.loc[i, 'day_range_max'] = np.floor(price.shift(1).loc[i, 'vwap'] * (1 + price.loc[i, 'max_range']/100))    
    
    
    sell_que = []
    buy_que  = []
    
    for i in price[price.index > '2019-11-05'].index:
        if np.abs(price.loc[i, 'close']/price.loc[i, 'day_range_min'] - 1) < 0.0004:
            sell_que.append(i)
        if np.abs(price.loc[i, 'close']/price.loc[i, 'day_range_max'] - 1) < 0.0004:
            buy_que.append(i)
    
    price['que'] = 0
    price.loc[price.index.isin(buy_que), 'que']  = 1
    price.loc[price.index.isin(sell_que), 'que'] = -1

    price['prc_mdf'] = 0
    price.loc[price['diff'] < 1/price['close'],'prc_mdf'] = 1
    price.drop(columns = ['ret1','ret2','diff','min_range', 'max_range'],inplace = True)

    return price

# Analyze Numeric resistant and supports

def Mental_Borders_Analysis(price, col = 'close', thresh = 10000):

    price['border_down'] = np.floor(price[col] / 10 ** np.floor(np.log10(price[col])))      * (10**np.floor(np.log10(price[col])))
    price['border_up']   = (np.floor(price[col] / 10 ** np.floor(np.log10(price[col]))) + 1) * (10**np.floor(np.log10(price[col])))
    
    price.loc[price[col]>=thresh, 'border_down'] = np.floor(price[col] / 10 ** (np.floor(np.log10(price[col]))-1))        * (10**(np.floor(np.log10(price[col]))-1))
    price.loc[price[col]>=thresh, 'border_up']   =  (np.floor(price[col] / 10 ** (np.floor(np.log10(price[col]))-1) + 1))   * (10**(np.floor(np.log10(price[col]))-1))   
    
    price['upper_bound'] = price[col] / price['border_up']   - 1
    price['lower_bound'] = price[col] / price['border_down'] - 1
    
    price['break_borders'] = 0
    
    price.loc[price['border_up'] > price['border_up'].shift(1),'break_borders'] =  1
    price.loc[(price['border_up'] < price['border_up'].shift(1)) & (price['prc_mdf'] == 0),'break_borders'] = -1
    
    price = price.drop(columns = ['border_down','border_up'])
    return price
    
    
# Calculate Price Features

def calculate_price_features(price, col, df_index, modify):

    # Handle some error in price columns
    for clm in ['adj_open','adj_high','adj_low']:
            price.loc[(price[clm] == 0) & (price['adj_vwap'] * price['adj_close'] > 0), clm] = price[clm].shift(1)
    
    # Handling special days like closed days and post-IPO days, clean the days range, and find buy and sell ques  
    if modify['Close_days']:
        price = Find_Closed_IPO_Dates(price, df_index = df_index)

    if modify['ques']:
        price = Clean_ranges_and_Detect_ques(price)
    
    # Add Pivots  
    if modify['pivots']:
        price = find_pivots(df_price_daily = price, col = 'adj_close', pct_range = 0.05)

    # Add index Info
    df_index_llimited = df_index[['index_vwap','regime','currnt_reg_days']]
    price = price.reset_index().merge(df_index_llimited.reset_index(), on=['date'], how='left').set_index(['date'])
    price = price.rename(columns={'regime' : 'regime_market','currnt_reg_days':'currnt_reg_days_market'})
    
    # Add Regime Info    
    if modify['regime']:        
        price1, _ , _ = regime_detection(price = price, col = col, coefs_regime = [1.5,3.5,5,8])
        price = since_last_regime_change(price)        
     
    # Periods
    warnings.filterwarnings('ignore')
    
    if modify['tse_info']:
        week_number   = 5
        month_number  = 22
        season_number = 45
        year_number   = 200
        
        # Return
        price['return_close']     = price['adj_close'] / price['adj_close'].shift(1) - 1
        price['diff_close_vwap']  = price['adj_close'] / price['adj_vwap'] - 1
        price['open_return']      = price['adj_open']  / price['adj_close'].shift(1) - 1
        price['during_ret_close'] = price['return_close'] - price['open_return']
        
        # Candle
        # Daily
        price.loc[price['adj_high']<np.maximum(price['adj_open'], price['adj_close']), 'adj_high'] = np.maximum(price['adj_open'], price['adj_close'])
        price.loc[price['adj_low'] >np.minimum(price['adj_open'], price['adj_close']), 'adj_low']  = np.minimum(price['adj_open'], price['adj_close'])
        
        price['up_shadow']   = price['adj_high'] / np.maximum(price['adj_open'], price['adj_close']) - 1
        price['down_shadow'] = np.minimum(price['adj_open'], price['adj_close']) / price['adj_low'] - 1
        price['body']        = (price['adj_close'] - price['adj_open']) / price['adj_open']      
        
        # Weekly
        price['high_w']    = price['adj_high'].rolling(window = week_number).max()
        price['low_w']     = price['adj_low'].rolling(window = week_number).min()
        price['move_av_w'] = price['adj_vwap'].rolling(window = week_number).mean()
        price['open_w']    = price['adj_close'].shift(week_number+1)
        
        price['rizesh_w']  = price['high_w']     / price[col] - 1
        price['khizesh_w'] = price[col] / price['low_w']      - 1
        price['body_w']    = price[col] / price['open_w']     - 1
        price['profit_w']  = price[col] / price['move_av_w']  - 1
        
        # Monthly
        price['high_m']    = price['adj_high'].rolling(window = month_number).max()
        price['low_m']     = price['adj_low'].rolling(window = month_number).min()
        price['move_av_m'] = price['adj_vwap'].rolling(window = month_number).mean()
        price['open_m']    = price['adj_close'].shift(month_number+1)
        
        price['rizesh_m']  = price['high_m']   / price[col]  - 1
        price['khizesh_m'] = price[col] / price['low_m']     - 1
        price['body_m']    = price[col] / price['open_m']    - 1
        price['profit_m']  = price[col] / price['move_av_m'] - 1
        
        # Yearly
        price['high_y']    = price['adj_high'].rolling(window = season_number).max()
        price['low_y']     = price['adj_low'].rolling(window = season_number).min()
        price['move_av_y'] = price['adj_vwap'].rolling(window = season_number).mean()
        price['open_y']    = price['adj_close'].shift(season_number+1)
        
        price['rizesh_y']  = price['high_y'] / price[col]    - 1
        price['khizesh_y'] = price[col] / price['low_y']     - 1
        price['body_y']    = price[col] / price['open_y']    - 1
        price['profit_y']  = price[col] / price['move_av_y'] - 1            
            
        # Regime and Fluctuation
        price['std'] = np.sqrt(price[col].pct_change().rolling(window = 10).std())
        price['body_rolling'] = np.abs(price[col].pct_change().rolling(window = 10).sum()) / np.abs(price[col].pct_change()).rolling(window = 10).sum()
        
        # Ind_Ins
        price['money_in']   = (price['buy_ind_value'] - price['sell_ind_value']) / (10**10)
        price['sarane_ind'] = (price['buy_ind_value'] / price['buy_ind_count']) / (price['sell_ind_value']/price['sell_ind_count']) 
        price['money_in_w'] = (price['money_in'].rolling(window = week_number).sum()) 
        price['money_in_m'] = (price['money_in'].rolling(window = month_number).sum()) 
        
        # Vol/Val
        price['ave_val']   = price['value'] / (price['value'].rolling(window = month_number).mean())
        price['ave_val_w'] = price['value'].rolling(window = week_number).mean() / price['value'].rolling(window = month_number).mean()

        price = Mental_Borders_Analysis(price, col = 'close', thresh = 10000)
    
    warnings.simplefilter('default')
    
    # Technical indicators
    if modify['technical']:
        price, technical_list = add_technical_indicators(price, col = col)
    
    return price


# Create Output and Make it ready for creating X_train, X_test, y_train, and y_test

def calculate_price_output_and_preproccessing(price, lag_days, col, outlier = 0, normalization = False):
    
    # Removing Outliers and Normalization
    list_modify_outliers = ['return_close','return_vwap','open_return','during_ret_close','during_ret_vwap','up_shadow','down_shadow','body','rizesh_w','khizesh_w','body_w','rizesh_m','khizesh_m','body_m','rizesh_y','khizesh_y','body_y',
                            'diff_ShT_LT','diff_ShT','diff_LT','std','body_rolling','money_in','money_in_w','money_in_m','sarane_ind','ave_val','ave_val_w']
    
    if outlier>0:
        for i in list_modify_outliers:
            mean = price[i].mean()
            std = price[i].std()
            floor = mean - outlier * std
            cap = mean + outlier * std
            price[i] = price[i].clip(lower=floor, upper=cap)

    
    # Mormalization
    if normalization == 'z_score':
        for i in list_modify_outliers:
            price[i] = (price[i] - price[i].mean()) / price[i].std()
    elif normalization == 'min_max':
        for i in list_modify_outliers:
            price[i] = (price[i] - price[i].min()) / (price[i].max() - price[i].min())

    # Preparing output       
    warnings.filterwarnings('ignore')
    out_put_list = []
    for lag in lag_days:
        
        price['ret_abs_'+str(lag)+'_days'] = price[col].shift(-lag) / price['adj_close'] - 1
        price['ret_rel_'+str(lag)+'_days'] = price[col].shift(-lag) / price['adj_close'] - price['index_vwap'].shift(-lag) / price['index_vwap']
        
        price['label_binary_abs_'+str(lag)+'_days'] = 0
        price['label_binary_rel_'+str(lag)+'_days'] = 0
        price.loc[price['ret_abs_'+str(lag)+'_days'] > 0,'label_binary_abs_'+str(lag)+'_days'] = 1
        price.loc[price['ret_rel_'+str(lag)+'_days'] > 0,'label_binary_rel_'+str(lag)+'_days'] = 1

        out_put_list.append('ret_abs_'+str(lag)+'_days')
        out_put_list.append('ret_rel_'+str(lag)+'_days')
        out_put_list.append('label_binary_abs_'+str(lag)+'_days')
        out_put_list.append('label_binary_rel_'+str(lag)+'_days')
    
    warnings.simplefilter('default') 

    # Feature List for each Category    
    with open('features_dictionary.pkl', 'rb') as file:
        features_dictionary = pickle.load(file)
        
    list_feature = []
    list_feature.extend(features_dictionary['SHORT_TERM'])
    list_feature.extend(features_dictionary['LONG_TERM'])
    list_feature.extend(features_dictionary['ACTIVITY'])
    list_feature.extend(features_dictionary['REGIME'])
    list_feature.extend(features_dictionary['TECHNICAL'])
    list_feature.extend(features_dictionary['PIVOT'])
    list_feature.extend(features_dictionary['GENERAL'])
    list_feature.extend(features_dictionary['PRICE'])
    list_feature.extend(out_put_list)
    
    # Making price dataframe     
    price     = price[list_feature]     
    price     = price.sort_index()
    new_order = ['symbol'] + ['status'] + [clumn for clumn in price.columns if (clumn != 'symbol') & (clumn != 'status')]
    price     = price[new_order]

    return price


# Preparing Data for Learning

def Merge_and_Clean_DFs(lag_model, time_series = False):

    # Read data from saved sources and make them ready and correct names
    price_fund_labels = pd.read_parquet('price_with_fundamental_features_added.parquet')  
    price_rtrn_labels = pd.read_parquet('price_ready_after_Labeling.parquet')
    price_info        = pd.read_parquet('price_ready_for_Labeling.parquet')

    # delete rows with Infs
    numeric_price = price_info.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric_price)
    rows_with_inf = inf_mask.any(axis=1)
    price_info = price_info[~rows_with_inf]

    # Adding Label and Fundamental data into price_info dataframe and dropping NAN rows.
    price_info = price_info.reset_index().merge(price_fund_labels.reset_index(), on=['date', 'symbol'], how='left').set_index(['date'])  
    price_info = price_info.reset_index().merge(price_rtrn_labels.reset_index(), on=['date', 'symbol'], how='left').set_index(['date'])   
    
    # Adding E/P and S/P data.
    price_info['E/P'] = price_info['Net Income (Loss)'] / price_info['market_cap'] 
    price_info['S/P'] = price_info['Sales']             / price_info['market_cap']  

    del price_fund_labels, price_rtrn_labels 
    price_info.dropna(inplace = True)

    # Download features list
    with open('features_dictionary.pkl', 'rb') as file:
        features_dictionary = pickle.load(file)
    
    # if we prepare data for LSTM model there is not much to do here, we just return the selected features. We create lags in Make_Data_Ready_for_LSTM function.
    if time_series:        
        short_term_features  = features_dictionary['SHORT_TERM']
        long_term_features   = features_dictionary['LONG_TERM']
        activity_features    = features_dictionary['ACTIVITY']
        regime_features      = features_dictionary['REGIME']
        technical_features   = features_dictionary['TECHNICAL']
        pivot_features       = features_dictionary['PIVOT']
        fundamental_features = features_dictionary['FUND']
        general_features     = features_dictionary['GENERAL']

    # For Ensemble models, we should create lags here. 
    else:        
        short_term_features = [] 
        long_term_features  = [] 
        activity_features   = []
        regime_features     = []
        technical_features  = []
        pivot_features      = []
        general_features    = []
        fundamental_features = features_dictionary['FUND']

        # Creating the list of lags.
        for key, value in lag_model.items():
            for i in range(1,value + 1):
                if key in features_dictionary['SHORT_TERM']:
                    short_term_features.append(key+'_L'+str(i))
                if key in features_dictionary['LONG_TERM']:
                    long_term_features.append(key+'_L'+str(i))
                if key in features_dictionary['ACTIVITY']:
                    activity_features.append(key+'_L'+str(i))
                if key in features_dictionary['REGIME']:
                    regime_features.append(key+'_L'+str(i))
                if key in features_dictionary['TECHNICAL']:
                    technical_features.append(key+'_L'+str(i))
                if key in features_dictionary['PIVOT']:
                    pivot_features.append(key+'_L'+str(i))
                if key in features_dictionary['GENERAL']:
                    general_features.append(key+'_L'+str(i))

        # Adding lags to price_info dataframe.
        P_whole = pd.DataFrame()        
        
        for sym in price_info['symbol'].unique().tolist()[0:]:
            P_stock = price_info[price_info['symbol'] == sym]            
            
            for key, value in lag_model.items():
                if key == 'vwap':
                    for i in range(1,value + 1):                          
                        P_stock[key+'_L'+str(i)] = P_stock['adj_vwap'].shift(i) / P_stock['adj_vwap'] - 1
                elif key == 'close':
                    for i in range(1,value + 1):                          
                        P_stock[key+'_L'+str(i)] = P_stock['adj_close'].shift(i) / P_stock['adj_close'] - 1
                else:
                    for i in range(1,value + 1):                
                        if i > 1:
                            P_stock[key+'_L'+str(i)] = P_stock[key].shift(i-1)
                        else:
                            P_stock[key+'_L'+str(i)] = P_stock[key]

            # To not have NANs because of creating Lags
            P_stock = P_stock[10:]
            
            P_whole = pd.concat([P_whole, P_stock])
            
        price_info = P_whole
        Features_created_with_lags = {'short':short_term_features, 'long':long_term_features, 'activity':activity_features, 'regime': regime_features, 'technical':technical_features,
                                      'pivot':pivot_features, 'general':general_features, 'fund': fundamental_features}
        
        return price_info, Features_created_with_lags

def Making_XY_TrainTest(feat_dict, feat_list_created, price_info, lag_days, lag_labels, time_series, split = 0.8, QC_check = False):
    
    # dividing price_info into p_train and p_test
    price_info = price_info.sort_index()
    sample_size = price_info.shape[0]
    seperation_date = price_info.reset_index()['date'].iloc[int(np.floor(price_info.shape[0]*split))]
    p_train = price_info.loc[price_info.index < seperation_date]
    p_test  = price_info.loc[price_info.index >= seperation_date]

    # setting X and y columns
    
    y_list = ['symbol','status','regime_L1','regime_market_L1','que','prc_mdf','period_ending_date']
    
    for lag in lag_days:
        y_list.append('label_binary_rel_'+str(lag)+'_days')
        y_list.append('label_binary_abs_'+str(lag)+'_days')
        y_list.append('ret_abs_'+str(lag)+'_days')
        y_list.append('ret_rel_'+str(lag)+'_days')

    for lag in lag_labels:
        y_list.append('label_multi_classes_' + str(lag) + '_days')
        
    if time_series:
        y_list.extend(['regime','regime_market'])
        y_list.remove('regime_L1')
        y_list.remove('regime_market_L1')
        

    features = ['symbol','status']    
    if feat_dict['price_Features_Short_term']:
        features.extend(feat_list_created['short'])
        
    if feat_dict['price_Features_Long_term']:
        features.extend(feat_list_created['long'])  
        
    if feat_dict['technical_features']:
        features.extend(feat_list_created['technical'])  
        
    if feat_dict['activity_features']:
        features.extend(feat_list_created['activity']) 
        
    if feat_dict['fundamental_features']:
        features.extend(feat_list_created['fund'])
        
    if feat_dict['regime_features']:
        features.extend(feat_list_created['regime']) 
        
    if feat_dict['pivot_features']:
        features.extend(feat_list_created['pivot'])
        
    if feat_dict['general_info']:
        features.extend(feat_list_created['general'])
  
    # setting X_train, X_test, y_train, y_test based on their determined columns
    p_train = p_train[features + y_list].dropna()
    p_test  = p_test[features + y_list].dropna()
    
    y_train = p_train[y_list]
    y_test  = p_test[y_list]
    X_train = p_train[features]
    X_test  = p_test[features]

    # # Quality Check of Data
    if QC_check:
        Quality_check(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test, seperation_date
    
    
# Checking the Quality of Data

def Quality_check(X_train, X_test, y_train, y_test):
    # Check for missing values in labels
    if pd.isnull(y_train['label']).any() or pd.isnull(y_test['label']).any():
        raise ValueError("Missing values found in labels")
    
    # Check for missing values in features
    if pd.isnull(X_train.iloc[:, 1:]).any().any() or pd.isnull(X_test.iloc[:, 1:]).any().any():
        raise ValueError("Missing values found in features")
    
    # Check for non-numeric data in features
    if not all(np.issubdtype(dtype, np.number) for dtype in X_train.iloc[:, 1:].dtypes):
        raise ValueError("Non-numeric data found in features")
    
    if np.isinf(X_train.iloc[:, 1:]).any().any() or np.isinf(X_test.iloc[:, 1:]).any().any():
        raise ValueError("Infinite values found in features")
    
    # Check for infinite values in features
    if np.isinf(X_train.iloc[:, 1:]).any().any() or np.isinf(X_test.iloc[:, 1:]).any().any():
        raise ValueError("Infinite values found in features")
    
    # Check for extreme outliers or anomalies in the data
    print("Checking for extreme outliers in the training data...")
    outlier_threshold = 5
    # Example threshold for z-score
    z_scores = np.abs((X_train.iloc[:, 1:] -X_train.iloc[:, 1:].mean()) / X_train.iloc[:, 1:].std())
    outliers = (z_scores > outlier_threshold).any(axis=1)
    if outliers.sum() > 0:
        print(f"Warning: {outliers.sum()} outliers detected in the training data.")
        

# Creating Lags for Models like LSTM

def create_dataset(data, target, lags):
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i])
        y.append(target[i])
    return np.array(X), np.array(y)


# Make Data Ready For Time series Models like LSTM and Transformers

def Make_Data_Ready_for_LSTM(X_train, X_test, y_train, y_test, label='label_four', lags=15, norm = 'minmax'):

    # For simplicity, just retain stocks that existed before seperation date
    X_test = X_test[X_test['symbol'].isin(X_train['symbol'].unique().tolist())]
    y_test = y_test[y_test['symbol'].isin(y_train['symbol'].unique().tolist())]
    
    # initialize combined_X_train, combined_X_test, combined_y_train, combined_y_test
    combined_X_train = []
    combined_y_train = []
    combined_X_test = []
    combined_y_test = []

    X_test_ts = pd.DataFrame()
    y_test_ts = pd.DataFrame()
    
    # for each stock create lags and normalization seperately.
    for sym in X_train['symbol'].unique().tolist():
        # seperate each stock data
        X_train_stock = X_train[X_train['symbol'] == sym].drop(columns=['symbol'])
        X_test_stock  = X_test[X_test['symbol'] == sym].drop(columns=['symbol'])
        
        y_train_stock = y_train[y_train['symbol'] == sym].drop(columns=['symbol'])
        y_test_stock  = y_test[y_test['symbol'] == sym].drop(columns=['symbol'])

        # doing the normalization for each stock seperately.
        if norm == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_stock = scaler.fit_transform(X_train_stock)
            X_test_stock = scaler.fit_transform(X_test_stock)
        
        # doing the required y transformations.
        label_encoder = LabelEncoder()
        if label == 'label_four':
            y_train_encoded = label_encoder.fit_transform(y_train_stock['label_four'] - 1)
            y_test_encoded  = label_encoder.fit_transform(y_test_stock['label_four'] - 1)
        elif label == 'label':
            y_train_encoded = label_encoder.fit_transform(y_train_stock['label_rel'])
            y_test_encoded  = label_encoder.fit_transform(y_test_stock['label_rel'])
        
        onehot_encoder = OneHotEncoder(sparse_output=False)
        y_train_encoded = y_train_encoded.reshape(len(y_train_encoded), 1) 
        y_test_encoded  = y_test_encoded.reshape(len(y_test_encoded), 1)
        
        onehot_y_train = onehot_encoder.fit_transform(y_train_encoded)
        onehot_y_test  = onehot_encoder.fit_transform(y_test_encoded)

        # Create X lags by create_dataset functions.
        X_train_stock, y_train_stock = create_dataset(X_train_stock, onehot_y_train, lags)
        X_test_stock,  y_test_stock  = create_dataset(X_test_stock, onehot_y_test, lags)

        if X_train_stock.shape[0] == 0 or X_test_stock.shape[0] == 0:
            continue

        combined_X_train.append(X_train_stock)
        combined_y_train.append(y_train_stock)
        combined_X_test.append(X_test_stock)
        combined_y_test.append(y_test_stock)        

        X_test_ts_stock = X_test[X_test['symbol'] == sym]
        X_test_ts_stock = X_test_ts_stock[lags:]
        y_test_ts_stock = y_test[y_test['symbol'] == sym]
        y_test_ts_stock = y_test_ts_stock[lags:]

        # Check any mismatch
        if (y_test_ts_stock.shape[0] != y_test_stock.shape[0]):
            print(sym)
            print(str(y_test_ts_stock.shape))
            print(str(y_test_stock.shape))

        X_test_ts = pd.concat([X_test_ts, X_test_ts_stock], axis = 0)
        y_test_ts = pd.concat([y_test_ts, y_test_ts_stock], axis = 0)
    
    # Create outputs
    combined_X_train = np.concatenate(combined_X_train)
    combined_y_train = np.concatenate(combined_y_train)
    combined_X_test = np.concatenate(combined_X_test)
    combined_y_test = np.concatenate(combined_y_test)
    
    combined_X_train = combined_X_train.reshape((combined_X_train.shape[0], combined_X_train.shape[1], combined_X_train.shape[2]))
    combined_X_test = combined_X_test.reshape((combined_X_test.shape[0], combined_X_test.shape[1], combined_X_test.shape[2]))

    return combined_X_train, combined_X_test, combined_y_train, combined_y_test, y_test_ts, X_test_ts, label_encoder


# Analyzing the output

def assign_label(row, quantiles_by_date, labels, col):
    try:
        bins = quantiles_by_date.loc[row.name].values
        unique_bins = pd.unique(bins)
        return pd.cut([row[col]], bins=unique_bins, labels=labels, include_lowest=True, duplicates='drop')[0]
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return None


def Analyze_output(y_proba_df, y_test, remove_ques, pred_is_binary, actual_is_binary, pred_label_based_on, actual_label_based_on, return_based_on):

    y_proba_df['symbol'] = y_test['symbol']
    y_proba_df['que'] = y_test['que']  
    y_proba_df[actual_label_based_on] = y_test[actual_label_based_on]
    y_proba_df[return_based_on] = y_test[return_based_on]

    labels = [1, 2, 3, 4]
    if pred_is_binary:
        y_proba_df = y_proba_df.groupby(y_proba_df.index).filter(lambda x: len(x) >= 0)
        quantiles_by_date_pred = y_proba_df.groupby(y_proba_df.index)[pred_label_based_on].quantile([0, 0.25, 0.5, 0.75, 1]).unstack()
        y_proba_df['label_predicted'] = y_proba_df.apply(lambda row: assign_label(row, quantiles_by_date_pred, labels, col = pred_label_based_on), axis=1)
    else:
        y_proba_df['label_predicted'] = y_proba_df[pred_label_based_on]
        y_proba_df['label_predicted'] = y_proba_df['label_predicted'] + 1

    if actual_is_binary: 
        quantiles_by_date_Ret = y_proba_df.groupby(y_proba_df.index)[actual_label_based_on].quantile([0, 0.25, 0.5, 0.75, 1]).unstack()
        y_proba_df['label_happened'] = y_proba_df.apply(lambda row: assign_label(row, quantiles_by_date_Ret, labels, col = 'return'), axis=1)
    else:
        y_proba_df['label_happened'] = y_proba_df[actual_label_based_on]
    
    y_proba_df['error'] = np.abs(y_proba_df['label_happened'] - y_proba_df['label_predicted'])

    very_highs, highs, lows, very_lows  = [], [], [], []
    very_highs_num, highs_num, lows_num, very_lows_num = [], [], [], [] 
    compare_high_low, Absolute_Error = [], []

    for i in y_proba_df.index.unique():
        if remove_ques:
            very_high  = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 4) & (y_proba_df['que'] != 1)][return_based_on].mean()
            high       = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 3) & (y_proba_df['que'] != 1)][return_based_on].mean()
            low        = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 2) & (y_proba_df['que'] != -1)][return_based_on].mean()
            very_low   = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 1) & (y_proba_df['que'] != -1)][return_based_on].mean()

            very_high_num = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 4) & (y_proba_df['que'] != 1)].shape[0]
            high_num      = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 3) & (y_proba_df['que'] != 1)].shape[0]
            low_num       = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 2) & (y_proba_df['que'] != -1)].shape[0]
            very_low_num  = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 1) & (y_proba_df['que'] != -1)].shape[0]
            
        else:            
            very_high  = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 4)][return_based_on].mean()
            high       = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 3)][return_based_on].mean()
            low        = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 2)][return_based_on].mean()
            very_low   = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 1)][return_based_on].mean()
    
            very_high_num = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 4)].shape[0]
            high_num      = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 3)].shape[0]
            low_num       = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 2)].shape[0]
            very_low_num  = y_proba_df[(y_proba_df.index == i) & (y_proba_df['label_predicted'] == 1)].shape[0]

        absolute_error = y_proba_df.loc[y_proba_df.index == i, 'error'].sum() / y_proba_df.loc[y_proba_df.index == i, 'error'].shape[0]
    
        comparing = 0
        if very_high > very_low:
            comparing = comparing + 1
        if high > low:
            comparing = comparing + 1
        if low > very_low:
            comparing = comparing + 1        
        compare_high_low.append(comparing/3)
    
        
        very_highs.append(very_high)
        highs.append(high)
        lows.append(low)
        very_lows.append(very_low)
        very_highs_num.append(very_high_num)
        highs_num.append(high_num)
        lows_num.append(low_num)
        very_lows_num.append(very_low_num)
        Absolute_Error.append(absolute_error)
        
    compare_df = pd.DataFrame({'Very_Highs': very_highs, 'Highs': highs, 'Lows': lows, 'Very_Lows': very_lows, 
                               'Very_Highs_N': very_highs_num, 'Highs_N': highs_num, 'Lows_N': lows_num, 'Very_Lows_N': very_lows_num,
                               'Compare': compare_high_low, 'Absolute_Error': Absolute_Error}, index=y_proba_df.index.unique())
    return compare_df, y_proba_df


def show_output(compare_df, y_proba_df, confusion_matrix = False):
    
    print('compare Classes = ' + f"{compare_df['Compare'].mean()*100:.2f}%")
    
    print('')
    
    print('return very_high is =   ' + f"{compare_df['Very_Highs'].mean()*100:.2f}" + ' for ' + f"{compare_df['Very_Highs_N'].mean():.0f}" + ' sample each day')
    print('return high is      =   ' + f"{compare_df['Highs'].mean()*100:.2f}" + ' for ' + f"{compare_df['Highs_N'].mean():.0f}" + ' sample each day')
    print('return low is       =   ' + f"{compare_df['Lows'].mean()*100:.2f}" + ' for ' + f"{compare_df['Lows_N'].mean():.0f}" + ' sample each day')
    print('return very low is  =   ' + f"{compare_df['Very_Lows'].mean()*100:.2f}" + ' for ' + f"{compare_df['Very_Lows_N'].mean():.0f}" + ' sample each day')
    
    print('')
    print('Bar_chart:')
    a = [compare_df['Very_Lows'].mean()* 100, compare_df['Lows'].mean()* 100, compare_df['Highs'].mean()* 100, compare_df['Very_Highs'].mean()* 100] 
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(a)), a, tick_label=range(1, len(a) + 1))
    plt.xlabel('Class')
    # plt.ylabel('Average Daily Return %')
    # plt.title('Multi-Class Classification Result')
    # plt.savefig('bar_chart.png')
    plt.show()
    plt.close()

    if confusion_matrix:
        conf_matrix = confusion_matrix(y_proba_df['label_happened'], y_proba_df['label_predicted'], labels=[1, 2, 3, 4])
        conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        # Plotting the confusion matrix with percentages
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('seperating_models')
        plt.savefig('confusionMatrix_seperating_models.png')
        plt.show()
        plt.close()
        
        
# Convert Jalali to Miladi

def convert_to_gregorian(shamsi_date):
    shamsi_date = shamsi_date.replace('/', '-')  # Replace '/' with '-' for compatibility
    shamsi_date = JalaliDate.fromisoformat(shamsi_date)
    return shamsi_date.to_gregorian().strftime('%Y-%m-%d')

# Correction the name of Income statement Items

def correct_name(df):

    unique_items = df.columns
    if 'Cost Of Revenue' in unique_items:
        df['Cost of Sales'] = df['Cost Of Revenue']
        df = df.drop(columns=['Cost Of Revenue'])

    if 'Service Revenue' in unique_items:
        df['Sales'] = df['Service Revenue']
        df = df.drop(columns=['Service Revenue'])
        
    return df


# Calculating Fundamental Features

def calculate_fundamental_features(df_return_whole):
        
    df_return_whole['change_sales'] = (df_return_whole['Sales'] - df_return_whole['Sales'].shift(1)) / np.abs(df_return_whole['Sales'].shift(1))
    df_return_whole['change_gross'] = (df_return_whole['Gross Profit (Loss)'] - df_return_whole['Gross Profit (Loss)'].shift(1)) / np.abs(df_return_whole['Gross Profit (Loss)'].shift(1))
    df_return_whole['change_operating'] = (df_return_whole['Operating Profit (Loss)'] - df_return_whole['Operating Profit (Loss)'].shift(1)) / np.abs(df_return_whole['Operating Profit (Loss)'].shift(1))
    df_return_whole['change_net'] = (df_return_whole['Net Income (Loss)'] - df_return_whole['Net Income (Loss)'].shift(1)) / np.abs(df_return_whole['Net Income (Loss)'].shift(1))
    
    df_return_whole.loc[df_return_whole['Sales'].shift(1) == 0, 'change_sales'] = (df_return_whole['Sales'] - df_return_whole['Sales'].shift(1)) / np.abs(df_return_whole['Sales'])
    df_return_whole.loc[df_return_whole['Gross Profit (Loss)'].shift(1) == 0, 'change_gross'] = (df_return_whole['Gross Profit (Loss)'] - df_return_whole['Gross Profit (Loss)'].shift(1)) / np.abs(df_return_whole['Gross Profit (Loss)'])
    df_return_whole.loc[df_return_whole['Operating Profit (Loss)'].shift(1) == 0, 'change_operating'] = (df_return_whole['Operating Profit (Loss)'] - df_return_whole['Operating Profit (Loss)'].shift(1)) / np.abs(df_return_whole['Operating Profit (Loss)'])
    df_return_whole.loc[df_return_whole['Net Income (Loss)'].shift(1) == 0, 'change_net'] = (df_return_whole['Net Income (Loss)'] - df_return_whole['Net Income (Loss)'].shift(1)) / np.abs(df_return_whole['Net Income (Loss)'])
    
    df_return_whole.loc[(df_return_whole['Sales'].shift(1) == 0) & (df_return_whole['Sales'] == 0), 'change_sales'] = 0
    df_return_whole.loc[(df_return_whole['Gross Profit (Loss)'].shift(1) == 0) & (df_return_whole['Gross Profit (Loss)'] == 0), 'change_gross'] = 0
    df_return_whole.loc[(df_return_whole['Operating Profit (Loss)'].shift(1) == 0) & (df_return_whole['Operating Profit (Loss)'] == 0), 'change_operating'] = 0
    df_return_whole.loc[(df_return_whole['Net Income (Loss)'].shift(1) == 0) & (df_return_whole['Net Income (Loss)'] == 0), 'change_net'] = 0
    
    # Change Margin_calculation
    df_return_whole['gross_ratio_change'] = (df_return_whole['Gross Profit (Loss)'] / df_return_whole['Sales']) - ((df_return_whole['Gross Profit (Loss)'].shift(1) / df_return_whole['Sales'].shift(1)))
    df_return_whole['oprtg_ratio_change'] = (df_return_whole['Operating Profit (Loss)'] / df_return_whole['Sales']) - ((df_return_whole['Operating Profit (Loss)'].shift(1) / df_return_whole['Sales'].shift(1)))
    df_return_whole['net_ratio_change']   = (df_return_whole['Net Income (Loss)'] / df_return_whole['Sales']) - ((df_return_whole['Net Income (Loss)'].shift(1) / df_return_whole['Sales'].shift(1)))
    
    df_return_whole.loc[(df_return_whole['Sales'] == 0) | ((df_return_whole['Sales'].shift(1) == 0)), 'gross_ratio_change'] = 0
    df_return_whole.loc[(df_return_whole['Sales'] == 0) | ((df_return_whole['Sales'].shift(1) == 0)), 'oprtg_ratio_change'] = 0
    df_return_whole.loc[(df_return_whole['Sales'] == 0) | ((df_return_whole['Sales'].shift(1) == 0)), 'net_ratio_change'] = 0

    df_return_whole['gross_margin'] = df_return_whole['Gross Profit (Loss)'] / df_return_whole['Sales']
    df_return_whole['operating_margin'] = df_return_whole['Operating Profit (Loss)'] / df_return_whole['Sales']
    df_return_whole['net_margin'] = df_return_whole['Net Income (Loss)'] / df_return_whole['Sales']

    df_return_whole.loc[df_return_whole['Sales'] == 0, 'gross_margin'] = 0 
    df_return_whole.loc[df_return_whole['Sales'] == 0, 'operating_margin'] = 0 
    df_return_whole.loc[df_return_whole['Sales'] == 0, 'net_margin'] = 0 
    
    return df_return_whole


# Preparing DataSets for Fundamental Analysis

def Calculate_fundamental_state(df_price_daily, df):

    # simple cleaning
    
    df = df.set_index(['period_ending_date','publish_date','symbol'])
    important_items = ['Sales', 'Cost of Sales', 'Gross Profit (Loss)', 'Sales, General and Administrative Expense', 'Other Operating Revenue (Expenses)', 'Operating Profit (Loss)',  
                       'Finance Expense', 'Income Tax Expense', 'Net Income (Loss)']
    
    df_pivot_fund = df.pivot(columns = 'statement_item_name', values ='value_origin')
    df_pivot_fund = correct_name(df_pivot_fund)
    df_pivot_fund = df_pivot_fund.rename(columns={'فروش': 'Sales', 'بهای تمام شده کالای فروش رفته': 'Cost of Sales', 'سود (زیان) ناخالص': 'Gross Profit (Loss)', 'هزینه های عمومی, اداری و تشکیلاتی': 'Sales, General and Administrative Expense', 
                                                  'خالص سایر درامدها (هزینه ها) ی عملیاتی': 'Other Operating Revenue (Expenses)', 'سود (زیان) عملیاتی': 'Operating Profit (Loss)', 'خالص سایر درآمدها و هزینه‌های غیرعملیاتی': 'Other Non Operating Revenue (Expenses)', 
                                                  'هزینه‌های مالی': 'Finance Expense', 'مالیات': 'Income Tax Expense', 'سود (زیان) خالص': 'Net Income (Loss)'})
    
    df_pivot_fund = df_pivot_fund[important_items].sort_index(level='period_ending_date')
    df_pivot_fund = df_pivot_fund.dropna(subset=important_items)
    df_pivot_fund = df_pivot_fund[~df_pivot_fund.index.get_level_values('period_ending_date').duplicated(keep='last')]

    # Removing dates before missing reports
    df_pivot_fund['period_ending_date_diff'] = df_pivot_fund.reset_index()['period_ending_date'].diff().dt.days.values
    df_pivot_fund['missing'] = df_pivot_fund['period_ending_date_diff'].apply(lambda x: False if 80 <= x <= 100 else True)
    df_pivot_fund.loc[pd.isna(df_pivot_fund['period_ending_date_diff']), 'missing'] = False

    if True in df_pivot_fund['missing'].unique():
        idx_last_True = df_pivot_fund[df_pivot_fund['missing'] == True].index[-1]
        df_pivot_fund = df_pivot_fund.loc[idx_last_True:]
    
    # Adding Margins (Copy and Paste from 'calculate_fundamental_features' function. we dont use it because it needs price info and ...)
    df_pivot_fund = calculate_fundamental_features(df_pivot_fund)
    margins_list = ['change_sales','change_gross','change_operating','change_net','gross_ratio_change','oprtg_ratio_change','net_ratio_change','gross_margin','operating_margin','net_margin']
    
    # Calculating long term and short term fundamental features
    warnings.filterwarnings('ignore')
    df_pivot_fund = set_fundamental_margin_tags(analysis = df_pivot_fund, column_name = 'gross_margin', LT_period = 5, type = 'gross')
    df_pivot_fund = set_fundamental_margin_tags(analysis = df_pivot_fund, column_name = 'net_margin', LT_period = 5, type = 'net')
    df_pivot_fund = set_fundamental_growth_tags(analysis = df_pivot_fund , LT_period = 3)
    warnings.simplefilter('default')       
    
    features_eng = ['net_margin_short_term','gross_margin_short_term','net_margin_long_term','gross_margin_long_term','net_growth_ma','gross_growth_ma','sales_growth_ma']
    df_pivot_fund = df_pivot_fund.dropna(subset=features_eng)
    
    # Find first date after report release and set it as index 
    df_price_daily.reset_index(inplace = True) 

    df_pivot_fund['publish_day'] = df_pivot_fund.index.get_level_values('publish_date').date
    df_pivot_fund['publish_day'] = pd.to_datetime(df_pivot_fund['publish_day'], errors='coerce')
    df_pivot_fund = df_pivot_fund.reset_index().sort_values(by='period_ending_date', ascending=True).set_index('publish_day')
    
    for specified_date in df_pivot_fund.index: 
        filtered_df = df_price_daily[df_price_daily['date'] > specified_date]
        first_date_after_specified_date = filtered_df['date'].min()
        df_pivot_fund.loc[specified_date,'date'] = first_date_after_specified_date

    # Merge data and make the augemented price dataframe ready
    df_price_daily = df_price_daily.drop(columns = ['symbol']).merge(df_pivot_fund.reset_index().drop_duplicates(subset='date', keep='last'), on=['date'], how='left').set_index(['date'])
    df_price_daily = df_price_daily[df_price_daily.index >= df_pivot_fund['date'].iloc[0]]
    df_price_daily = df_price_daily[['symbol','period_ending_date','publish_date'] + features_eng + important_items + margins_list]
    
    df_price_daily['publish'] = 0
    df_price_daily['publish1'] = 0
    df_price_daily.loc[pd.notna(df_price_daily['net_margin_long_term']), 'publish'] = 2
    for i in df_price_daily[df_price_daily['publish'] == 2].index:
        df_price_daily.loc[i+timedelta(1):i+timedelta(30),'publish1'] = 1

    df_price_daily.loc[(df_price_daily['publish1'] == 1) & (df_price_daily['publish'] == 2), 'publish'] = 2
    df_price_daily.loc[(df_price_daily['publish1'] == 1) & (df_price_daily['publish'] != 2), 'publish'] = 1
    

    df_price_daily['nrows'] = np.arange(1, len(df_price_daily) + 1) - 1
    twos = df_price_daily.loc[df_price_daily['publish'] == 2,'nrows'].tolist()
    twos.append(df_price_daily['nrows'].iloc[-1])

    df_price_daily['tilllast_report'] = 0
    
    for i in range(0,len(twos)-1):
        for j in range(twos[i]+1,twos[i+1]):
            df_price_daily.loc[df_price_daily['nrows'] == j, 'tilllast_report'] = j - twos[i]
    
                
    df_price_daily = df_price_daily.ffill().dropna()
    df_price_daily = df_price_daily.drop(columns = ['publish1','nrows'])
    
    return df_pivot_fund, df_price_daily

# Create Fundamental tags based on Grwoth and Margins

def set_fundamental_growth_tags(analysis, LT_period):
        
    analysis['Sales_MA'] = 0
    analysis['GROSS_MA'] = 0
    analysis['NET_MA']   = 0

    analysis['sales_growth_ma'] = 0
    analysis['gross_growth_ma'] = 0
    analysis['net_growth_ma']   = 0
    
    for i in range(analysis.shape[0]):
        Sales_MA = 0
        Gross_MA = 0
        Net_MA   = 0
        div = 0
        
        if i < LT_period:                    
            for j in range(i+1):
                Sales_MA = Sales_MA + analysis['Sales'].iloc[i-j]               * np.exp(-j/3)
                Gross_MA = Gross_MA + analysis['Gross Profit (Loss)'].iloc[i-j] * np.exp(-j/3)
                Net_MA   = Net_MA   + analysis['Net Income (Loss)'].iloc[i-j]   * np.exp(-j/3) 
                div  = div  + np.exp(-j/3)
        
        else:
            for j in range(LT_period):
                Sales_MA = Sales_MA + analysis['Sales'].iloc[i-j]               * np.exp(-j/3)
                Gross_MA = Gross_MA + analysis['Gross Profit (Loss)'].iloc[i-j] * np.exp(-j/3)
                Net_MA   = Net_MA   + analysis['Net Income (Loss)'].iloc[i-j]   * np.exp(-j/3) 
                div  = div  + np.exp(-j/3)

        analysis['Sales_MA'].iloc[i]  = Sales_MA / div
        analysis['GROSS_MA'].iloc[i]  = Gross_MA / div
        analysis['NET_MA'].iloc[i]    = Net_MA   / div

        analysis['sales_growth_ma'].iloc[i]  = min(max(int(20 * (analysis['Sales'].iloc[i]               - analysis['Sales_MA'].iloc[i]) / np.abs(analysis['Sales_MA'].iloc[i]))  ,-10),10)
        analysis['gross_growth_ma'].iloc[i]  = min(max(int(20 * (analysis['Gross Profit (Loss)'].iloc[i] - analysis['GROSS_MA'].iloc[i]) / np.abs(analysis['GROSS_MA'].iloc[i]))  ,-10),10)
        analysis['net_growth_ma'].iloc[i]    = min(max(int(20 * (analysis['Net Income (Loss)'].iloc[i]   - analysis['NET_MA'].iloc[i])   / np.abs(analysis['NET_MA'].iloc[i]) / 2),-10),10)

        if analysis['Gross Profit (Loss)'].iloc[i] * analysis['Gross Profit (Loss)'].iloc[i-1] < 0:
            analysis['gross_growth_ma'].iloc[i] = 10 * np.sign(analysis['Gross Profit (Loss)'].iloc[i])
            
        if analysis['Net Income (Loss)'].iloc[i] * analysis['Net Income (Loss)'].iloc[i-1] < 0:
            analysis['net_growth_ma'].iloc[i] = 10 * np.sign(analysis['Net Income (Loss)'].iloc[i])
            
    return analysis

def set_fundamental_margin_tags(analysis, column_name, LT_period, type):
    
    if type == 'gross':
        thresh = gross_margin_thresh = [-float('inf'), 0, 0.10, 0.30, 0.5, float('inf')]
    if type == 'net':
        thresh = net_margin_thresh =   [-float('inf'), 0, 0.10, 0.25, 0.3, float('inf')]
    
    analysis[type + 'tag'] = pd.cut(analysis[column_name], bins=thresh, labels=[1, 2, 3, 4, 5])

    output_col = 'long_term_analysis'
    analysis[output_col]                = 0
    analysis[type+'_wgtd']              = 0
    analysis[type+'_margin_short_term'] = 0
    analysis[type+'_margin_long_term']  = 0
    
    analysis['row_number'] = np.arange(1, len(analysis) + 1)
    
    for i in range(analysis.shape[0]):
        coef = 0
        div = 0
        if i < LT_period:        
            count_of_a = analysis[type + 'tag'].iloc[0:i+1].value_counts().get(1, 0) / (i+1)
            count_of_b = analysis[type + 'tag'].iloc[0:i+1].value_counts().get(2, 0) / (i+1)
            count_of_c = analysis[type + 'tag'].iloc[0:i+1].value_counts().get(3, 0) / (i+1)
            count_of_d = analysis[type + 'tag'].iloc[0:i+1].value_counts().get(4, 0) / (i+1)
            count_of_e = analysis[type + 'tag'].iloc[0:i+1].value_counts().get(5, 0) / (i+1)
            
            if type == 'gross':
                for j in range(i+1):
                    coef = coef + analysis[type + '_margin'].iloc[i-j] * np.exp(-j/4)
                    div  = div  + np.exp(-j/4)   
            if type == 'net':
                for j in range(i+1):
                    coef = coef + analysis[type + '_margin'].iloc[i-j] * np.exp(-j/2)
                    div  = div  + np.exp(-j/2)             
                        
        else:
            count_of_a = analysis[type + 'tag'].iloc[i-LT_period:i+1].value_counts().get(1, 0) / (LT_period)
            count_of_b = analysis[type + 'tag'].iloc[i-LT_period:i+1].value_counts().get(2, 0) / (LT_period)
            count_of_c = analysis[type + 'tag'].iloc[i-LT_period:i+1].value_counts().get(3, 0) / (LT_period)
            count_of_d = analysis[type + 'tag'].iloc[i-LT_period:i+1].value_counts().get(4, 0) / (LT_period)
            count_of_e = analysis[type + 'tag'].iloc[i-LT_period:i+1].value_counts().get(5, 0) / (LT_period) 

            if type == 'gross':
                for j in range(LT_period):
                    coef = coef + analysis[type + '_margin'].iloc[i-j] * np.exp(-j/4)
                    div  = div  + np.exp(-j/4)
            if type == 'net':
                 for j in range(LT_period):
                    coef = coef + analysis[type + '_margin'].iloc[i-j] * np.exp(-j/2)
                    div  = div  + np.exp(-j/2)
            
        moving_average = coef / div
        ratio = (analysis[type + '_margin'].iloc[i] - moving_average) / np.abs(moving_average)
        
        if type == 'gross':
            if i>2:
                if (count_of_a >= 0.6):
                    analysis[output_col].iloc[i] = 1  
                    
                elif (count_of_a >= 0.3) | (analysis[type + 'tag'].iloc[i] == 1):
                    analysis[output_col].iloc[i] = 2   
                    
                elif (count_of_b + count_of_a >= 0.5) | (analysis[type + 'tag'].iloc[i-1] == 1):
                    analysis[output_col].iloc[i] = 3    
                    
                elif (count_of_b + count_of_c >= 0.6):
                    analysis[output_col].iloc[i] = 4
                    
                elif (count_of_b + count_of_c > 0.3):
                    analysis[output_col].iloc[i] = 5    
                    
                else:
                    analysis[output_col].iloc[i] = 6
            else:
                if count_of_a>=1:
                    analysis[output_col].iloc[i] = 1
                    
                elif count_of_b>=1:
                    analysis[output_col].iloc[i] = 2
                    
                elif count_of_c>=1:
                    analysis[output_col].iloc[i] = 3
                    
                else:
                    analysis[output_col].iloc[i] = 4

        if type == 'net':
            if i>2:
                if (count_of_a >= 0.6):
                    analysis[output_col].iloc[i] = 1  
                    
                elif (count_of_a >= 0.3) | (analysis[type + 'tag'].iloc[i] == 1):
                    analysis[output_col].iloc[i] = 2   
                    
                elif (count_of_b + count_of_a >= 0.5) | (analysis[type + 'tag'].iloc[i-1] == 1):
                    analysis[output_col].iloc[i] = 3    
                    
                elif (count_of_b + count_of_c >= 0.6):
                    analysis[output_col].iloc[i] = 4
                    
                elif (count_of_b + count_of_c > 0.3):
                    analysis[output_col].iloc[i] = 5    
                    
                else:
                    analysis[output_col].iloc[i] = 6
            else:
                if count_of_a>=1:
                    analysis[output_col].iloc[i] = 1
                    
                elif count_of_b>=1:
                    analysis[output_col].iloc[i] = 2
                    
                elif count_of_c>=1:
                    analysis[output_col].iloc[i] = 3
                    
                else:
                    analysis[output_col].iloc[i] = 4
        
        analysis[type+'_wgtd'].iloc[i]   = moving_average
        
        if type == 'gross':
            analysis['gross_margin_short_term'].iloc[i] = min(max(int(20 * (ratio)), -10),10)
            analysis['gross_margin_long_term'].iloc[i]  = int(10 / 6 * analysis[output_col].iloc[i])
        if type == 'net':
            analysis['net_margin_short_term'].iloc[i] = min(max(int(20 * (ratio) / 2), -10),10)
            analysis['net_margin_long_term'].iloc[i]  = int(10 / 6 * analysis[output_col].iloc[i])
            
    analysis[output_col].iloc[0:4] = int(analysis[output_col].iloc[5:7].mean())
    return analysis.drop(columns = ['row_number'])
    
    