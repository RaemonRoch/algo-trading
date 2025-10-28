import pandas as pd
from ta import add_all_ta_features

def load_data_with_all_features() -> pd.DataFrame:
    df = pd.read_csv("../data/Binance_BTCUSDT_1h.csv", header=1)
    df.rename(columns=lambda x:x.lower(), inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = add_all_ta_features(df, open='open', high='high', low='low', close='close', volume='volume btc')
    df['year'] =  df['date'].dt.year
    df['month'] =  df['date'].dt.month
    df['day'] =  df['date'].dt.day
    df['hour'] =  df['date'].dt.hour
    df.set_index('date', inplace=True)
    df.drop(['unix', 'symbol'], axis=1, inplace=True)
    df['y'] = df['close'].pct_change().shift(-1)
    df['y'] = df['y'].apply(lambda x: 1 if x > 0 else 0)
    return df