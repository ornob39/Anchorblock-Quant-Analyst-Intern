import os 
import pandas as pd #pandas==1.2.5
from datetime import datetime, timezone
pd.options.mode.chained_assignment = None
import pytz
import pyfolio as pf
import time
import calendar
import glob

from zipline.data import bundles as bundles_module
from zipline.data.bundles import register, unregister
from zipline.data.bundles.csvdir import csvdir_equities
from zipline.utils.calendar_utils import register_calendar, get_calendar
from exchange_calendars.exchange_calendar_xdse import XDSExchangeCalendar

# def getTimestamp(date):
#     monthlist = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#     month = [i for i in range(len(monthlist)) if date[:3]==monthlist[i]][0]+1
#     return datetime(year=int(date[-4:]), month=month, day=int(date[4:6]), hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

# def to_seconds(date):
#     return time.mktime(date.timetuple())

def getValue(val):
    if val=="-":
        res = 0
    else:
        newval, mul = float(val[:-1]), val[-1]
        res = newval*1000 if mul=='K' else (newval*1000000 if mul=='M' else 0 ) 
    return res

# def cleanData(dataframe):
#     seconds = []
#     df = dataframe.rename(columns={'Date':'timestamp', 'Price':'close', 'Open':'open', 'High':'high', 'Low':'low', 'Vol.':'volume', 'Change %':'change'})
#     for i in range(len(df)):
#         df['timestamp'][i] = getTimestamp(df['timestamp'][i])
#         df['volume'][i] = getValue(df['volume'][i])
#         df['open'][i] = float(df['open'][i])
#         df['high'][i] = float(df['high'][i])
#         df['low'][i] = float(df['low'][i])
#         df['close'][i] = float(df['close'][i])
#         seconds.append(int(calendar.timegm(df['timestamp'][i].timetuple())))
#     df.loc[:,'seconds'] = pd.Series(seconds)
#     df = df.reindex(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'seconds'], index=df.index[::-1])
#     df.reset_index(drop=True, inplace=True)
#     df.set_index(keys='timestamp', inplace=True)
#     # df = df.drop(df.index[[372, 502, 546, 1397]])
#     return df

def easy_clean(dataframe):
    df = dataframe.rename(columns={'Date':'timestamp', 'Price':'close', 'Open':'open', 'High':'high', 'Low':'low', 'Vol.':'volume', 'Change %':'change'})
    df["timestamp"] = pd.DatetimeIndex(df.timestamp).strftime('%Y-%m-%d %H:%M:%S+%M:%S')
    df["open"] = (df["open"].str.replace(",", "")).astype(float)
    df["high"] = (df["high"].str.replace(",", "")).astype(float)
    df["low"] = (df["low"].str.replace(",", "")).astype(float)
    df["close"] = (df["close"].str.replace(",", "")).astype(float)
    df.set_index(df.timestamp, inplace=True, drop=True)
    df["volume"] = [getValue(vol) for vol in df.volume]
    df.drop(["change", "timestamp"], axis=1, inplace=True)
    df.sort_index()
    return df


def second_stage(csv_path, missing, extra):

    # datadir = f'{repo_dir}/data/data_v1/daily/{name}.csv'
    # savedir = f'{repo_dir}/data/daily/{name}.csv'
    df = pd.read_csv(csv_path)

    # adding the missing values
    for i in range(len(missing)):
        y, m, d = int(missing[i][:4]), int(missing[i][5:7]), int(missing[i][8:])
        date = datetime(year=y, month=m, day=d, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        s = int(calendar.timegm(date.timetuple()))
        o, h, l, c, v = 0, 0, 0, 0, 0
        df.loc[len(df.index)] = [date, o, h, l, c, v] #insert new row

    # removing the extra values
    indices = []
    for i in range(len(df)):
        date = str(df['timestamp'][i])[:10]
        if date in extra:
            indices.append(i)
    df = df.drop(df.index[indices]) #delete the row
    #df.sort_values(by=['seconds'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index('timestamp', drop=True, inplace=True)
    #df.drop(columns='seconds', inplace=True)

    # adding the ohlcv values from previous row value
    for i in range(len(df)):
        o, h, l, c = float(df['open'][i]), float(df['high'][i]), float(df['low'][i]), float(df['close'][i])
        if o==0.0 and h==0.0 and l==0.0 and c==0.0:
            df['open'][i] = float(df['open'][i-1])
            df['high'][i] = float(df['high'][i-1])
            df['low'][i] = float(df['low'][i-1])
            df['close'][i] = float(df['close'][i-1])

    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # saving for main ingestion use
    df.to_csv(csv_path)
    return ("Reprocessed and saved")

def z_ingest(csv_path, bundle_name, calendar_string):
    #register(bundle_name, csvdir_equities(["daily"], csv_dir_path), calendar_name=calendar_string)
    dates = []
    try:
      bundles_module.ingest(bundle_name)
    except AssertionError as e:
        dates.append(e)
        msg = str(dates[0]).split('\n')
        # spliting the way into dates
        missing, extra = msg[1][19:-1].split(',')[::2], msg[2][17:-1].split(',')[::2]
        missing[0], extra[0] = missing[0][11:21], extra[0][11:21]
        missing[1:], extra[1:] = [missing[i][12:22] for i in range(1, len(missing))], [extra[i][12:22] for i in range(1, len(extra))]
        print(f'Got : {int(msg[0][3:8])}, Missing : {len(missing)}, Extra : {len(extra)} || Add : {len(missing)-len(extra)}, Expected dataframe length : {int(msg[0][3:8])+len(missing)-len(extra)}.')
        second_stage(csv_path, missing, extra)