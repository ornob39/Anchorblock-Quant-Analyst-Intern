import pandas as pd
from zipline.data.bundles.csvdir import csvdir_equities
from zipline.data.bundles import register


def read_custom_csv(filepath):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


bundle_name = "custom_csv"
filepath = "/home/shell007/Documents/Anchorblock-Quant-Analyst-Intern/Codes/zipline/btc_ohlv.csv"
register(bundle_name, csvdir_equities(filepath), calendar_name="NYSE")
