import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import zipline
import os
import zipline
from zipline.api import order_target, record, symbol, order, order_target_percent


def initialize(context):
    context.asset = "BTC"


def handle_data(context, data):
    price = data.current(context.asset, "price")
    if price > 25000:
        order_target_percent(context.asset, 1.0)
    elif price < 18000:
        order_target_percent(context.asset, -1.0)

    record(BTC=data.current(context.asset, "price"))


# zipline run --bundle cryptocompare_daily -f buy_on_price.py --start 2021-8-20 --end 2022-8-20 -o bop.pickle --no-benchmark
