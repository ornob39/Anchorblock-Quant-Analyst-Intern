import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import zipline
from zipline.data import bundles
from zipline.utils.run_algo import load_extensions
import os

load_extensions(
    default=True,
    extensions=[],
    strict=True,
    environ=os.environ,
)
from zipline.data.bundles import register, unregister
from zipline.data.bundles.csvdir import csvdir_equities
from zipline.utils.calendar_utils import register_calendar, get_calendar
from zipline.finance import commission, slippage


from zipline.api import (
    order,
    record,
    symbol,
    get_datetime,
    order_target_percent,
    order_target_value,
    set_benchmark,
    get_open_orders,
)
from zipline import run_algorithm
from zipline.api import order_target, record, symbol
from pyfolio.utils import extract_rets_pos_txn_from_zipline
import pyfolio as pf
import seaborn as sns
from pyfolio.utils import extract_rets_pos_txn_from_zipline


stock = "BTC"


def initialize(context):
    context.has_ordered = False
    context.stocks = symbol(stock)


def handle_data(context, data):
    if not context.has_ordered:
        order(symbol(stock), 10000)
        context.has_ordered = True

def analyze(context, perf):
    returns, positions, transactions = extract_rets_pos_txn_from_zipline(perf)

# zipline run --bundle cryptocompare_daily -f buy_and_hold.py --start 2021-8-20 --end 2022-8-20 -o bnh.pickle --no-benchmark
