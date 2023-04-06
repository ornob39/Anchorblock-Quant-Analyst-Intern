from zipline import run_algorithm
from zipline.api import (
    order_target_percent,
    symbol,
    set_commission,
    set_benchmark,
    order,
)
from zipline.finance.commission import PerTrade
import pandas as pd
import pyfolio as pf
from pyfolio.utils import extract_rets_pos_txn_from_zipline
import warnings

warnings.filterwarnings("ignore")

stock = "BTC"


def initialize(context):
    context.stock = "BTC"
    context.has_ordered = False
    context.rolling_window = 90
    set_commission(PerTrade(cost=5))
    set_benchmark(False)


def handle_data(context, data):
    if not context.has_ordered:
        order(symbol(stock), 10000)
        context.has_ordered = True


def analyze(context, perf):
    returns, positions, transactions = extract_rets_pos_txn_from_zipline(perf)


# start_date = pd.to_datetime('2021-8-20', utc= True)
# end_date = pd.to_datetime('2022-8-20', utc= True)
# print(start_date)

# results = run_algorithm(start=start_date, end=end_date, initialize=initialize, capital_base=10000, handle_data=handle_data, analyze=analyze, data_frequency='daily', bundle='cryptocompare_daily')

# zipline run --bundle cryptocompare_daily -f signals_and_strategies.py --start 2021-8-20 --end 2022-8-20 -o sns.pickle --no-benchmark
