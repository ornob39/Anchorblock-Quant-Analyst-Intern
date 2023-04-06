import zipline
from zipline.api import (
    order_target,
    record,
    symbol,
    order,
    order_value,
    order_target_percent,
)

stocks = ["BTC"]
stock = "BTC"


def initialize(context):
    context.asset = symbol("BTC")


def handle_data(context, data):
    ma = data.history(context.asset, "price", 10, "1d").mean()
    if data.current(context.asset, "price") > ma:
        order_target_percent(context.asset, 0.5)
    else:
        order_target_percent(context.asset, 0)


# zipline run --bundle cryptocompare_daily -f simple_ma.py --start 2021-8-20 --end 2022-8-20 -o sma.pickle --no-benchmark
