import zipline
from zipline.api import order_target, record, symbol, order

stocks = ["BTC"]
stock = "BTC"


def initialize(context):
    context.i = 0
    context.asset = symbol("BTC")


def handle_data(context, data):
    context.i += 1
    if context.i < 30:
        return

    short_mavg = data.history(
        context.asset, "price", bar_count=10, frequency="1d"
    ).mean()
    long_mavg = data.history(
        context.asset, "price", bar_count=30, frequency="1d"
    ).mean()

    if short_mavg < long_mavg:
        order(symbol(stock), 100)
    elif short_mavg > long_mavg:
        order(symbol(stock), 0)
    record(
        BTC=data.current(context.asset, "price"),
        short_mavg=short_mavg,
        long_mavg=long_mavg,
    )


# zipline run --bundle cryptocompare_daily -f dual_ma.py --start 2021-8-20 --end 2022-8-20 -o dual_ma.pickle --no-benchmark
