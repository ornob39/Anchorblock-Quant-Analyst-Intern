from datetime import datetime, timedelta
import pandas as pd
import pytz
import requests
import time
import pickle
import os


def getSymbolFromExchange(
    exchange="Coinbase",
    conversion="USD",
    apikey="5c7ed28f27882a16f8483b7aeefa0fb7f59ca7f89a9da7040e0f750db3e04713",
):
    """Makes a list of symbols that exist in the given exchange for the given conversion coin.

    Args:
        exchange (str, optional): Exchange name. Defaults to 'CCCAGG'.
        conversion (str, optional): Symbol of the conversion coin. Defaults to 'USD'.
        apikey (str, optional): Cryptocompare api key. Defaults to "5c7ed28f27882a16f8483b7aeefa0fb7f59ca7f89a9da7040e0f750db3e04713".

    Returns:
        list: List of coins that satisfy the criteria.
    """
    base = (
        f"https://min-api.cryptocompare.com/data/v2/pair/mapping/exchange?e={exchange}"
    )
    api = f"&api_key={apikey}"
    url = base + api
    response = requests.get(url).json()
    if response["Data"] == {}:
        return False
    response["Data"]["current"]
    coin_list = [
        pair["exchange_fsym"]
        for pair in response["Data"]["current"]
        if pair["exchange_tsym"] == conversion
    ]
    return coin_list


def getExchanges(
    apikey="5c7ed28f27882a16f8483b7aeefa0fb7f59ca7f89a9da7040e0f750db3e04713",
):
    """Makes a list of exchanges that exist in cryptocompare.
    Args:
        apikey (str, optional): Cryptocompare api key. Defaults to "5c7ed28f27882a16f8483b7aeefa0fb7f59ca7f89a9da7040e0f750db3e04713".

    Returns:
        list: List of exchanges.
    """
    base = f"https://min-api.cryptocompare.com/data/v2/pair/mapping/exchange?"
    api = f"&api_key={apikey}"
    url = base + api
    response = requests.get(url).json()
    return list(response["Data"].keys())


def coinList(
    exchanges=["Coinbase"],
    conversion="USD",
    apikey="5c7ed28f27882a16f8483b7aeefa0fb7f59ca7f89a9da7040e0f750db3e04713",
):
    """Makes a list of symbols from all available exchanges in cryptocompare for a given conversion symbol.

    Args:
        exchanges (list, optional): List of exchanges. Defaults to ['Coinbase'].
        conversion (str, optional): Symbol of the conversion coin. Defaults to 'USD'.
        apikey (str, optional): Cryptocompare api key. Defaults to "5c7ed28f27882a16f8483b7aeefa0fb7f59ca7f89a9da7040e0f750db3e04713".

    Returns:
        list: List of coins.
    """

    coin_list = list()
    for exchange in exchanges:
        coins = getSymbolFromExchange(exchange=exchange, conversion=conversion)
        if not coins:
            continue
        coin_list = list(set().union(coin_list, coins))
    return coin_list


def getDailyBars(
    start,
    end,
    coin,
    conversion="USD",
    exchange="CCCAGG",
    apikey="5c7ed28f27882a16f8483b7aeefa0fb7f59ca7f89a9da7040e0f750db3e04713",
):
    """Fetches OHLCV data from cryptocompare for the given parameters and returns a dataframe

    Args:
        start (str): Start date of data. Format "%Y-%m-%d". Example '2022-04-05'.
        end (str): End date of data; format "%Y-%m-%d", example '2022-07-20'.
        coin (str): The symbol of the coin for which data will be fetched.
        conversion (str, optional): The symbol of the conversion currency. Defaults to 'USD'.
        exchange (str, optional): Name of the exchange from where data will be fetched. Defaults to 'CCCAGG'.
        apikey (str, optional): Cryptocompare api key. Defaults to "5c7ed28f27882a16f8483b7aeefa0fb7f59ca7f89a9da7040e0f750db3e04713".

    Returns:
        pd.DataFrame: Dataframe containing the OHLCV data.
    """

    timeFormat = "%Y-%m-%d"
    start = datetime.strptime(start, timeFormat) + timedelta(days=1)
    end = datetime.strptime(end, timeFormat) + timedelta(days=1)
    limit = (end - start).days
    base = f"https://min-api.cryptocompare.com/data/v2/histoday?"
    params = f"fsym={coin}&tsym={conversion}&limit={limit}&e={exchange}&toTs={time.mktime(end.timetuple())}"
    api = f"&api_key={apikey}"
    url = base + params + api
    response = requests.get(url).json()
    if response["Data"] == {}:
        return pd.DataFrame()
    df = pd.DataFrame(response["Data"]["Data"])
    df.drop(["volumefrom", "conversionType", "conversionSymbol"], axis=1, inplace=True)
    df.rename(columns={"volumeto": "volume", "time": "timestamp"}, inplace=True)
    df = df.reindex(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = df["timestamp"].apply(
        lambda x: datetime.fromtimestamp(x, pytz.utc).replace(
            hour=0, minute=0, second=0
        )
    )
    df = df.set_index("timestamp")
    return df


def coinLibrary(start, end, coin_list, conversion="USD", exchange_list=["Coinbase"]):
    """Makes a dictionary that contains the information on which coin has data for the desired timeframe in which exchange.

    Args:
        start (str): Start date of data. Format "%Y-%m-%d". Example '2022-04-05'.
        end (str): End date of data; format "%Y-%m-%d", example '2022-07-20'.
        coin_list (list): List of the symbols to make the dictionary for.
        conversion (str, optional): The symbol of the conversion currency. Defaults to 'USD'.
        exchange_list (list, optional): List of the exchanges to search for data. Defaults to ['Coinbase'].

    Returns:
        Dictionary: A dictionary that contains the information on which coin has data for the desired timeframe in which exchange.
    """
    coin_dict = {}
    if "CCCAGG" in exchange_list:
        exchanges = exchange_list
    else:
        exchanges = ["CCCAGG"] + exchange_list
    for coin in coin_list:
        coinInfo = {}
        for exchange in exchanges:
            df = getDailyBars(
                start=start,
                end=end,
                coin=coin,
                conversion=conversion,
                exchange=exchange,
            )
            if df.empty:
                continue
            elif df.iloc[0].sum() == 0:
                continue
            else:
                coinInfo["exchange"] = exchange
                coinInfo["conversion"] = conversion
                coinInfo["start"] = start
                coinInfo["end"] = end
                break
        if coinInfo == {}:
            # pass
            print(f"{coin} does not have data for the given timeframe.")
        else:
            coin_dict[coin] = coinInfo
            # print(f"{coin} has data for the given timeframe in {coinInfo['exchange']}.")
    return coin_dict


def downloader(coin_dict, savedir):
    """Download the OHLCV data for the given time frame and save them as csvs in the desired location.

    Args:
        coin_dict (dictionary): A dictionary that contains the information on which coin has data for
                                the desired timeframe in which exchange.
        savedir (str): Path of the save directory.
    """
    for coin in list(coin_dict.keys()):
        df = getDailyBars(
            start=coin_dict[coin]["start"],
            end=coin_dict[coin]["end"],
            coin=coin,
            conversion=coin_dict[coin]["conversion"],
            exchange=coin_dict[coin]["exchange"],
        )
        df.to_csv(savedir + f"{coin}.csv")


def download(coin_dict, savedir):
    if os.path.exists(savedir):
        for f in os.listdir(savedir):
            os.remove(os.path.join(savedir, f))
        downloader(coin_dict, savedir)
    else:
        os.makedirs(savedir)
        downloader(coin_dict, savedir)


# write list to binary file
def write_list(filename, a_list):
    # store list in binary file so 'wb' mode
    with open(filename, "wb") as fp:
        pickle.dump(a_list, fp)


# Read list to memory
def read_list(filename):
    # for reading also binary mode is important
    with open(filename, "rb") as fp:
        n_list = pickle.load(fp)
        return n_list
