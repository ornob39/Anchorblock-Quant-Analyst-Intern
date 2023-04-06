import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def rolling_sharpe(ret):
    return np.multiply(np.divide(ret.mean(), ret.std()), np.sqrt(252))

def get_rolling_stats(result, r_window):
    result["rolling_sharpe"] = result["portfolio_value"].pct_change().rolling(r_window).apply(rolling_sharpe)
    result["rolling_vol"] = result["portfolio_value"].pct_change().rolling(r_window).std()
    stats = [math.log(result.rolling_sharpe.mean())/result.rolling_sharpe.std(), result.portfolio_value.mean(), result.rolling_vol.mean()]
    return result, stats

def show_rolling_stats(result, r_window):
    result, stats = get_rolling_stats(result, r_window)
    fig, ax = plt.subplots(1, 3, figsize=(18,4))
    result['rolling_sharpe'].plot(ax = ax[0], title='Rolling sharpe')
    ax[0].axhline(0,color='red',ls='--')
    result['rolling_vol'].plot(ax = ax[1], title='Rolling vol')
    result['portfolio_value'].plot(ax = ax[2], title='Portfolio value')
    plt.show()
    return result

def view_performance(prefs, pref_result, repo_dir, vindex=False):
    """
    Compares strategy results against input prices pref
    If pref is a list, uses volume-weighted index of stocks in the list
    If pref is a string, uses price returns of corresponding stock
    """
    plt.figure(figsize=(20, 8))
    
    if type(prefs)==list:
        dsex = vol_weighted_index(prefs, repo_dir).loc[pref_result.index[0]:]
        plt.plot(dsex.index, (1+dsex.pct_change()).cumprod(), label="Volume-weighted Index", color="black")
    elif type(prefs)==str:
        dsex = pd.read_csv(csi_dir + "/daily/" + str(prefs) +'.csv')
        try:
            dsex.index = pd.DatetimeIndex(dsex.Date)
            dsex.sort_index(inplace=True)
            dsex["close"] = (1+(dsex.Open.str.replace(",", "")).astype(float).pct_change()).cumprod()
            dsex = dsex.loc[str(pref_result.index[0].year):]
            plt.plot(dsex.index, dsex.close, label=prefs, color="black")
        except:
            dsex.set_index(pd.DatetimeIndex(dsex.timestamp), drop=True, inplace=True)
            dsex = dsex.loc[str(pref_result.index[0].year):]
            dsex.sort_index(inplace=True)
            plt.plot(dsex.index, (1+(dsex.close.pct_change())).cumprod(), label=prefs, color="black")
        
    plt.plot(pref_result.index, (1+(pref_result.portfolio_value.pct_change())).cumprod(), label="AMA value", color="green")
    plt.legend()
    display(plt.show())
    return 

def vol_weighted_index(prefs, repo_dir):
    path = repo_dir + 'daily'
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    master = {}
    # loop over the list of csv files
    for f in csv_files:
        # print the location and filename
        filename = f.split("/")[-1].split(".")[0]
        if filename in prefs:
            # read the csv file
            df = pd.read_csv(f)
            df.index = pd.DatetimeIndex(df.timestamp)
            df.sort_index(inplace=True)
            # take volume and close data
            master[filename] = (1+(df.close.astype(float).pct_change())).cumprod()
            master[filename+"_v"] = df.volume
    pvs = pd.DataFrame(master)
    pvs["totals"] = 0
    pvs["vols"] = 0
    for stock in prefs:
        pvs["totals"] += pvs[stock]*pvs[stock+"_v"]
        pvs["vols"] += pvs[stock+"_v"]
    return (pvs["totals"]/pvs["vols"])