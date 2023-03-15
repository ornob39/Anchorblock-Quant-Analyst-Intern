import pandas as pd
import scipy.stats
import numpy as np


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    The wealth index
    The previous peaks
    Percent Drawdowns"""
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    return pd.DataFrame(
        {
            "Wealth": wealth_index,
            "Previous Peaks": previous_peaks,
            "Drawdown": drawdowns,
        }
    )


def get_ffme_returns():
    me_m = pd.read_csv(
        "data/Portfolios_Formed_on_ME_monthly_EW.csv",
        header=0,
        index_col=0,
        na_values=-99.99,  # type: ignore
    )
    returns = me_m[["Lo 10", "Hi 10"]]
    returns.columns = ["SmallCap", "LargeCap"]
    returns = returns / 100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period("M")
    return returns


def get_hfi_returns():
    """Load and format the EDHEC Hedge Fund Index Data"""
    hfi = pd.read_csv(
        "data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True
    )
    hfi = hfi / 100
    hfi.index = hfi.index.to_period("M")  # type: ignore
    return hfi


def get_ind_returns():
    ind = (
        pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)
        / 100
    )
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


def semideviation(r):
    """Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame"""
    # is_negative = r < 0
    # return r[is_negative].std(ddof=0)
    excess = r - r.mean()  # Demean the retutns
    excess_negative = excess[excess < 0]  # Take only the returns below the mean
    excess_negative_squared = excess_negative**2
    n_negative = (excess < 0).sum()  # Count the number of negative returns
    return (excess_negative_squared.sum() / n_negative) ** 0.5  # Semideviation


def skewness(r):
    """Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series"""
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)  # Use the population standard deviation, not sample
    exp = (demeaned_r**3).mean()
    return exp / (sigma_r**3)


def kurtosis(r):
    """Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series"""
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)  # Use the population standard deviation, not sample
    exp = (demeaned_r**4).mean()
    return exp / (sigma_r**4)


def is_normal(r, level=0.01):
    """Apply Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise"""
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r, level=5):
    """Returns the historic Value at Risk at a specified level"""
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")


from scipy.stats import norm


def var_gaussian(r, level=5, modified=False):
    """Returns the Parametric Gaussian VaR of a Series or DataFrame"""
    # Compute the Z score assuming it was Gaussian
    z = norm.ppf(level / 100)
    if modified:
        # Modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * (s**2) / 36
        )
    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r, level=5):
    """Computes the Conditional VaR of Series or DataFrame"""
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level)
        return -(r[is_beyond].mean())
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    # Check if the series is empty
    if returns.empty:
        raise ValueError("Series is empty")
    # Calculate the standard deviation of the returns
    std = returns.std()
    # Return the annualized volatility
    return std * np.sqrt(periods_per_year)


def annualized_returns(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized returns of a returns stream"""
    # Check if the series is empty
    if returns.empty:
        raise ValueError("Series is empty")
    # Calculate the compounded growth of the investment
    compounded_growth = (1 + returns).prod()

    # Calculate the number of periods
    n_periods = returns.shape[0]

    # Calculate the annualized rate of return
    return (compounded_growth ** (periods_per_year / n_periods)) - 1


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float, periods_per_year: int = 12
) -> float:
    """Calculate the annualized Sharpe ratio of a returns stream"""

    # Convert the annual risk free rate to a per period
    rf_per_period = ((1 + risk_free_rate) ** (1 / periods_per_year)) - 1
    # Calculate the excess returns
    excess_returns = returns - rf_per_period
    # Calculate the annualized excess returns
    ann_excess_returns = annualized_returns(excess_returns, periods_per_year)
    # Calculate the annualized volatility of the excess returns
    ann_volatility = annualized_volatility(returns, periods_per_year)
    # Calculate the Sharpe ratio
    return ann_excess_returns / ann_volatility
