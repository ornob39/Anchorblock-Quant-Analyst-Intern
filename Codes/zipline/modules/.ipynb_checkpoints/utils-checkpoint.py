from zipline.api import order, record, symbol
from zipline import run_algorithm
from zipline.utils.run_algo import load_extensions
from zipline.data import bundles
from zipline.data.data_portal import DataPortal
from zipline.utils.calendar_utils import get_calendar
from datetime import datetime, timedelta
import warnings
import os

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import scipy.stats as stats

def get_symbols(datadir):
    coin_list = []
    files = os.listdir(datadir)
    for i in range(len(files)):
        coin_list.append(files[i][:-4])
    return coin_list

load_extensions(
    default=True,
    extensions=[],
    strict=True,
    environ=os.environ,
)

class dailyBars:
    """This class contains the methods to retrieve data from the ingested bundle for the given timeframe.

    Args:
            coins (list, optional): A list of coins for which the price data will be returned. Defaults to ['BTC'].
            bundle (str, optional): Name of the bundle from where to extract data. Defaults to 'cryptocompare_daily'.
            calendar (exchange_calendars object): The exchange calendar object that the bundle follows.
    """
    def __init__(self, calendar, coins=['BTC'], bundle='cryptocompare_daily'):
        self.coins = coins
        self.bundle = bundle
        self.calendar = calendar
        self.extensions = load_extensions(
                                        default=True,
                                        extensions=[],
                                        strict=True,
                                        environ=os.environ,
                                        )
        self.bundle_data = bundles.load(self.bundle)
        self.data = DataPortal(
                        self.bundle_data.asset_finder,
                        trading_calendar=self.calendar,
                        first_trading_day=self.bundle_data.equity_daily_bar_reader.first_trading_day,
                        equity_minute_reader=None,
                        equity_daily_reader=self.bundle_data.equity_daily_bar_reader,
                        adjustment_reader=self.bundle_data.adjustment_reader,
                        )
        self.sids = self.bundle_data.asset_finder.sids
        self.assets = self.bundle_data.asset_finder.lookup_symbols(self.coins, as_of_date=None)
        self.pca = None
        self.num_pc = None
        self.features = None
        self.fct = None
        self.fct_ret = None
        self.fct_exp = None
        self.km = None
        self.corr = None
        self.cov = None
        self.eig_val = None
        self.eig_vec = None
        self.fct_rl = None
        self.zscores = None

    def ohlcvData(self, start, end, value_list=['close']):    
        """Generates a dataframe containing OHLCV (as per value_list) data of coins for a given timeframe.

        Args:
            start (str): Start date of price data. Format "%Y-%m-%d". Example '2022-04-05'.
            end (str): End date of price data; format "%Y-%m-%d", example '2022-07-20'.
            value_list (list, optional): A list of values to be returned. Example ['open', 'high', 'low', 'close', 'volume']. Defaults to ['close'].
        Returns:
            pd.DataFrame: A dataframe containing the daily OHLCV values of the sysmbols for a given timeframe.
        """
        warnings.filterwarnings("ignore")

        tmp = {}
    
        for value in value_list:
            df_value = self.data.get_history_window(self.assets,
                            end_dt = pd.Timestamp(end, tz='utc'),
                            bar_count = len(self.calendar.sessions_in_range(start,end)),
                            frequency = '1d',
                            field = value,
                            data_frequency = 'daily'
                            )
            df_value.columns = [eq.symbol for eq in df_value.columns]

            tmp[value] = pd.DataFrame(df_value, index=df_value.index)
            df = pd.concat(tmp, axis=1)
        return df

    def avgPriceData(self, start, end):
        """Calculate daily average price from OHLC data and return a dataframe containing prices of coins for a given timeframe.

        Args:
            start (str): Start date of price data. Format "%Y-%m-%d". Example '2022-04-05'.
            end (str): End date of price data; format "%Y-%m-%d", example '2022-07-20'.

        Returns:
            pd.DataFrame: A dataframe containing the daily prices of the sysmbols for a given timeframe.
        """
        warnings.filterwarnings("ignore")

        df = self.ohlcvData(start, end, value_list=['open', 'high', 'low', 'close'])
        return pd.concat([df['open'],df['high'],df['low'],df['close']]).groupby(level=0).mean()

    def pctReturn(self, start, end, periods=1):
        """Computes the percent changes for the given period.

        Args:
            start (str): Start date of price data. Format "%Y-%m-%d". Example '2022-04-05'.
            end (str): End date of price data; format "%Y-%m-%d", example '2022-07-20'.
            periods (int, optional): Number of periods to compute percent change for. Defaults to 1.

        Returns:
            pd.DataFrame: A dataframe containing the percent changes of the coins for the given periods.

        Note:
            The returned dataframe will ommit the first n=periods rows as they will have NaN values in them.
        """
        warnings.filterwarnings("ignore")
        
        price = self.data.get_history_window(self.assets,
                                        end_dt = pd.Timestamp(end, tz='utc'),
                                        bar_count = len(self.calendar.sessions_in_range(start,end))+periods,
                                        frequency = '1d',
                                        field = 'close',
                                        data_frequency = 'daily'
                                        )
        price.columns = [eq.symbol for eq in price.columns]

        self.pct_ret = price.pct_change(periods = periods)[periods:]
        return self.pct_ret

    def normalPCA(self, features, num_pc=1):
        """Creates a PCA object with the given number of components and fits it on the given features.

        Args:
            num_pc (int, optional): Number of components to perform PCA with. Defaults to 1.
            features (pd.DataFrame): The features to fit the PCA on.

        Returns:
            PCA object.
        """
        self.features = features
        self.num_pc = num_pc
        self.pca = PCA(n_components=num_pc)
        self.pca.fit(self.features)
        percentage = self.pca.explained_variance_ratio_
        percentage_cum = np.cumsum(percentage)
        print('{0:.2f}% of the variance is explained by the first {1} principal components'.format(percentage_cum[-1]*100, self.num_pc))
        return self.pca

    def factor(self):
        """Computes factors from features and PCA components and returns the dataframe.

        Returns:
            pd.DataFrame: Dataframe containing the factors.
        """
        fct = np.asarray(self.features).dot(self.pca.components_.T)
        fct = pd.DataFrame(columns=[f'factor{n}' for n in range(1, self.num_pc+1)], 
                            index=self.features.index,
                            data=fct)
        self.fct = fct
        return self.fct

    def factorExposures(self):
        """Returns the factor exposures of features as a dataframe.

        Returns:
            pd.DataFrame: Dataframe containing the factor exposures.
        """
        fct_exp = pd.DataFrame(index=[f'Portfolio{n}' for n in range(1, self.num_pc+1)], 
                                columns=self.features.columns,
                                data = self.pca.components_).T
        self.fct_exp = fct_exp
        return self.fct_exp

    def kmeansElbow(self, data, num_km=2):
        """Generates the elbow plot to help select the optimum number of clusters.

        Args:
            data (pd.DataFrame): The dataframe to perform clustering on.
            num_km (int, optional): Number of first n components to use for clustering. Defaults to 2.
        """
        distortions = []
        X = np.asarray(data.iloc[:,:num_km])
        for i in range(1, self.num_pc+1):
            self.km = KMeans(
                n_clusters=i, init='random',
                n_init=30, max_iter=1500,
                tol=1e-06, random_state=0
            )
            self.km.fit(X)
            distortions.append(self.km.inertia_)

        # plot
        plt.plot(range(1, self.num_pc+1), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()

    def kmeans(self, data, num_km=2, num_cl=3):
        """Performs KMeans clustering on data and returns a dataframe of samples and their assigned clusters.

        Args:
            data (pd.DataFrame): The dataframe to perform clustering on.
            num_km (int, optional): Number of first n components to use for clustering. Defaults to 2.
            num_cl (int, optional): Number of clusters to make. Defaults to 3.

        Returns:
            pd.DataFrame: Dataframe containing the samples and their assigned clusters.
        """
        self.km = KMeans(
                    n_clusters=num_cl, init='random',
                    n_init=30, max_iter=1500, 
                    tol=1e-06, random_state=0
                    )
        X = np.asarray(data.iloc[:,:num_km])
        self.km.fit(X)
        y_km = self.km.predict(X)
        temp = pd.DataFrame({'Symbol': list(data.index),
                                'Cluster': list(y_km+1)})
        
        # plot the 3 clusters
        for cluster in range(num_cl):
                plt.scatter(
                X[y_km == cluster, 0], X[y_km == cluster, 1],
                s=50,marker='s', edgecolor='black',
                label=f'cluster {cluster+1}'
                )
        
        # plot the centroids
        plt.scatter(
            self.km.cluster_centers_[:, 0], self.km.cluster_centers_[:, 1],
            s=100, marker='*',
            c='red', edgecolor='black',
            label='centroids'
            )
        plt.legend(scatterpoints=1)
        plt.title('Clusters and Centroids')
        plt.grid()
        plt.show() 
        return temp

    def correlation(self, data):
        """Plots a coolwarm heatmap of the correlation matrix.

        Args:
            data (pd.DataFrame): A dataframe containing the percent changes of the coins for the given periods.

        Outcomes:
            Stores the correlation matrix in self.corr attribute and plots the heatmap.
        """
        corr = (data).corr()
        self.corr = corr
        return corr.style.background_gradient(cmap='coolwarm')

    def covariance(self, data):
        """Plots a coolwarm heatmap of the covarience matrix.

        Args:
            data (pd.DataFrame): A dataframe containing the percent changes of the coins for the given periods.

        Outcomes:
            Stores the covariance matrix in self.cov attribute and plots the heatmap.
        """
        cov = (data).cov()
        self.cov = cov
        return cov.style.background_gradient(cmap='coolwarm')

    def plot(self, data, title):
        """Plots the timeseries of the given dataframe

        Args:
            data (pd.DataFrame): The dataframe that needs to be plotted.
            title (str): The title of the plot
        """
        data.plot(linewidth=1, figsize=(8,5))
        plt.title(title)
        plt.legend(ncol=2)
        plt.xticks(rotation=45)
        plt.xlim((data.index[0],data.index[-1]))
        plt.show()

    def cumulative(self, data, offset=0):
        """Computes the cumulative data of the columns of a dataframe and returns a new dataframe.

        Args:
            data (pd.DataFrame): The dataframe for which the cumulative data are computed.
            offset (int, optional): The starting value of the cumsum. Defaults to 0.

        Returns:
            pd.DataFrame: Dataframe containing the cumulative values.
        """
        temp = pd.DataFrame()
        for col in data.columns:
            temp[col] = data[col].cumsum() + offset
        return temp

    def scale(self, data, limit=(-1,1)):
        """Computes the scaled data of the columns of a dataframe and returns a new dataframe.

        Args:
            data (pd.DataFrame): The dataframe for which the scaled data are computed.
            limit (tuple, optional): The range within which the data are scaled.

        Returns:
            pd.DataFrame: Dataframe containing the scaled values. Defaults to (-1,1).
        """
        scaler = MinMaxScaler(limit)
        temp = pd.DataFrame()
        for col in data.columns:
            temp[col] = scaler.fit_transform(data[col].values.reshape(-1,1)).reshape(-1)
        temp.set_index(data.index, inplace=True)
        return temp
        
    def rollingPCA(self, features, window=1, num_pc=1):
        """Performs rolling PCA on daily price data and returns numpy arrays containing explained variances (eigen values) and
        PCA components (eigen vectors) for each timestamp of the given timeframe.

        Args:
            features (pd.DataFrame): The features to fit the PCA on.
            window (int, optional): The window size for performing rolling operation. Defaults to 1.
            num_pc (int, optional): Number of components for PCA. Defaults to 1.

        Returns:
            ndarray (float, size(timeframe,num_pc)): A numpy array containing the explained variances for each timestamp of the timeframe.
            ndarray (float, size(timeframe,num_pc,len(coins))): A numpy array containing the components for each timestamp of the timeframe.
        """
        
        self.features = features
        self.num_pc = num_pc
        timeFormat = "%Y-%m-%d"
        start = list(features.index)[0].strftime(timeFormat)
        end = list(features.index)[-1].strftime(timeFormat)
        start = self.calendar.sessions_window(pd.Timestamp(start, tz='utc'),-(window-1))[0].strftime(timeFormat)
        
        pct_ret_ = self.pctReturn(start, end, periods=1)
        eig_val = list()
        eig_vec = list()
        
        def rolling_pca(window_data):
            pca = PCA(n_components=num_pc)
            pca.fit(pct_ret_.iloc[window_data])
            eig_val.append(pca.explained_variance_)
            eig_vec.append(pca.components_)
            return True

        # Create a df containing row indices for the workaround
        df_idx = pd.DataFrame(np.arange(pct_ret_.shape[0]))

        # Use `rolling` to apply the PCA function
        _ = df_idx.rolling(window, min_periods=window).apply(rolling_pca, raw=True)

        # The results are now contained here:
        self.eig_val = np.array(eig_val)
        self.eig_vec = np.array(eig_vec)
        return self.eig_val, self.eig_vec

    def rollingfactor(self):
        """Computes factors from features and rolling PCA components and returns the dataframe.

        Returns:
            pd.DataFrame: Dataframe containing the factors.
        """
        # Initialize an empty df of appropriate size for the output
        self.fct_rl = pd.DataFrame(np.zeros((self.features.shape[0], self.features.shape[1])),
                                    index=self.features.index,
                                    columns=[f'factor{n}' for n in range(1, self.num_pc+1)])

        # Note: Instead of attempting to return the result, 
        #       it is written into the previously created output array.
        def factorRet(window_data):
            ret = np.asarray(self.features.iloc[int(window_data)]).reshape(1,-1)
            vec = self.eig_vec[int(window_data)].T
            fct = ret.dot(vec)
            self.fct_rl.iloc[int(window_data)] = fct
            return True

        # Create a df containing row indices for the workaround
        df_idx = pd.DataFrame(np.arange(self.features.shape[0]))

        # Use `rolling` to apply the PCA function
        _ = df_idx.rolling(1).apply(factorRet, raw=True)

        # The results are now contained here:
        return self.fct_rl
    
    def factorReturn(self,factors):
        """Computes factor returns of coins using given factors.

        Args:
            factors (pd.DataFrame): The factors used to compute factor returns.

        Returns:
            pd.DataFrame: A dataframe containing the factor returns of the sysmbols for the given factors.
        """
        tmp = {}
        fct_names = factors.columns
        pct = np.asarray(self.features)
        for fct in fct_names:
            fct_arr = np.asarray(factors[fct]).reshape(-1,1)
            tmp[fct] = pd.DataFrame(columns=[self.features.columns], 
                                    index=self.features.index,
                                    data=pct-fct_arr)
            df = pd.concat(tmp, axis=1)
        self.fct_ret = df
        return self.fct_ret
    
    def makeFactorPLot(self, fct_ret, pct_ret=None, fct_list=['factor0','factor1']):
        """Makes a plot grid of the given factor returns.

        Args:
            fct_ret (pd.DataFrame): The factor returns to plot.
            pct_ret (pd.DataFrame): The percentage returns to plot. Only required if 'factor0' is in fct_list.
                                    Defaults to None.
            fct_list (list, optional): A list of factor returns to be plot. Defaults to ['factor0','factor1'].
        """
        coins = list(fct_ret.columns.levels[1])
        ncols = 3
        nrows = int(np.ceil(len(coins)/ncols))
        for factor in fct_list:
            fig = plt.figure(figsize=(14,12))
            fig.suptitle(factor)
            plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.2, 
                            hspace=0.8)

            for coin in coins:
                plt.subplot(nrows, ncols, coins.index(coin)+1)
                if factor == 'factor0':
                    pct_ret[coin].plot()
                    plt.title(coin)
                else:
                    fct_ret[factor][coin].plot()
                    plt.title(coin)
                    
    def makeZscore(self, fct_ret):
        """Compute z-scores of factor returns.
        
        Args:
            fct_ret (pd.DataFrame): The factor returns forcomputing z-scores.
        
        Returns:
            pd.DataFrame: A dataframe containing the z-scores of the given factor returns.
        """
        self.zscores = stats.zscore(fct_ret, axis=0)
        return self.zscores