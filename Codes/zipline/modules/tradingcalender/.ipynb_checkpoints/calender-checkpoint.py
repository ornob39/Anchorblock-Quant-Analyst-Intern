import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities
from zipline.utils.calendar_utils import register_calendar
# from zipline.data.bundles.csvdir_dse import csvdir_dse
from datetime import time
from itertools import chain
from pandas.tseries.holiday import Holiday
from pytz import timezone, UTC
from zipline.utils.calendar_utils import TradingCalendar
from exchange_calendars.exchange_calendar import HolidayCalendar

ShaheedDay = Holiday("Shaheed Day", month=2, day=21)
BirthdayBangabandhu = Holiday("BirthdayBangabandhu", month=3, day=17, start_date="2010")
IndependenceDay = Holiday("Independence Day", month=3, day=26)
BengaliNewYear = Holiday("BengaliNewYear", month=4, day=14)
MayDay = Holiday("May Day", month=5, day=1)
BankHolidayJuly = Holiday("July Bank Holiday", month=7, day=1)
BankHolidayDec = Holiday("Dec Bank Holiday", month=12, day=31)
MournDay = Holiday("National Mourn Day", month=8, day=15)
VictoryDay = Holiday("Victory Day", month=12, day=16)
ChristmasDay = Holiday("Christmas", month=12, day=25)




class XDSExchangeCalendar(TradingCalendar):
    """
    Exchange calendar for the Dhaka Stock Exchange (XDSE).


    Open Time: 10:00 Sunday-Thursday;
    Close Time: 14:30 Sunday-Thursday;

    Regularly-Observed Holidays:
    - Shaheed Day  (Feb 21)
    - Independence Day (Mar 26)
    - Shab-e-Barat
    - Shab-e-Qadar
    - Buddha Purnima
    - Bank Holiday
    - Janmastami
    - Durga-Puja
    - Bengali New Year(April 14)
    - Labour Day (May 1)
    - Juma-Tul-Wida (last Friday of Ramadan)
    - Eil-ul-Fitr (1st-3rd Shawwal)
    - Eid-ul-Azha (10th-11th Zil-Hajj)
    - National Mourn Day (Aug 15)
    - Muharram
    - Eid Milad-un-Nabi (12th Rabi-ul-Awal)
    - Christmas (Dec 25)

    Occasional holidays and adhoc holidays are observed
    """

    @property
    def name(self):
        return "XDSE"

    @property
    def tz(self):
        return timezone("Asia/Dhaka")

    open_times = ((None, time(10, 1)),)
    close_times = ((None, time(14, 29)),)

    @property
    def weekmask(self):
        return "1111001"

    @property
    def bound_start(self):
        return pd.Timestamp("2008", tz=UTC).floor("D")

    @property
    def bound_end(self):
        return pd.Timestamp.now(tz=UTC).floor("D") + pd.DateOffset(years=1)

    # ashura Holidays
    ashura = pd.to_datetime(
        [
            "2008-01-20",
            "2009-01-08",
            "2009-12-28",
            "2010-12-17",
            "2011-12-06",
            "2012-11-25",
            "2013-11-15",
            "2014-11-04",
            "2015-10-24",
            "2016-10-12",
            "2017-10-01",
            "2018-09-21",  # friday
            "2019-09-10",
            "2020-08-30",
            "2021-08-20",  # friday
            "2022-08-09",
        ]
    )

    ## Bijoy Dashami / dussehra / durga puja
    durga_puja = pd.to_datetime(
        [
            "2008-10-09",
            "2009-09-28",
            "2010-10-17",
            "2011-10-06",
            "2012-10-24",
            "2013-10-14",
            "2014-10-04",
            "2015-10-22",
            "2016-10-11",
            "2017-09-30",
            "2018-10-19",
            "2019-10-08",
            "2020-10-26",
            "2021-10-15",
            "2022-10-05",
        ]
    )

    ## Janmashtami
    janmashtami = pd.to_datetime(
        [
            "2008-08-24",
            "2009-08-13",
            "2010-09-01",
            "2011-08-22",
            "2012-08-09",
            "2013-08-28",
            "2014-08-17",
            "2015-09-05",
            "2016-08-25",
            "2017-08-14",
            "2018-09-02",
            "2019-08-24",
            "2020-08-11",
            "2021-08-30",
            "2022-08-18",
        ]
    )

    ## Buddha Purnima / Vesak / birthday of buddha
    buddha_purnima = pd.to_datetime(
        [
            "2008-05-19",
            "2009-05-09",
            "2010-05-27",
            "2011-05-17",
            "2012-05-06",
            "2013-05-23",
            "2014-05-13",
            "2015-05-03",
            "2016-05-21",
            "2017-05-10",
            "2018-04-29",
            "2019-05-18",
            "2020-05-07",
            "2021-05-26",
            "2022-05-15",
        ]
    )

    # Election day
    general_election_day = pd.to_datetime(
        [
            "2008-12-29",
            "2014-01-05",
            "2018-12-30",
        ]
    )

    # Shab e barat / mid shaban
    shab_e_barat = pd.to_datetime(
        [
            "2008-08-17",
            "2009-08-07",
            "2010-07-28",
            "2011-07-18",
            "2012-07-06",
            "2013-06-25",
            "2014-06-14",
            "2015-06-03",
            "2016-05-23",
            "2017-05-10",
            "2018-05-02",
            "2019-04-22",
            "2020-04-09",
            "2021-03-30",
            "2022-03-19",
        ]
    )

    # Quadr
    qadr = pd.to_datetime(
        [
            "2008-09-28",
            "2009-09-18",
            "2010-09-07",
            "2011-08-28",
            "2012-08-16",
            "2013-08-06",
            "2014-07-26",
            "2015-07-15",
            "2016-07-03",
            "2017-06-23",
            "2018-06-13",
            "2019-06-02",
            "2020-05-21",
            "2021-05-10",
            "2022-04-29",
        ]
    )

    # Eid ul Fitr
    eid_ul_fitr = pd.to_datetime(
        [
            *pd.date_range("2008-09-29", "2008-10-04"),
            *pd.date_range("2009-09-20", "2009-09-26"),
            *pd.date_range("2010-09-09", "2010-09-12"),
            *pd.date_range("2011-08-29", "2011-09-01"),
            *pd.date_range("2012-08-19", "2012-08-23"),
            *pd.date_range("2013-08-07", "2013-08-11"),
            *pd.date_range("2014-07-27", "2014-07-31"),
            *pd.date_range("2015-07-16", "2015-07-20"),
            *pd.date_range("2016-07-04", "2016-07-07"),
            *pd.date_range("2017-06-25", "2017-06-27"),
            *pd.date_range("2018-06-14", "2018-06-17"),
            *pd.date_range("2019-06-03", "2019-06-06"),
            *pd.date_range("2020-05-24", "2020-05-26"),
            *pd.date_range("2021-05-13", "2021-05-15"),
            *pd.date_range("2022-05-02", "2022-05-04"),
        ]
    )

    # Eid ul Azha
    eid_ul_azha = pd.to_datetime(
        [
            *pd.date_range("2008-12-07", "2008-12-11"),
            *pd.date_range("2009-11-27", "2009-11-30"),
            *pd.date_range("2010-11-16", "2010-11-18"),
            *pd.date_range("2011-11-06", "2011-11-10"),
            *pd.date_range("2012-10-25", "2012-10-29"),
            *pd.date_range("2013-10-15", "2013-10-17"),
            *pd.date_range("2014-10-05", "2014-10-09"),
            *pd.date_range("2015-09-23", "2015-09-27"),
            *pd.date_range("2016-09-11", "2016-09-15"),
            *pd.date_range("2017-09-01", "2017-09-03"),
            *pd.date_range("2018-08-21", "2018-08-23"),
            *pd.date_range("2019-08-11", "2019-08-14"),
            *pd.date_range("2020-07-31", "2020-08-02"),
            *pd.date_range("2021-07-20", "2021-07-22"),
            *pd.date_range("2022-07-09", "2022-07-11"),
        ]
    )

    # Eid e miladunnabi
    eid_e_miladunnabi = pd.to_datetime(
        [
            "2008-03-21",
            "2009-03-10",
            "2010-02-27",
            "2011-02-16",
            "2012-02-05",
            "2013-01-25",
            "2014-01-14",
            "2015-01-04",
            "2016-12-13",
            "2017-12-01",
            "2018-11-21",
            "2019-11-10",
            "2020-10-30",
            "2021-10-20",
            "2022-10-09",
        ]
    )

    # Extra adhoc holidays
    adhoc_days = pd.to_datetime(
        [
            "2011-02-17",  ## extra for eid e miladunnabi
            "2013-10-13",  ## aditional for durga puja and eid ul azha
            # "2021-07-04",  ## Covid special holiday for banks
            "2021-07-11",  # special holiday for covid
            "2021-08-01",  # special holiday for covid
            "2021-08-08",  # special holiday for covid
            "2021-08-04",  # unknown holiday
        ]
    )

    covid_holidays = pd.to_datetime([*pd.date_range("2020-03-26", "2020-05-30")])
    # random_holidays = pd.to_datetime(
    #     [
    #         *pd.date_range("2008-03-2", "2008-03-05"),
    #         "2008-03-27",
    #         "2009-10-15",
    #         "2011-01-23",
    #         "2011-01-24",
    #         "2011-02-23",
    #         "2011-02-24",
    #         "2011-04-28",
    #         "2011-05-02",
    #         "2011-12-01",
    #         "2011-12-04",
    #         "2012-01-17",
    #         "2012-02-26",
    #         "2012-07-31",
    #         "2013-03-21",
    #         "2015-04-28",
    #         # "2015-12-31",  #-> has value
    #         # "2016-07-16", #-> has value
    #         # "2016-09-24", #-> has value
    #         "2019-02-28",
    #         # "2019-10-16",
    #         # "2019-10-17",
    #         # "2020-01-08",
    #         # "2020-03-16",
    #         # "2020-03-01",
    #         "2016-04-17",
    #         #missing,
    #         "2017-04-04",
    #         "2018-04-25",
    #         "2019-05-26",
    #         "2020-07-07",
    #         "2020-07-30",
    #         "2021-05-12",
    #         "2022-01-09",
    #         "2015-04-15",
    #     ]
    # )
    # # For ICCBBANK
    # extra_holidays = pd.to_datetime(
    #     [
    #         "2015-12-31",
    #         "2016-07-16",
    #         "2016-09-24",
    #         "2020-03-29",
    #     ]
    # )
    

    @property
    def regular_holidays(self):
        return HolidayCalendar(
            [
                ShaheedDay,
                IndependenceDay,
                BirthdayBangabandhu,
                MournDay,
                BankHolidayDec,
                BankHolidayJuly,
                ChristmasDay,
                MayDay,
                BengaliNewYear,
                VictoryDay,
            ]
        )

    @property
    def adhoc_holidays(self):
        return list(
            chain(
                self.eid_e_miladunnabi,
                self.buddha_purnima,
                self.eid_ul_fitr,
                self.eid_ul_azha,
                self.ashura,
                self.durga_puja,
                self.adhoc_days,
                self.qadr,
                self.janmashtami,
                self.general_election_day,
                self.shab_e_barat,
                self.covid_holidays,
                # self.random_holidays,
                # self.extra_holidays
            )
        )
# register_calendar("XDSE", XDSExchangeCalendar())

# start_session = pd.Timestamp('2016-1-3', tz='utc')
# end_session = pd.Timestamp('2017-1-1', tz='utc')

# register(
#     "dsebundle",
#     csvdir_equities(
#       ["daily"],
#       "/root/codespace/data/dsedata"
#       ),
#     calendar_name="XDSE",
# )

