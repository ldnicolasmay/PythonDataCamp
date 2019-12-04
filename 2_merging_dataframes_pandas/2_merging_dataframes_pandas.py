# 2_merging_dataframes_pandas.py


# 1 Preparing Data

# Reading Multiple Data FIles

import pandas as pd

base_dir = "./2_merging_dataframes_pandas/Summer Olympic medals/"

df_brnz = pd.read_csv(base_dir + "Bronze.csv")
df_slvr = pd.read_csv(base_dir + "Silver.csv")
df_gold = pd.read_csv(base_dir + "Gold.csv")

df_brnz
df_slvr
df_gold

base_dir = "./2_merging_dataframes_pandas/Summer Olympic medals/"
filenames = ["Bronze.csv", "Silver.csv", "Gold.csv"]

dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(base_dir + filename))

dataframes[0].head()


(df_brnz.index == df_slvr.index).all()
(df_brnz.index == df_gold.index).all()

df_medals = df_brnz.copy()
df_medals.columns = ['NOC', 'Country', 'Bronze']
df_medals['Silver'] = df_slvr['Total']
df_medals['Gold'] = df_gold['Total']
df_medals.head(10)


# Reindexing DataFrames

weather1 = pd.DataFrame.from_dict(
    {
        'Month': ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        'Max TemperatureF': [68, 60, 68, 84, 88, 89,
                             91, 86, 90, 84, 72, 68]
    }).\
    set_index('Month')
weather1

weather2 = weather1.sort_index()
weather2

weather3 = weather1.sort_index(ascending=False)
weather3

weather4 = weather1.sort_values('Max TemperatureF')
weather4

weather2 = weather

# weather1 = pd.DataFrame.from_dict(
#     {
#         'Month': ["2019-01", "2019-02", "2019-03", 
#                   "2019-04", "2019-05", "2019-06",
#                   "2019-07", "2019-08", "2019-09",
#                   "2019-10", "2019-11", "2019-12"],
#         'Max TemperatureF': [68, 60, 68, 84, 88, 89,
#                              91, 86, 90, 84, 72, 68]
#     }).\
#     set_index('Month')
# weather1.index = pd.to_datetime(weather1.index, format="%Y-%m")
# weather1
# weather1.resample('3M', closed='left').mean()

year = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

weather1 = pd.DataFrame.from_dict(
    {
        'Month': ["Apr", "Jan", "Jul", "Oct"],
        'Mean TemperatureF': [61.956044, 32.133333, 32.133333, 32.133333]
    }).\
    set_index('Month')
weather1

weather1.reindex(year)
weather2 = weather1.reindex(year)

# weather1.reindex(year, method='ffill')
# weather3 = weather1.reindex(year, method='ffill')
weather1.reindex(year).ffill()
weather3 = weather1.reindex(year).ffill()

base_dir = "./2_merging_dataframes_pandas/Baby names/"
df_names_1881 = pd.read_csv(base_dir + "names1881.csv", header=None)
df_names_1981 = pd.read_csv(base_dir + "names1981.csv", header=None)
df_names_1881.head(10)
df_names_1981.head(10)

df_names_1881.reindex(df_names_1981.index)
common_names = df_names_1881.reindex(df_names_1981.index)
common_names.shape
common_names = common_names.dropna()
common_names.shape
common_names.head(20)


# Arithmetic with Series & DataFrames

base_dir = "./2_merging_dataframes_pandas/"
weather = pd.read_csv(base_dir + "pittsburgh2013.csv",
                      index_col='Date', parse_dates=True)
weather.head(20)

weather.loc["2013-07-01":"2013-07-07", 'PrecipitationIn']
weather.loc["2013-07-01":"2013-07-07", 'PrecipitationIn'] * 2.54  # convert cm

week1_range = weather.loc["2013-07-01":"2013-07-07", 
                          ['Min TemperatureF', 'Max TemperatureF']]
week1_range
week1_mean = weather.loc["2013-07-01":"2013-07-07", 'Mean TemperatureF']
week1_mean

week1_range / week1_mean  # doesn't work as we'd like

week1_range.divide(week1_mean, axis='rows')  # use `divide` instead

week1_mean.pct_change()
week1_mean.pct_change() * 100

base_dir = "./2_merging_dataframes_pandas/Summer Olympic medals/"
df_brnz = pd.read_csv(base_dir + "bronze_top5.csv", index_col='Country')
df_slvr = pd.read_csv(base_dir + "silver_top5.csv", index_col='Country')
df_gold = pd.read_csv(base_dir + "gold_top5.csv", index_col='Country')

df_brnz
df_slvr
df_gold

df_brnz + df_slvr  # doesn't work as we'd like

df_brnz.add(df_slvr)  # works like the `+` operator :(
df_brnz.add(df_slvr, fill_value=0)  # use `add` with `fill_value` parameter

df_brnz + df_slvr + df_gold
df_brnz.add(df_slvr, fill_value=0).add(df_gold, fill_value=0)  # chain `add`s


df_january = pd.DataFrame.from_dict(
    {
        'Company': ["Acme Corp", "Hooli", "Initech", "MediaCore", "Streeplex"],
        'Units': [19, 17, 20, 10, 13]
    }).\
    set_index('Company')

df_february = pd.DataFrame.from_dict(
    {
        'Company': ["Acme Corp", "Hooli", "Initech", "Vandelay Inc"],
        'Units': [15, 3, 13, 25]
    }).\
    set_index('Company')

df_january
df_february
df_january + df_february


base_dir = "./2_merging_dataframes_pandas/"
df_weather = pd.read_csv(base_dir + "pittsburgh2013.csv")

df_temps_f = \
    df_weather[['Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF']]
df_temps_f
df_temps_c = (df_temps_f - 32) * 5/9
df_temps_c

df_temps_c.columns = df_temps_c.columns.str.replace('F', 'C')
df_temps_c


base_dir = "./2_merging_dataframes_pandas/GDP/"
df_gdp = pd.read_csv(base_dir + "gdp_usa.csv", 
                     parse_dates=True, index_col='DATE')
df_gdp
df_gdp.plot()

df_gdp_post2008 = df_gdp.loc["2008":, :]
df_gdp_post2008.plot()

df_gdp_post2008.resample('A').last()
df_gdp_post2008.resample('A').last().pct_change().plot()

(df_gdp.resample('A').mean().pct_change() * 100).plot()

df_gdp_us = pd.read_csv(base_dir + "gdp_usa.csv",
                        parse_dates=True, index_col='DATE')
df_gdp_zh = pd.read_csv(base_dir + "gdp_china.csv",
                        parse_dates=True, index_col='Year')
df_gdp_us.index.name = 'date'
df_gdp_zh.index.name = 'date'
df_gdp_us.columns = ['gdp_us']
df_gdp_zh.columns = ['gdp_zh']
df_gdp_us
df_gdp_zh

df_gdp = df_gdp_zh.copy()
df_gdp['gdp_us'] = df_gdp_us['gdp_us']
df_gdp.plot()
(df_gdp.pct_change() * 100).plot()


base_dir = "./2_merging_dataframes_pandas/"
df_sp500 = pd.read_csv(base_dir + "sp500.csv",
                       parse_dates=True, index_col='Date')
df_exchg = pd.read_csv(base_dir + "exchange.csv", 
                       parse_dates=True, index_col='Date')

df_sp500
df_exchg

dollars = df_sp500[['Open', 'Close']]
dollars

pounds = dollars.multiply(df_exchg['GBP/USD'], axis='rows')
pounds



# 2 Concatenating Data


# 3 Merging Data


# 4 Case Study - Summer Olympics
