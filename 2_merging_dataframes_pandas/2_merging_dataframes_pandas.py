# 2_merging_dataframes_pandas.py


# 1 Preparing Data

# Reading Multiple Data FIles

import pandas as pd

base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Summer Olympic medals/"

df_brnz = pd.read_csv(base_dir + "Bronze.csv")
df_slvr = pd.read_csv(base_dir + "Silver.csv")
df_gold = pd.read_csv(base_dir + "Gold.csv")

df_brnz
df_slvr
df_gold

base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Summer Olympic medals/"
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

weather1 = pd.DataFrame.from_dict({
    'Month': ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    'Max TemperatureF': [68, 60, 68, 84, 88, 89,
                         91, 86, 90, 84, 72, 68]}).\
    set_index('Month')
weather1

weather2 = weather1.sort_index()
weather2

weather3 = weather1.sort_index(ascending=False)
weather3

weather4 = weather1.sort_values('Max TemperatureF')
weather4

# weather2 = weather1

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

weather1 = pd.DataFrame.from_dict({
    'Month': ["Apr", "Jan", "Jul", "Oct"],
    'Mean TemperatureF': [61.956044, 32.133333, 32.133333, 32.133333]}).\
    set_index('Month')
weather1

weather1.reindex(year)
weather2 = weather1.reindex(year)

# weather1.reindex(year, method='ffill')
# weather3 = weather1.reindex(year, method='ffill')
weather1.reindex(year).ffill()
weather3 = weather1.reindex(year).ffill()

base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Baby names/"
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

base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/"
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

base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Summer Olympic medals/"
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


df_january = pd.DataFrame.from_dict({
    'Company': ["Acme Corp", "Hooli", "Initech", "MediaCore", "Streeplex"],
    'Units': [19, 17, 20, 10, 13]}).\
    set_index('Company')

df_february = pd.DataFrame.from_dict({
    'Company': ["Acme Corp", "Hooli", "Initech", "Vandelay Inc"],
    'Units': [15, 3, 13, 25]}).\
    set_index('Company')

df_january
df_february
df_january + df_february


base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/"
df_weather = pd.read_csv(base_dir + "pittsburgh2013.csv")

df_temps_f = \
    df_weather[['Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF']]
df_temps_f
df_temps_c = (df_temps_f - 32) * 5/9
df_temps_c

df_temps_c.columns = df_temps_c.columns.str.replace('F', 'C')
df_temps_c


base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/GDP/"
df_gdp = pd.read_csv(base_dir + "gdp_usa.csv", 
                     parse_dates=True, index_col='DATE')
df_gdp
# df_gdp.plot()

df_gdp_post2008 = df_gdp.loc["2008":, :]
# df_gdp_post2008.plot()

df_gdp_post2008.resample('A').last()
# df_gdp_post2008.resample('A').last().pct_change().plot()

# (df_gdp.resample('A').mean().pct_change() * 100).plot()

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
# df_gdp.plot()
# (df_gdp.pct_change() * 100).plot()


base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/"
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

# Appending & Concatenating Series

import pandas as pd

northeast = pd.Series(["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"])
south = pd.Series(["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL",
                   "KY", "MS", "TN", "AR", "LA", "OK", "TX"])
midwest = pd.Series(["IL", "IN", "MN", "MO", "NE", "ND", "SD", "IA", "KS", "MI",
                     "OH", "WI"])
west = pd.Series(["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA",
                  "HI", "OR", "WA"])

east = northeast.append(south)
east
east.index
east.loc[3]

new_east = northeast.append(south).reset_index(drop=True)
new_east
new_east.index
new_east.loc[3]

east = pd.concat([northeast, south])
east
east.index
east.loc[3]

new_east = pd.concat([northeast, south], ignore_index=True)
new_east
new_east.index
new_east.loc[3]


base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Summer Olympic medals/"

df_brnz = pd.read_csv(base_dir + "bronze_top5.csv", index_col='Country')
s_brnz = df_brnz['Total']
s_brnz

df_slvr = pd.read_csv(base_dir + "silver_top5.csv", index_col='Country')
s_slvr = df_slvr['Total']
s_slvr

df_gold = pd.read_csv(base_dir + "gold_top5.csv", index_col='Country')
s_gold = df_gold['Total']
s_gold


s_brnz.append(s_slvr)


base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Sales/"

df_jan = pd.read_csv(base_dir + "sales-jan-2015.csv", 
                     parse_dates=True, index_col='Date')
df_feb = pd.read_csv(base_dir + "sales-feb-2015.csv", 
                     parse_dates=True, index_col='Date')
df_mar = pd.read_csv(base_dir + "sales-mar-2015.csv", 
                     parse_dates=True, index_col='Date')

quarter1 = df_jan['Units'].append(df_feb['Units']).append(df_mar['Units'])
quarter1.loc["2015-01-27":"2015-02-02"]
quarter1.loc["2015-02-26":"2015-03-07"]


units = []

for month in [df_jan, df_feb, df_mar]:
    units.append(month['Units'])

quarter1 = pd.concat(units)
quarter1.loc["2015-01-27":"2015-02-02"]
quarter1.loc["2015-02-26":"2015-03-07"]


# Appending & Concatenating DataFrames

base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Baby names/"

df_names_1881 = pd.read_csv(base_dir + "names1881.csv", header=None)
df_names_1981 = pd.read_csv(base_dir + "names1981.csv", header=None)

df_names_1881.columns = ['name', 'gender', 'count']
df_names_1981.columns = ['name', 'gender', 'count']

df_names_1881.head(10)
df_names_1981.head(10)

df_names_1881['year'] = 1881
df_names_1981['year'] = 1981

df_names_1881.head(10)
df_names_1981.head(10)

df_combined_names = df_names_1881.append(df_names_1981, ignore_index=True)
df_combined_names.head()
df_combined_names.tail()

df_names_1881.shape
df_names_1981.shape
df_combined_names.shape

df_combined_names[df_combined_names['name'] == 'Morgan']
df_combined_names.loc[df_combined_names['name'] == 'Morgan', :]


weather_max = pd.DataFrame.from_dict({
    'Month': ["Jan", "Apr", "Jul", "Oct"],
    'Max TemperatureF': [68, 89, 91, 84]}).\
    set_index(['Month'])
weather_mean = pd.DataFrame.from_dict({
    'Month': ["Apr", "Aug", "Dec", "Feb", "Jan", "Jul",
              "Jun", "Mar", "May", "Nov", "Oct", "Sep"],
    'Mean TemperatureF': ["53.100000", "70.000000", "34.935484",
                          "28.714286", "32.354839", "72.870968",
                          "70.133333", "35.000000", "62.612903",
                          "39.800000", "55.451613", "63.766667"]}).\
    set_index(['Month'])

weather_max
weather_mean

weather = pd.concat([weather_max, weather_mean], axis='columns', sort=False)
weather


base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Summer Olympic medals/"
medals = []

for medal in ["bronze", "silver", "gold"]:
    file_name = "{}{}_top5.csv".format(base_dir, medal)
    columns = ['Country', medal]
    df_medal = \
        pd.read_csv(file_name, header=0, index_col='Country', names=columns)
    medals.append(df_medal)

df_medals = pd.concat(medals, axis='columns', sort=False)
df_medals


# Concatenation, Keys, & MultiIndexes

base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Summer Olympic medals/"
medals = []

medals_list = ["bronze", "silver", "gold"] 

for medal in medals_list:
    file_name = "{}{}_top5.csv".format(base_dir, medal)
    medal_df = pd.read_csv(file_name, index_col='Country')
    medals.append(medal_df)

df_medals = pd.concat(medals, keys=medals_list, axis='rows', sort=False)
df_medals


df_medals_sorted = df_medals.sort_index(level=0)
df_medals_sorted

df_medals_sorted.loc[('silver', 'Italy'), :]
df_medals_sorted.loc[(slice(None), 'Soviet Union'), :]

df_medals_sorted.loc['bronze', :]
df_medals_sorted.loc[('bronze', slice(None)), :]

idx = pd.IndexSlice
df_medals_sorted.loc[idx[:, 'United Kingdom'], :]
df_medals_sorted.loc[(slice(None), 'United Kingdom'), :]

uk1 = df_medals_sorted.loc[idx[:, 'United Kingdom'], :]
uk2 = df_medals_sorted.loc[(slice(None), 'United Kingdom'), :]
(uk1 == uk2).all().all()


base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Sales/"

df_jan = pd.read_csv(base_dir + "sales-jan-2015.csv", 
                     parse_dates=True, index_col='Date')
df_feb = pd.read_csv(base_dir + "sales-feb-2015.csv", 
                     parse_dates=True, index_col='Date')
df_mar = pd.read_csv(base_dir + "sales-mar-2015.csv", 
                     parse_dates=True, index_col='Date')

dataframes = [df_jan, df_feb, df_mar]
dataframes

quarter1 = pd.concat(dataframes, 
                     keys=['Hardware', 'Software', 'Service'],
                     axis=1)
quarter1

idx = pd.IndexSlice
quarter1.loc["2015-02-02":"2015-02-08", idx[:, 'Company']]


month_list = [('january', df_jan), 
              ('february', df_feb), 
              ('march', df_mar)]
month_list

month_dict = {}

for month_name, month_data in month_list:
    month_dict[month_name] = month_data.groupby('Company').sum()
month_dict

sales = pd.concat(month_dict)
sales

idx = pd.IndexSlice
sales.loc[idx[:, 'Mediacore'], :]


# Outer & Inner Joins

import numpy as np
import pandas as pd

A = np.arange(8).reshape(2, 4) + 0.1
B = np.arange(6).reshape(2, 3) + 0.2
C = np.arange(12).reshape(3, 4) + 0.3
A
B
C

np.hstack([A, B])
np.concatenate([A, B], axis=1)
(np.hstack([A, B]) == np.concatenate([A, B], axis=1)).all()

np.vstack([A, C])
np.concatenate([A, C], axis=0)
(np.vstack([A, C]) == np.concatenate([A, C], axis=0)).all()

# inner join

df_brnz
df_slvr
df_gold

medal_list = [df_brnz, df_slvr, df_gold]

pd.concat(medal_list, 
          axis='columns', 
          keys=['Bronze', 'Silver', 'Gold'], 
          join='inner')


base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/GDP/"

df_gdp_us = pd.read_csv(base_dir + "gdp_usa.csv",
                        parse_dates=True, index_col='DATE')
df_gdp_zh = pd.read_csv(base_dir + "gdp_china.csv",
                        parse_dates=True, index_col='Year')
df_gdp_us.index.name = 'Year'
df_gdp_zh.index.name = 'Year'
df_gdp_us.columns = ['US']
df_gdp_zh.columns = ['China']
df_gdp_us
df_gdp_zh

df_gdp_us_ann = df_gdp_us.resample('A').last().pct_change(10).dropna()
df_gdp_zh_ann = df_gdp_zh.resample('A').last().pct_change(10).dropna()
df_gdp_us_ann
df_gdp_zh_ann

gdp = pd.concat([df_gdp_zh_ann, df_gdp_us_ann],
                axis='columns',
                join='inner')
gdp
gdp.resample('10A').last()



# 3 Merging Data

# Merging DataFrames

import pandas as pd

population = pd.DataFrame.from_dict({
    'Zipcode': [16855, 15681, 18657, 17307, 15635],
    '2010 Census Population': [282, 5241, 11985, 5899, 220]
    })
population

cities = pd.DataFrame.from_dict({
    'Zipcode': [17545, 18455, 17307, 15705, 16833, 16220, 18618, 16855, 16623,
                15635, 15681, 18657, 15279, 17231, 18821],
    'City': ["MANHEIM", "PRESTON PARK", "BIGLERVILLE", "INDIANA", 
             "CURWENSVILLE", "CROWN", "HARVEYS LAKE", "MINERAL SPRINGS",
             "CASSVILLE", "HANNASTOWN", "SALTSBURG", "TUNKHANNOCK", 
             "PITTSBURGH", "LEMASTERS", "GREAT BEND"],
    'State': "PA"})
cities

pd.concat([cities, population], axis='columns', join='outer')
pd.concat([cities.set_index('Zipcode'), population.set_index('Zipcode')],
          axis='columns',
          join='outer')
pd.merge(cities, population)
pd.merge(cities, population, how='outer')
pd.merge(cities, population, how='outer', on='Zipcode')
pd.merge(cities, population, how='outer', 
         left_on='Zipcode', right_on='Zipcode')


base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Summer Olympic medals/"

df_brnz = pd.read_csv(base_dir + "bronze_top5.csv", index_col='Country')
df_brnz.columns = ['Bronze']
df_slvr = pd.read_csv(base_dir + "silver_top5.csv", index_col='Country')
df_slvr.columns = ['Silver']
df_gold = pd.read_csv(base_dir + "gold_top5.csv", index_col='Country')
df_gold.columns = ['Gold']

pd.merge(df_brnz, df_slvr, how='outer', on='Country')
pd.merge(df_brnz, df_gold, how='outer', on='Country')
df_brnz.\
    merge(df_slvr, how='outer', on='Country').\
    merge(df_gold, how='outer', on='Country')

df_brnz = pd.read_csv(base_dir + "Bronze.csv")
df_slvr = pd.read_csv(base_dir + "Silver.csv")
df_gold = pd.read_csv(base_dir + "Gold.csv")

df_brnz
df_slvr
df_gold

pd.merge(df_brnz, df_slvr)
pd.merge(df_brnz, df_gold)

df_brnz.loc[df_brnz['NOC'] == "IRL", :]
df_slvr.loc[df_slvr['NOC'] == "IRL", :]

pd.merge(df_brnz, df_slvr, how='outer')
pd.merge(df_brnz, df_slvr, how='outer', on=['NOC', 'Country'])
pd.merge(df_brnz, df_slvr, how='outer', on=['NOC', 'Country'],
         suffixes=['_bronze', '_silver'])

# df_brnz.columns = ['NOC', 'Country', 'Bronze']
# df_slvr.columns = ['NOC', 'Country', 'Silver']
# df_gold.columns = ['NOC', 'Country', 'Gold']

# df_brnz.merge(df_slvr, how='outer')
# df_brnz.merge(df_slvr, how='outer', on='NOC')
# df_brnz.merge(df_slvr, how='outer', on=['NOC', 'Country'])


counties = pd.DataFrame.from_dict({
    'CITY NAME': ["SALTSBURG", "MINERAL SPRINGS", "BIGLERVILLE", 
                  "HANNASTOWN", "TUNKHANNOCK"],
    'COUNTY NAME': ["INDIANA", "CLEARFIELD", "ADAMS", "WESTMORELAND", 
                    "WYOMING"]})
counties

cities

pd.merge(cities, counties, left_on='City', right_on='CITY NAME', how='outer')


# Joining DataFrames

df_brnz = pd.read_csv(base_dir + "Bronze.csv")
df_slvr = pd.read_csv(base_dir + "Silver.csv")
df_gold = pd.read_csv(base_dir + "Gold.csv")

df_brnz_5 = df_brnz.sort_values('Total', ascending=False).head(5)
df_slvr_5 = df_slvr.sort_values('Total', ascending=False).head(5)
df_gold_5 = df_gold.sort_values('Total', ascending=False).head(5)

pd.merge(df_brnz_5, df_slvr_5, how='inner', on=['NOC', 'Country'],
         suffixes=['_bronze', '_silver'])

pd.merge(df_brnz_5, df_slvr_5, how='left', on=['NOC', 'Country'],
         suffixes=['_bronze', '_silver'])
pd.merge(df_brnz_5, df_gold_5, how='left', on=['NOC', 'Country'],
         suffixes=['_bronze', '_gold'])


population = pd.DataFrame.from_dict({
    'Zip Code ZCTA': [57538, 59916, 37660, 2860],
    '2010 Census Population': [322, 130, 40038, 45199]}).\
    set_index('Zip Code ZCTA')
population

unemployment = pd.DataFrame.from_dict({
    'Zip': [2860, 46167, 1097, 80808],
    'unemployment': [0.11, 0.02, 0.33, 0.07],
    'participants': [34447, 4800, 42, 4310]}).\
    set_index('Zip')
unemployment

# .join() matches on the index
population.join(unemployment, how='left')  # left is default
population.join(unemployment, how='right')
population.join(unemployment, how='inner')
population.join(unemployment, how='outer')


# Ordered Merges

base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Sales/"

software = pd.read_csv(base_dir + "feb-sales-Software.csv").\
    sort_values('Date')
hardware = pd.read_csv(base_dir + "feb-sales-Hardware.csv").\
    sort_values('Date')

software
hardware

pd.merge(hardware, software)
pd.merge(hardware, software, how='outer')
pd.merge(hardware, software, how='outer').sort_values('Date')

pd.merge_ordered(hardware, software) # default how='outer'
pd.merge_ordered(hardware, software, 
                 on=['Date', 'Company'], suffixes=['_hardware', '_software'])



# 4 Case Study - Summer Olympics

# Medals in the Summer Olympics

import pandas as pd

base_dir = "/home/hynso/Documents/Learning/PythonDataCamp/2_merging_dataframes_pandas/Summer Olympic medals/"

df_ioc_codes = \
    pd.read_csv(base_dir + 
                "Summer Olympic medalists 1896 to 2008 - IOC COUNTRY CODES.csv")
df_editions = \
    pd.read_csv(base_dir +
                "Summer Olympic medalists 1896 to 2008 - EDITIONS.tsv",
                sep='\t')
df_medalists = \
    pd.read_csv(base_dir +
                "Summer Olympic medalists 1896 to 2008 - ALL MEDALISTS.tsv",
                sep='\t', skiprows=4)

df_ioc_codes.shape
df_ioc_codes.head()
df_ioc_codes.memory_usage(index=True).sum() / 1024**2  # size in MB

df_editions.shape
df_editions.head()

df_medalists.shape
df_medalists.head()
df_medalists.tail()
df_medalists.memory_usage(index=True).sum() / 1024**2  # size in MB


df_edits = df_editions[['Edition', 'Grand Total', 'City', 'Country']]
df_edits.shape
df_edits.head()
df_edits.tail()

df_codes = df_ioc_codes[['Country', 'NOC']]
df_codes.shape
df_codes.head()
df_codes.tail()


# Quantifying Performance

df_medals = df_medalists[['Athlete', 'NOC', 'Medal', 'Edition']]
df_medals.shape
df_medals.head()
df_medals.tail()

medal_counts = \
    pd.pivot_table(df_medals, 
                   values=['Athlete'], index='Edition', columns='NOC', 
                   aggfunc='count')


totals = df_edits.set_index('Edition')['Grand Total']
totals

fractions = medal_counts.divide(totals, axis='rows')
fractions.head()
fractions.tail()


mean_fractions = fractions.expanding().mean()
mean_fractions

fractions_change = mean_fractions.pct_change() * 100
fractions_change = fractions_change.reset_index()

fractions_change.head()
fractions_change.tail()


# Reshaping and Plotting

hosts = pd.merge(df_edits, df_codes, how='left')
hosts.head()
hosts = hosts[['Edition', 'NOC']].set_index('Edition')
hosts.head()

hosts.loc[hosts.NOC.isnull()]
hosts.loc[1972, 'NOC'] = 'FRG'
hosts.loc[1980, 'NOC'] = 'URS'
hosts.loc[1988, 'NOC'] = 'KOR'

hosts = hosts.reset_index()
hosts


reshaped = pd.melt(fractions_change, id_vars='Edition', value_name='Change')
reshaped.shape
fractions_change.shape

chn = reshaped[reshaped['NOC'] == 'CHN']
chn.tail()


merged = pd.merge(reshaped, hosts, how='inner')
merged.head()

influence = merged.set_index('Edition').sort_index()
influence


import matplotlib.pyplot as plt

change = influence['Change']

ax = change.plot(kind='bar')

ax.set_ylabel("% Change of Host Country Medal Count")
ax.set_title("Is there a Host Country Advantage?")
ax.set_xticklabels(df_edits['City'])

plt.show()
