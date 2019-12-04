# 0_pandas_foundations.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

outcomes = []

for x in range(10):
    coin = np.random.randint(0, 2)

    if coin == 0:
        outcomes.append("heads")
    else:
        outcomes.append("tails")

print(outcomes)

# Accumulating steps

tails = [0]

for x in range(10):
    coin = np.random.randint(0, 2)

    tails.append(tails[x] + coin)

print(tails)


# Another accumulator

def roll_die():
    return np.random.randint(1, 7)


random_walk = [0]

for n in range(100):
    step = random_walk[n]

    dice = roll_die()

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + roll_die()

    random_walk.append(step)

print(random_walk)

# Another...

all_walks = []

for i in range(10):
    random_walk = [0]

    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)

        random_walk.append(step)

    all_walks.append(random_walk)

print(all_walks)

# Another still...

all_walks = []

for i in range(10):
    random_walk = [0]

    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)

        random_walk.append(step)

    all_walks.append(random_walk)

np_aw = np.array(all_walks)

plt.plot(np_aw)
plt.show()

plt.clf()

np_aw_t = np.transpose(np_aw)

plt.plot(np_aw_t)
plt.show()

# Another still yet...

all_walks = []

for i in range(250):
    random_walk = [0]

    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)

        if np.random.rand() <= 0.001:
            step = 0

        random_walk.append(step)

    all_walks.append(random_walk)

np_aw_t = np.transpose(np.array(all_walks))

print(np_aw_t)

plt.plot(np_aw_t)
plt.show()

# And yet another still yet...

# Simulate random walk 500 times
all_walks = []

for i in range(500):
    random_walk = [0]

    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)
        if np.random.rand() <= 0.001:
            step = 0

        random_walk.append(step)

    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np_aw_t[-1, :]
print(ends)

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()

print(len(ends[ends >= 60]) / len(ends))

# Graphical EDA

import pandas as pd
import matplotlib.pyplot as plt

df_iris = pd.read_csv("./iris.csv")
df_iris.head()

df_iris.plot(x='sepal.length', y='sepal.width')
plt.show()

df_iris.plot(x='sepal.length', y='sepal.width', kind='scatter')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

df_iris.plot(y='sepal.length', kind='box')
plt.ylabel("Sepal Width (cm)")
plt.show()

df_iris.plot(y='sepal.length', kind='hist')
plt.xlabel("Sepal Length (cm)")
plt.show()

df_iris.plot(y='sepal.length', kind='hist',
             bins=30, range=(4, 8), density=True)
plt.xlabel("Sepal Length (cm)")
plt.show()

df_iris.plot(y='sepal.length', kind='hist',
             bins=30, range=(4, 8), cumulative=True, density=True)
plt.xlabel("Sepal Length (cm)")
plt.title("Cumulative distribution function (CDF)")
plt.show()

df_iris.plot(y='sepal.length', kind='hist')
# df_iris.plt.hist(y='sepal.length')  # doesn't seem to work
df_iris.hist()  # works but has different syntax

# Statistical EDA

df_iris['sepal.length'].count()
df_iris[['sepal.length', 'sepal.width']].count()
df_iris.count()

df_iris['sepal.length'].mean()
df_iris[['sepal.length', 'sepal.width']].mean()
df_iris.mean()

df_iris['sepal.length'].std()
df_iris[['sepal.length', 'sepal.width']].std()
df_iris.std()

df_iris['sepal.length'].median()
df_iris[['sepal.length', 'sepal.width']].median()
df_iris.median()

df_iris['sepal.length'].quantile(0.25)
df_iris[['sepal.length', 'sepal.width']].quantile(0.25)
df_iris.quantile(0.25)

df_iris['sepal.length'].quantile([0.25, 0.75])
df_iris[['sepal.length', 'sepal.width']].quantile([0.25, 0.75])
df_iris.quantile([0.25, 0.75])
df_iris.quantile([0.25, 0.75]).T

df_iris.min()

df_iris.max()

df_iris.describe()

# Indexing time series

df_wx = pd.read_csv("./weather_data_austin_2010.csv",
                    parse_dates=True, index_col="Date")
df_wx.info()
df_wx.head()
df_wx.loc['2010-01-01 00':'2010-01-01 06']

df_wx.head(10)
# Hourly => Daily downsample
df_wx_day = df_wx.resample('D').mean()
df_wx_day

# Hourly => Minutely upsample
df_wx_min = df_wx.resample('T').ffill()
df_wx_min

# Moving average / Rolling mean
df_wx_mva = df_wx.rolling(window=24).mean()
df_wx_mva

df_wx_day = df_wx.resample('D').mean()
df_wx_day_mva = df_wx_day.rolling(window=7).mean()
df_wx_day_mva

df_wx_day['Temperature'].plot()
df_wx_day_mva['Temperature'].plot()
plt.show()
plt.clf()

# import pytz

df_wx2 = df_wx.copy(deep=True)
pd.Series(df_wx2.index). \
    dt.tz_localize('US/Central', ambiguous='NaT', nonexistent='NaT')

########################

df = pd.read_csv("./NOAA_QCLCD_2011_hourly_13904.txt", header=None)

column_labels = 'Wban,date,Time,StationType,sky_condition,sky_conditionFlag,visibility,visibilityFlag,wx_and_obst_to_vision,wx_and_obst_to_visionFlag,dry_bulb_faren,dry_bulb_farenFlag,dry_bulb_cel,dry_bulb_celFlag,wet_bulb_faren,wet_bulb_farenFlag,wet_bulb_cel,wet_bulb_celFlag,dew_point_faren,dew_point_farenFlag,dew_point_cel,dew_point_celFlag,relative_humidity,relative_humidityFlag,wind_speed,wind_speedFlag,wind_direction,wind_directionFlag,value_for_wind_character,value_for_wind_characterFlag,station_pressure,station_pressureFlag,pressure_tendency,pressure_tendencyFlag,presschange,presschangeFlag,sea_level_pressure,sea_level_pressureFlag,record_type,hourly_precip,hourly_precipFlag,altimeter,altimeterFlag,junk'
column_labels_list = column_labels.split(',')
df.columns = column_labels_list

list_to_drop = [
    'sky_conditionFlag',
    'visibilityFlag',
    'wx_and_obst_to_vision',
    'wx_and_obst_to_visionFlag',
    'dry_bulb_farenFlag',
    'dry_bulb_celFlag',
    'wet_bulb_farenFlag',
    'wet_bulb_celFlag',
    'dew_point_farenFlag',
    'dew_point_celFlag',
    'relative_humidityFlag',
    'wind_speedFlag',
    'wind_directionFlag',
    'value_for_wind_character',
    'value_for_wind_characterFlag',
    'station_pressureFlag',
    'pressure_tendencyFlag',
    'pressure_tendency',
    'presschange',
    'presschangeFlag',
    'sea_level_pressureFlag',
    'hourly_precip',
    'hourly_precipFlag',
    'altimeter',
    'record_type',
    'altimeterFlag',
    'junk'
]
df_dropped = df.drop(list_to_drop, axis='columns')
print(df_dropped.head())

df_dropped['date'] = df_dropped['date'].astype(str)
df_dropped['Time'] = df_dropped['Time'].apply(lambda x: '{:0>4}'.format(x))
date_string = df_dropped['date'] + df_dropped['Time']
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')
df_clean = df_dropped.set_index(date_times)
print(df_clean.head())

print(df_clean.loc['2011-06-20 08':'2011-06-20 09', 'dry_bulb_faren'])
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'],
                                           errors='coerce')
print(df_clean.loc['2011-06-20 08':'2011-06-20 09', 'dry_bulb_faren'])
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'],
                                       errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'],
                                            errors='coerce')
df_clean['visibility'] = pd.to_numeric(df_clean['visibility'],
                                       errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'],
                                            errors='coerce')

print(df_clean['dry_bulb_faren'].median())
print(df_clean.loc['2011-04':'2011-06', 'dry_bulb_faren'].median())
print(df_clean.loc['2011-01', 'dry_bulb_faren'].median())

daily_mean_2011 = df_clean.resample('D').mean()
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values
df_climate = pd.read_csv("./weather_data_austin_2010.csv",
                         parse_dates=True, index_col='Date')
daily_climate = df_climate.resample('D').mean()
daily_temp_climate = daily_climate['Temperature']
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())

is_sky_clear = df_clean['sky_condition'] == 'CLR'
sunny = df_clean[is_sky_clear]
sunny_daily_max = sunny.resample('D').max()
sunny_daily_max.head()

is_sky_overcast = df_clean['sky_condition'].str.contains('OVC')
overcast = df_clean[is_sky_overcast]
print(overcast)
overcast_daily_max = overcast.resample('D').max()
overcast_daily_max.head()

sunny_daily_max_mean = sunny_daily_max.mean()
overcast_daily_max_mean = overcast_daily_max.mean()
sunny_daily_max_mean - overcast_daily_max_mean

weekly_mean = df_clean[['visibility', 'dry_bulb_faren']].resample('W').mean()
print(weekly_mean.corr())
weekly_mean.plot(subplots=True)
plt.show()

is_sky_clear = df_clean['sky_condition'] == 'CLR'
resampled = is_sky_clear.resample('D')
print(resampled)
sunny_hours = resampled.sum()
total_hours = resampled.count()
sunny_fraction = sunny_hours / total_hours
print(sunny_fraction)

monthly_max = \
    df_clean[['dew_point_faren', 'dry_bulb_faren']].resample('M').max()
monthly_max.plot(kind='hist', bins=8, alpha=0.5, subplots=True)
plt.show()

weekly_max = \
    df_clean[['dew_point_faren', 'dry_bulb_faren']].resample('W').max()
weekly_max.plot(kind='hist', bins=8, alpha=0.5, subplots=True)
plt.show()

august_max = df_climate.loc['2010-08', 'Temperature'].max()
print(august_max)
august_2011 = df_clean.loc['2011-08', 'dry_bulb_faren'].resample('D').max()
print(august_2011)
august_2011_high = august_2011[august_2011 > august_max]
print(august_2011_high)
august_2011_high.plot(kind='hist', bins=25, density=True, cumulative=True)



# 