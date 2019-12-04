# 1_manipulating_dataframes_pandas.py


# Indexing DataFrames

import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd
import numpy as np

month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
eggs = [47, 110, 221, 77, 132, 205]
salt = [12.0, 50.0, 89.0, 87.0, np.nan, 60.0]
spam = [17, 31, 72, 20, 52, 55]

# sales = \
#     pd.DataFrame.from_dict(
#         {"month": month, "eggs": eggs, "salt": salt, "spam": spam}
#     )
# sales
# sales = sales.set_index(sales['month'], drop=True, append=False)
# del sales['month']
# sales

sales = pd.read_csv("./1_manipulating_dataframes_pandas/sales/sales.csv",
                    index_col='month')
sales

sales['salt']['Jan']
sales.eggs['Mar']

# .loc and .iloc accessors -- prefereable to direct indexing (as above)
sales.loc['May', 'spam']
sales.iloc[4, 2]

sales_new = sales[['salt', 'eggs']]
sales_new


# Slicing DataFrames

sales
sales['eggs']  # results in pandas Series

sales['eggs'][1:4]  # Series result
sales['eggs'][4]    # value result

sales.loc[:, 'eggs':'salt']  # DataFrame result
sales.loc['Jan':'Apr', :]
sales.loc['Mar':'May', 'salt':'spam']

sales.iloc[2:5, 1:]

sales.loc['Jan':'May', ['eggs', 'spam']]
sales.iloc[[0, 4, 5], 0:2]

sales['eggs']    # result Series
sales[['eggs']]  # result DataFrame


election = \
    pd.read_csv(
        "./1_manipulating_dataframes_pandas/pennsylvania2012_turnout.csv",
        index_col='county'
    )
election.head()

election.loc['Perry':'Potter', :]
election.loc['Potter':'Perry':-1, :]  # reverse row order

election.loc[:, :'Obama']
election.loc[:, 'Obama':'winner']
election.loc[:, 'Romney':]

rows = ['Philadelphia', 'Centre', 'Fulton']
cols = ['winner', 'Obama', 'Romney']
election.loc[rows, cols]


# Filtering DataFrames

sales.salt > 60
sales[sales.salt > 60]
enough_salt_sold = sales.salt > 60
sales[enough_salt_sold]

sales[(sales.salt >= 50) & (sales.eggs < 200)]
sales[(sales.salt >= 50) | (sales.eggs < 200)]

sales2 = sales.copy(deep=True)
sales2['bacon'] = [0, 0, 50, 60, 70, 80]
sales2

sales2.loc[:, sales2.all()]  # keep columns where _all_ values are non-zero
sales2.loc[:, sales2.any()]  # keep columns where _any_ values are non-zero

sales2.loc[:, sales2.isnull().any()]  # keep columns where _any_ values are NaN
sales2.loc[:, sales2.isnull().all()]  # keep columns where _all_ values are NaN

# keep cols where _any_ values are not NaN
sales2.loc[:, sales2.notnull().any()]
# keep cols where _all_ values are not NaN
sales2.loc[:, sales2.notnull().all()]

sales.dropna(how='any')  # drop rows where all row-wise tuple values are NaN
sales.dropna(how='all')  # drop rows where all row-wise tuple values are NaN

# filter `eggs` column based on `salt` column values
sales.eggs[sales.salt > 55]

sales3 = sales.copy(deep=True)
sales3.eggs[sales.salt > 55] += 5
sales
sales3
sales3 - sales

high_turnout = election.turnout > 70.0
high_turnout_df = election[high_turnout]
high_turnout_df

too_close = election.margin < 1.0
election2 = election.copy()
election2.winner[too_close] = np.nan
election2
election2.loc[election2.winner.isnull(), :]

df_titanic = pd.read_csv("./1_manipulating_dataframes_pandas/titanic.csv")
df_titanic.info()
df_t = df_titanic.loc[:, ['age', 'cabin']]
df_t.shape
df_t.dropna(how='any')
df_t.dropna(how='any').shape
df_titanic.dropna(thresh=1000, axis='columns')
df_titanic.dropna(thresh=1000, axis='columns').info()


# Transforming DataFrames

sales.floordiv(12)
np.floor_divide(sales, 12)


def dozens(n):
    return n // 12


sales.apply(dozens)  # use `apply` on non-index columns

sales.apply(lambda n: n // 12)

sales2 = sales.copy()
sales2['dozens_of_eggs'] = sales.eggs.floordiv(12)
sales2

sales2.index
sales2.index.str.upper()
sales2.index = sales2.index.str.upper()
sales2

# use `map` on an index (unfortunate)
sales2.index = sales2.index.map(str.lower)
sales2

sales2['salty_eggs'] = sales2.salt + sales2.dozens_of_eggs
sales2

df_wx = pd.read_csv("./1_manipulating_dataframes_pandas/pittsburgh2013.csv")
df_wx


def to_celsius(F):
    return 5 / 9 * (F - 32)


df_celsius = \
    df_wx[['Mean TemperatureF', 'Mean Dew PointF']].apply(to_celsius)
df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']
df_celsius

election.head()
election2 = election.copy()
red_vs_blue = {'Obama': "blue", 'Romney': "red"}
election2['color'] = election['winner'].map(red_vs_blue)
election2.head()


turnout_zscore = zscore(election2['turnout'])
type(turnout_zscore)  # numpy.ndarray
election2['turnout_zscore'] = turnout_zscore
election2.head()


# Index Objects and Labeled Data

# Indexes are immutable (like dictionary keys), and...
# they're homogenous in data type (like numpy arrays)


prices = [10.70, 10.86, 10.74, 10.71, 10.79]
shares = pd.Series(prices)
shares

days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
shares = pd.Series(prices, index=days)
shares

shares.index
type(shares.index)
shares.index[2]
shares.index[:2]
shares.index[-2:]

shares.index.name  # `None`
shares.index.name = 'weekday'
shares.index.name
shares.index
shares

# shares.index[2] = 'Wednesday' # Error... indexes are immutable
shares.index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
shares
shares.index.name = "Weekday"
shares


sales = pd.read_csv("./1_manipulating_dataframes_pandas/sales/sales.csv",
                    index_col='month')
sales

sales
sales2 = sales.copy()
new_idx = [idx.upper() for idx in sales2.index]
sales2.index = new_idx
sales2

sales2.index.name = 'MONTHS'
sales2
sales2.columns.name
sales2.columns.name = 'PRODUCTS'
sales2

sales2 = sales.copy()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales2.index = months
sales2


# Hierarchical Indexing

state = ['CA', 'CA', 'NY', 'NY', 'TX', 'TX']
month = [1, 2, 1, 2, 1, 2]
eggs = [47, 110, 221, 77, 132, 205]
salt = [12.0, 50.0, 89.0, 87.0, np.nan, 60.0]
spam = [17, 31, 72, 20, 52, 55]

sales_mi_raw = pd.DataFrame.from_dict({'state': state,
                                       'month': month,
                                       'eggs': eggs,
                                       'salt': salt,
                                       'spam': spam})
sales_mi = sales_mi_raw.set_index(['state', 'month'])
sales_mi

sales_mi.loc[['CA', 'TX']]
sales_mi.loc[['CA', 'TX'], :]  # same as above
sales_mi['CA':'TX']
sales_mi.loc[['eggs', 'spam']]
sales_mi[['eggs', 'spam']]
sales_mi.loc[['CA', 'TX'], ['eggs', 'spam']]

sales_mi.sort_index()

sales_mi2 = sales_mi_raw.copy()
sales_mi2 = sales_mi2.set_index('state')
sales_mi2
sales_mi2.loc['NY']

sales_mi.loc[('NY', 1), :]
sales_mi.loc[(['CA', 'TX'], 2), :]
sales_mi.loc[(slice(None), 2), :]
sales_mi.loc[(['CA', 'NY'], slice(None)), :]


# Pivoting DataFrames


id = [1, 2, 3, 4]
treatment = ['A', 'A', 'B', 'B']
gender = ['F', 'M', 'F', 'M']
response = [5, 3, 8, 9]

df_trials = pd.DataFrame.from_dict({'id': id,
                                    'treatment': treatment,
                                    'gender': gender,
                                    'response': response})
df_trials

df_trials.pivot(index='treatment', columns='gender', values='response')
df_trials.pivot(index='treatment', columns='gender')

pd.pivot_table(df_trials, index='id', columns=['treatment', 'gender'],
               values='response')
pd.pivot_table(df_trials, index='id', columns=['gender', 'treatment'],
               values='response')

df_users = pd.read_csv("./1_manipulating_dataframes_pandas/users.csv",
                       index_col=0)
df_users

df_users.pivot(index='weekday', columns='city', values='visitors')
df_users.pivot(index='weekday', columns='city', values='signups')
df_users.pivot(index='weekday', columns='city')

df_trials
df_trials_idx = df_trials.set_index(['treatment', 'gender'])
df_trials_idx
df_trials_idx.unstack(level='gender')
df_trials_idx.unstack(level='treatment')

df_trials_by_gender = df_trials_idx.unstack(level='gender')
df_trials_by_gender
df_trials_by_gender.stack(level='gender')

df_trials_idx
df_trials_idx.swaplevel(0, 1)

df_trials_idx.swaplevel(0, 1).sort_index()

df_users
df_users2 = df_users.set_index(['city', 'weekday']).sort_index()
df_users2
df_users2.unstack(level='city')
df_users2.unstack(level='weekday')
df_users2.unstack(level='city').stack('city')  # yields df_user2

df_users2.unstack(level='city').stack('city').swaplevel(0, 1).sort_index()
foo = df_users2.unstack(level='city').stack(
    'city').swaplevel(0, 1).sort_index()
df_users2 == foo

df_trials
df_trials.pivot(index='treatment', columns='gender', values='response')
foo = df_trials.pivot(index='treatment', columns='gender', values='response')
foo['treatment'] = foo.index
foo.columns.name = None
foo.index.name = None
foo = foo.reset_index(drop=True)
foo
pd.melt(foo, id_vars=['treatment'])
pd.melt(foo, id_vars=['treatment'], value_vars=['F', 'M'])
pd.melt(foo, id_vars=['treatment'], value_vars=['F', 'M'],
        var_name='gender', value_name='response')

df_users
df_users.\
    pivot(index='weekday', columns='city', values='visitors').\
    reset_index().\
    melt(id_vars=['weekday'], value_vars=['Austin', 'Dallas'],
         value_name='visitors').\
    set_index(['weekday', 'city']).\
    sort_index()

df_users
pd.melt(df_users, id_vars=['weekday', 'city'],
        value_vars=['visitors', 'signups'])

df_users
df_users.pivot_table(index='weekday', columns='city')
df_users.pivot_table(index='weekday', columns='city', values='visitors')

df_users
df_users.pivot_table(index='weekday', aggfunc='count')
df_users.pivot_table(index='weekday', values='visitors', aggfunc='count')

df_users
df_users.pivot_table(index='weekday', aggfunc=sum)
df_users.pivot_table(index='weekday', aggfunc=sum, margins=True)


# Categoricals and GroupBy


# df_sales = pd.read_csv("./1_manipulating_dataframes_pandas/sales/sales.csv",
#                        index_col='month')
# df_sales
df_sales = pd.DataFrame.from_dict(
    {
        'weekday': ["Sun", "Sun", "Mon", "Mon"],
        'city': ["Austin", "Dallas", "Austin", "Dallas"],
        'bread': [139, 237, 326, 456],
        'butter': [20, 45, 70, 98]
    }
)
df_sales

df_sales.groupby('weekday').count()
df_sales.groupby('weekday').sum()
df_sales.groupby('weekday').mean()
df_sales.groupby('weekday').std()
df_sales.groupby('weekday').first()
df_sales.groupby('weekday').last()
df_sales.groupby('weekday').min()
df_sales.groupby('weekday').max()
df_sales.groupby('weekday').median()

df_sales.groupby('weekday')['bread'].sum()
df_sales.groupby('weekday').sum()['bread']
df_sales.groupby('weekday')['bread'].sum() == \
    df_sales.groupby('weekday').sum()['bread']

df_sales.groupby('weekday')[['bread', 'butter']].sum()

df_sales.groupby(['city', 'weekday']).count()

s_customers = pd.Series(["Dave", "Alice", "Bob", "Alice"])

df_sales.groupby(s_customers)['bread'].sum()

df_sales['weekday']
df_sales['weekday'].value_counts()
df_sales['weekday'].unique()

df_sales2 = df_sales.copy()
df_sales2['weekday'] = df_sales2['weekday'].astype('category')
df_sales2['weekday']
# using category types in dataframes uses less memory and speeds up `groupby`

df_titanic = pd.read_csv("./1_manipulating_dataframes_pandas/titanic.csv")
df_titanic

df_titanic.groupby('pclass')['survived'].count()

df_titanic.groupby(['embarked', 'pclass'])['survived'].count()


# GroupBy and Aggregation

# df_life = \
#     pd.read_csv("./1_manipulating_dataframes_pandas/gapminder_tidy.csv",
#                 index_col='Country')
life_fname = \
    "https://s3.amazonaws.com/assets.datacamp.com/production/course_1650/datasets/life_expectancy.csv"
df_life = pd.read_csv(life_fname, index_col='Country')
df_life

regions_fname = \
    "https://s3.amazonaws.com/assets.datacamp.com/production/course_1650/datasets/regions.csv"
df_regions = pd.read_csv(regions_fname, index_col='Country')
df_regions

df_life.groupby(df_regions['region'])['2010'].mean()

df_sales
df_sales.groupby('city')[['bread', 'butter']].agg(['min', 'max'])
df_sales.groupby('city')[['bread', 'butter']].agg(['sum', 'count', 'mean'])


def data_range(series):
    return series.max() - series.min()


df_sales
df_sales.groupby('city')[['bread', 'butter']].agg(data_range)
df_sales.groupby('weekday')[['bread', 'butter']].agg(data_range)
df_sales.\
    groupby(s_customers)[['bread', 'butter']].\
    agg({'bread': 'sum', 'butter': data_range})

df_titanic
df_titanic.groupby('pclass')[['age', 'fare']].agg(['max', 'median'])
df_titanic.\
    groupby('pclass')[['age', 'fare']].\
    agg(['max', 'median']).\
    loc[:, ('age', 'max')]
df_titanic.\
    groupby('pclass')[['age', 'fare']].\
    agg(['max', 'median']).\
    loc[:, ('fare', 'median')]

df_gapmind = \
    pd.read_csv("./1_manipulating_dataframes_pandas/gapminder_tidy.csv",
                index_col=['Year', 'region', 'Country'])
df_gapmind


def spread(series):
    return series.max() - series.min()


df_gapmind.\
    groupby(level=['Year', 'region']).\
    agg({'population': 'sum', 'child_mortality': 'mean', 'gdp': spread})


df_gapmind.\
    groupby(level=['Year', 'region'])['population'].\
    mean().\
    plot()

df_gapmind.\
    groupby(level=['Year', 'region'])['fertility'].\
    mean().\
    unstack().\
    plot()

df_gapmind.\
    groupby(level=['Year', 'region'])['life'].\
    mean().\
    unstack().\
    plot()

df_gapmind.\
    groupby(level=['Year', 'region'])['population'].\
    mean().\
    unstack().\
    plot()

df_gapmind.\
    groupby(level=['Year', 'region'])['child_mortality'].\
    mean().\
    unstack().\
    plot()

df_gapmind.\
    groupby(level=['Year', 'region'])['gdp'].\
    mean().\
    unstack().\
    plot()

df_sales = pd.read_csv("./1_manipulating_dataframes_pandas/sales/sales-feb-2015.csv",
                       index_col='Date', parse_dates=True)
df_sales

df_sales.groupby(df_sales.index.strftime('%a'))['Company'].count()
df_sales.groupby(df_sales.index.strftime('%a'))['Units'].mean()


# GroupBy and Transformation


df_auto = pd.read_csv("./1_manipulating_dataframes_pandas/auto-mpg.csv")
df_auto

zscore(df_auto['mpg'])
df_auto.groupby('yr')['mpg'].transform(zscore)

# The `agg` method applies reduction
# The `transform` method applies a function element-wise to groups
# For when a split-apply-combine operations aren't neatly solved by aggregration
# or transformation, there's the `apply` method.


def zscore_with_year_and_name(group):
    df = pd.DataFrame(
        {
            'mpg': zscore(group['mpg']),
            'year': group['yr'],
            'name': group['name']
        }
    )
    return df

df_auto.groupby('yr').apply(zscore_with_year_and_name)
foo = df_auto.groupby('yr')
foo.apply(zscore_with_year_and_name)

df_gapmind = \
    pd.read_csv("./1_manipulating_dataframes_pandas/gapminder_tidy.csv",
                index_col='Country')

df_gapmind_2010 = df_gapmind[df_gapmind['Year'] == 2010]
df_gapmind_2010

standardized = \
    df_gapmind_2010.\
    groupby('region')['life', 'fertility'].\
    transform(zscore)
standardized

outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)
outliers

gm_outliers = df_gapmind_2010.loc[outliers]
gm_outliers

df_titanic = pd.read_csv("./1_manipulating_dataframes_pandas/titanic.csv")
df_titanic.head()

def impute_median(series):
    return series.fillna(series.median())

titanic_age = \
    df_titanic.groupby(['sex', 'pclass'])['age'].transform(impute_median)
titanic_age
df_titanic.tail(10)


def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)':z , 'regional spread(gdp)':s})

regional = df_gapmind_2010.groupby('region')
reg_disp = regional.apply(disparity)
reg_disp.loc[['United States', 'United Kingdom', 'China'], :]


# GroupBy and Filtering

df_auto.groupby('yr')['mpg'].mean()

splitting = df_auto.groupby('yr')
type(splitting)
splitting.groups
type(splitting.groups)
splitting.groups.keys()

for group_name, group in splitting:
    avg = group['mpg'].mean()
    print(group_name, avg)

for group_name, group in splitting:
    avg = group.loc[group['name'].str.contains('chevrolet'), 'mpg'].mean()
    print(group_name, avg)

chevy_means = \
    { 
        year:group.loc[group['name'].str.contains('chevrolet'), 'mpg'].mean() 
        for year, group in splitting
    }
chevy_means

chevy = df_auto['name'].str.contains('chevrolet')
df_auto.groupby(['yr', chevy])['mpg'].mean()


def c_deck_survival(gr):
    c_passengers = gr['cabin'].str.startswith('C').fillna(False)
    return gr.loc[c_passengers, 'survived'].mean()

by_sex = df_titanic.groupby('sex')
c_surv_by_sex = by_sex.apply(c_deck_survival)
c_surv_by_sex


df_sales = \
    pd.read_csv("./1_manipulating_dataframes_pandas/sales/sales-feb-2015.csv",
                index_col='Date', parse_dates=True)
df_sales

by_company = df_sales.groupby('Company')
by_company.groups
by_company['Units'].sum()
by_com_sum = by_company['Units'].sum()
by_com_sum
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)
by_com_filt

under10 = (df_titanic['age'] < 10).map({True: 'under 10', False: 'over 10'})
survived_mean_1 = df_titanic.groupby(under10)['survived'].mean()
survived_mean_1

survived_mean_2 = df_titanic.groupby([under10, 'pclass'])['survived'].mean()
survived_mean_2



# Case Study: Olympic Medals

import pandas as pd

df_medals = \
    pd.read_csv("./1_manipulating_dataframes_pandas/all_medalists.csv")
df_medals.head()

df_medals[df_medals['NOC'] == "USA"]
df_medals[df_medals['NOC'] == "USA"].groupby('Edition')['Medal'].count()

df_medals['NOC'].value_counts()
df_medals['NOC'].value_counts().head(15)

df_counted = \
    df_medals.pivot_table(index='NOC', values='Athlete', columns='Medal',
                          aggfunc='count')
df_counted['totals'] = df_counted.sum(axis='columns')
df_counted = df_counted.sort_values('totals', ascending=False)
df_counted.head(15)


# Understanding the Column Labels

df_medals.head(10)
df_medals.loc[:, ['Event_gender', 'Gender']]
df_medals.loc[:, ['Event_gender', 'Gender']].drop_duplicates()

df_medals.groupby(['Event_gender', 'Gender'])['Medal'].count()

df_medals.loc[(df_medals.Event_gender == "W") & (df_medals.Gender == "Men"), :]


# Constructing Alternative Country Rankings

df_medals.groupby('NOC')['Sport'].nunique()
df_medals.groupby('NOC')['Sport'].nunique().sort_values(ascending=False)
df_medals.\
    groupby('NOC')['Sport'].\
    nunique().\
    sort_values(ascending=False).\
    head(15)

during_cold_war = \
    (df_medals['Edition'] >= 1952) & (df_medals['Edition'] <= 1988)
is_usa_or_urs = df_medals['NOC'].isin(['USA', 'URS'])
df_medals_cold_war = df_medals.loc[during_cold_war & is_usa_or_urs, :]
df_medals_cold_war

df_medals_cold_war.groupby('NOC')['Sport'].nunique()
df_medals_cold_war.\
    groupby('NOC')['Sport'].\
    nunique().\
    sort_values(ascending=False)

df_medals.\
    pivot_table(index='Edition',
                columns='NOC',
                values='Athlete',
                aggfunc='count')
df_medals.\
    pivot_table(index='Edition',
                columns='NOC',
                values='Athlete',
                aggfunc='count').\
    loc[1952:1988, ['USA', 'URS']]
df_medals.\
    pivot_table(index='Edition',
                columns='NOC',
                values='Athlete',
                aggfunc='count').\
    loc[1952:1988, ['USA', 'URS']].\
    idxmax(axis='columns')
df_medals.\
    pivot_table(index='Edition',
                columns='NOC',
                values='Athlete',
                aggfunc='count').\
    loc[1952:1988, ['USA', 'URS']].\
    idxmax(axis='columns').\
    value_counts()


# Reshaping DataFrames for Visualization

df_medals[df_medals['NOC'] == 'USA']
df_medals[df_medals['NOC'] == 'USA'].\
    groupby(['Edition', 'Medal'])['Athlete'].\
    count()
df_medals[df_medals['NOC'] == 'USA'].\
    groupby(['Edition', 'Medal'])['Athlete'].\
    count().\
    unstack().\
    loc[:, ['Bronze', 'Silver', 'Gold']]
df_medals[df_medals['NOC'] == 'USA'].\
    groupby(['Edition', 'Medal'])['Athlete'].\
    count().\
    unstack().\
    loc[:, ['Bronze', 'Silver', 'Gold']].\
    plot()
df_medals[df_medals['NOC'] == 'USA'].\
    groupby(['Edition', 'Medal'])['Athlete'].\
    count().\
    unstack().\
    loc[:, ['Bronze', 'Silver', 'Gold']].\
    plot.area()

df_medals_categ = df_medals.copy()
df_medals_categ['Medal'] = \
    pd.Categorical(values=df_medals['Medal'],
                   categories=['Bronze', 'Silver', 'Gold'],
                   ordered=True)
df_medals_categ[df_medals['NOC'] == "USA"].\
    groupby(['Edition', 'Medal'])['Athlete'].\
    count().\
    unstack(level='Medal').\
    plot.area()