#!/usr/bin/env python

# IMPORTING DATA IN PYTHON (PART 1)


# 1 Introduction and Flat Files

# Welcome to the Course!

base_dir = \
    "/home/hynso/Documents/Learning/PythonDataCamp/3_importing_data_python_1/"

filename = base_dir + "seaslug.txt"
file = open(filename, mode='r')
text = file.read()
text
print(text)
file.close()

with open(filename, mode='r') as file:
    print(file.read())



#- ! ls # for jupyter notebooks only

file = open(base_dir + "moby_dick.txt", mode='r')

print(file.read())
print(file.closed)

file.close()
print(file.closed)


with open(base_dir + "moby_dick.txt", mode='r') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())


# Importing Flat Files Using NumPy

import numpy as np

filename = base_dir + "mnist_kaggle_some_rows.csv"
data = np.loadtxt(filename, delimiter=',')
data
type(data)

data_str = np.loadtxt(filename, delimiter=',', dtype=str)
data_str
type(data_str)


digits = np.loadtxt(filename, delimiter=',')
type(digits)

im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))
im_sq

import matplotlib.pyplot as plt

plt.imshow(im_sq, cmap='Greys', interpolation='nearest')
plt.show()


data_sel = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=[0, 2])
data_sel


filename = base_dir + "seaslug.txt"

data = np.loadtxt(filename, delimiter='\t', dtype=str)
data[0]
data[1]

data_float = np.loadtxt(filename, delimiter='\t', dtype=float, skiprows=1)
data_float[0]
data_float[9]
data_float

import matplotlib.pyplot as plt

plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel("time (min.)")
plt.ylabel("percentage of larvae")
plt.show()


data = np.genfromtxt(base_dir + "titanic_sub.csv", delimiter=',', names=True,
                     dtype=None, encoding=None)
data
type(data)
data[0]
np.shape(data)
data['PassengerId']
data['Sex']

data = np.recfromcsv(base_dir + "titanic_sub.csv", encoding=None)
data
data[0]
np.shape(data)
# data['PassengerId']  # Doesn't work... why?
# data['Sex']          # Doesn't work... why?


# Importing Flat Files Using pandas

import pandas as pd

filename = base_dir + "titanic_sub.csv"
data = pd.read_csv(filename)
data.head()
type(data)

data_array = data
type(data_array)
data_array[:5, :]


filename = base_dir + "mnist_kaggle_some_rows.csv"
data = pd.read_csv(filename, nrows=5, header=0)
data.head()

data_array = data.values
data_array



# 2 Importing Data from Other File Types

import pickle

fruit = {'peaches': 13, 'apples': 4, 'oranges': 11}

file = open(base_dir + "pickled_fruit.pkl", mode='wb')
pickle.dump(fruit, file)
file.closed
file.close()

with open(base_dir + "pickled_fruit.pkl", mode='rb') as file:
    data = pickle.load(file)
data

import pandas as pd

file = base_dir + "battledeath.xlsx"
data = pd.ExcelFile(file)
data.sheet_names

df1 = data.parse(sheet_name=data.sheet_names[0])
df2 = data.parse(sheet_name=data.sheet_names[1])
df1
df2


import os

wd = os.getcwd()
wd
os.listdir(wd)


import pickle

with open(base_dir + "pickled_fruit.pkl", mode = 'rb') as file:
    d = pickle.load(file)

d
type(d)


import pandas as pd

file = base_dir + "battledeath.xlsx"
xls = pd.ExcelFile(file)
xls.sheet_names

df1 = xls.parse('2002')
df1
df2 = xls.parse('2004')
df2
df3 = xls.parse(1)
(df3 == df2).all().all()


df1 = xls.parse(0, skiprows=[0], names=['Country', 'AAM due to War (2002)'])
df1.head()
df2 = xls.parse(1, usecols=[0], skiprows=[0], names=['Country'])
df2.head()


# Importing SAS/Stata Files Using pandas

import pandas as pd
from sas7bdat import SAS7BDAT

with SAS7BDAT(base_dir + 'sales.sas7bdat') as file:
    df_sas = file.to_data_frame()
df_sas.head()

df_sas = pd.read_sas(base_dir + "sales.sas7bdat")
df_sas.head()

df_stata = pd.read_stata(base_dir + "disarea.dta")
df_stata.head()


# Importing HDF5 Files

import h5py

filename = base_dir + "L-L1_LOSC_4_V1-1126259446-32.hdf5"
data = h5py.File(filename, mode='r')
type(data)
data.keys()

type(data['meta'])
data['meta'].keys()
data['meta']['Description'].value
data[('meta')].keys()
data[('meta')][('Description')].value

data['strain'].keys()
data[('strain')][('Strain')].value
type(data[('strain')][('Strain')].value)


# Importing MATLAB Files -- DON'T CARE ABOUT THIS

# import scipy.io

# filename = base_dir + "foobar.mat"
# mat = scipy.io.loadmat(filename)
# type(mat)
