#!/usr/bin/env python


# IMPORTING DATA IN PYTHON (PART 1)


# 1 Importing Data from the Internet


# Importing Flat Files from the Web

base_dir = \
    "/home/hynso/Documents/Learning/PythonDataCamp/4_importing_data_python_2/"

from urllib.request import urlretrieve

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/" + \
      "wine-quality/winequality-white.csv"

urlretrieve(url, base_dir + 'winequality-white.csv')


from urllib.request import urlretrieve
import pandas as pd

url = "https://s3.amazonaws.com/assets.datacamp.com/production/" + \
    "course_1606/datasets/winequality-red.csv"

urlretrieve(url, base_dir + "winequality-red.csv")

df = pd.read_csv(base_dir + "winequality-red.csv", sep=';')
df


import matplotlib.pyplot as plt

url = "https://s3.amazonaws.com/assets.datacamp.com/production/" + \
    "course_1606/datasets/winequality-red.csv"

df = pd.read_csv(url, sep=';')
df.head()

pd.DataFrame.hist(df.iloc[:, 0:1])
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()


url = "http://s3.amazonaws.com/assets.datacamp.com/course/" + \
    "importing_data_into_r/latitude.xls"

xls = pd.read_excel(url, sheet_name=None)

xls.keys()
xls.items()

xls['1700'].head()


# HTTP Requests to Import Files from the Web

base_dir = \
    "/home/hynso/Documents/Learning/PythonDataCamp/4_importing_data_python_2/"

# To extract html from wikipedia homepage...

# 1. Import req'd functions
from urllib.request import urlopen, Request
# 2. Specifify the URL
url = "https://www.wikipedia.org/"
# 3. Package the GET request using the function `Request`
request = Request(url)
# 4. Send the request using `urlopen` and catch the response w/ var `response`
response = urlopen(request) # returns http response obj w/ assoc'd read method
# 5. Apply the read method to the response object
html = response.read()
# 6. Close the response
response.close()

print(html)


# Doing the same as above with `requests` package

# 1. Import requests package
import requests
# 2. Specify the url
url = "https://www.wikipedia.org"
# 3. Package and send the request, then catch the response w/ 1 fxn: `get`
r = requests.get(url)
# 4. Apply `text` method(?)/property(?) to the response, which returns html
#    as a string
text = r.text

print(text)


from urllib.request import urlopen, Request

url = "http://www.datacamp.com/teach/documentation"

request = Request(url)

response = urlopen(request) 

html = response.read()
print(type(response))

response.close()

print(html)


import requests

url = "http://www.datacamp.com/teach/documentation"

r = requests.get(url)

text = r.text

print(text)


# 3. Scraping the Web in Python

from bs4 import BeautifulSoup
import requests

url = "https://www.crummy.com/software/BeautifulSoup/"

r = requests.get(url)

html_doc = r.text

soup = BeautifulSoup(html_doc)

soup.prettify()
soup.title
soup.get_text()

for link in soup.find_all('a'):
    print(link.get('href'))



# 2 Interacting with APIs to Import Data form the Web

# Introduction to APIs and JSONs

import json

with open(base_dir + 'snakes.json', mode='r') as json_file:
    json_data = json.load(json_file)

type(json_data)

for key, value in json_data.items():
    print(key + ":", value)


with open(base_dir + 'a_movie.json', mode='r') as json_file:
    json_data = json.load(json_file)

for k in json_data.keys():
    print(k + ":", json_data[k])


# APIs and Interacting with the World Wide Web

import requests

url = "http://www.omdbapi.com/?apikey=dcb7bf0d&t=hackers"

r = requests.get(url)
json_data = r.json()

for key, value in json_data.items():
    print(key + ":", value)


import requests

url = "http://www.omdbapi.com/?apikey=dcb7bf0d&t=the+big+lebowski"
r = requests.get(url)
r.text


import requests

url = "http://www.omdbapi.com/?apikey=dcb7bf0d&t=the+big+lebowski"
r = requests.get(url)
json_data = r.json()

for k in json_data.keys():
    print(k + ":", json_data[k])


import requests

url = "https://en.wikipedia.org/w/" + \
    "api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza"
r = requests.get(url)
json_data = r.json()

json_data
json_data['query']
json_data['query']['pages']
json_data['query']['pages']['24768']
json_data['query']['pages']['24768']['extract']



# 3 Diving Deep into the Twitter API

# The Twitter API and Authentication

# Check out tw_auth.py


