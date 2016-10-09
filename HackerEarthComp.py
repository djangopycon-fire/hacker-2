
# coding: utf-8

# ## HackerEarth Competition notebook

# ### Import modules

# In[249]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

# for prediction, use machine learning
from sklearn import preprocessing, cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[250]:

inputData = pd.read_csv('ign.csv')


# In[251]:

inputData.head()


# ### Listing the platforms with the most "Editor's Choice" awards

# In[252]:

res = []

# we groupby platform
# groupby -> platform as first and rest as dataframe
# data contains rows grouped 
for platform, data in inputData.groupby('platform'):
    c = data[data['editors_choice'] == 'Y'].shape[0]
    res.append((platform, c))

res.sort(key=lambda x: x[1], reverse=True)
res = pd.DataFrame.from_records(res, columns=('Platform', 'Number of EC Awards'))
res.sort()


# #### Determine if the number of games by a platform in a year have any effect on the editors choice awards?

# In[253]:

# we can check how many games per platform are released in a year and how many gets award, calculate percentage
# can use above code, nah lets write again (remove comment)
calcPer = []

for platform, data in inputData.groupby('platform'):
    c = data[data['editors_choice'] == 'Y'].shape[0]
    calcPer.append((platform, c, data.shape[0], float(c)/data.shape[0]))

calcPer = pd.DataFrame.from_records(calcPer, columns=('Platform', 'Total Awards', 'Total Games', 'Award Rate'))
calcPer
    


# ** Well I think that the total number of games released on a platform in a year have no effect on the number of editor's choice awards presented to games for a platform. **

# ### Calculate Macintosh's average award count?

# In[254]:

# number of awards given to Macintosh
number_of_awards_for_macintosh = res.loc[res['Platform'] == 'Macintosh']['Number of EC Awards']

# total number of games released on Macintosh platform
macplatform = inputData.groupby('platform')['platform']
total_mac_games = macplatform.get_group('Macintosh').count()

# Macintosh average award count
c = float(number_of_awards_for_macintosh) / total_mac_games
print c


# ** So from above we can infer that for every two Mac games released and reviewed, one gets the Editor's Choice award. Total number of Macintosh games is 81 and total number of awards awarded are 40 (roughly half) **

# ### Finding optimal month for releasing a game?

# In[255]:

# we'll just group by total games released in a given month, months are in number (1-12)
# better to have a list of months for better view

import calendar

# inputData['release_month'] = inputData['release_month'].apply(lambda m: calendar.month_abbr[m])

# # inputData.head()
# d = inputData.groupby('release_month')['release_month']
# d.count()

res_months = []

for month, data in inputData.groupby('release_month'):
    res_months.append((calendar.month_abbr[month], data.shape[0]))

res_months = pd.DataFrame.from_records(res_months, columns=('month', 'count'))
res_months


# In[256]:

# plotting for better view (just cause we can)
res_months[['count']].plot.bar(x = res_months['month'], legend=True)


# ** So from the above graph we can infer that the optimal month for releasing game is November followed by October. Meaning companies have been releasing most of the games in October-November period. **

# ### Analyse the percentage growth in the gaming industry over the years

# In[257]:

# we can check for the total number of games released in the years

tgy = []

for year, data in inputData.groupby('release_year'):
    tgy.append((year, data.shape[0]))

tgy = pd.DataFrame.from_records(tgy, columns=('Year', 'Total Games'))
tgy


# *** It would seem that the release of video games has decreased over the years, lets plot a graph to better view it. ***

# In[258]:

# let plot this one too for better view
tgy[['Total Games']].plot.bar(x = tgy['Year'], legend=True)


# ** We can confirm that the growth rate of the games industry is decreasing with the years as less and less games are released every year. Also it can be that the data is not proper or is incomplete with more older games. Also the data doesn't consist of revenue, maybe companies are releasing less games but each game is earning a lot of revenue. **

# ## Lets build a predictive model for predicting which games will win editor's choice awards in a given year?

# In[259]:

inputData.describe()

# target column
target_col = ['editors_choice']

# numerical variables
num_cols = ['score', 'release_year', 'release_month', 'release_day']
# categorical variables
cat_cols = ['score_phrase', 'platform', 'genre']

other_col = ['Unnamed: 0']
extra_cols = ['url', 'title']

# combine num_cols and cat_cols
num_cat_cols = num_cols + cat_cols


# In[260]:

# check missing values
inputData.isnull().any()


# In[261]:

# create label encoders for categorical features
for item in cat_cols:
    num = LabelEncoder()
    
    inputData[item] = num.fit_transform(inputData[item].astype('str'))

# target variable is also categorical, so convert it
inputData['editors_choice'] = num.fit_transform(inputData['editors_choice'].astype('str'))


# In[262]:

# create features
features=list(set(list(inputData.columns))-set(target_col)-set(other_col)-set(extra_cols))
features


# In[263]:

X = inputData[list(features)].values
y = inputData['editors_choice'].values


# In[264]:

# split the data into train data and test data (70-30)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)


# In[265]:

# lets train
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)


# In[266]:

print clf.score(X_test, y_test)

