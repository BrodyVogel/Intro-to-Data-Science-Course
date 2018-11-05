#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:34:22 2017

@author: vogebr01
"""

import pandas as pd
import numpy as np
import datetime 
import plotly.plotly as py
import plotly.graph_objs as go


sports = pd.read_csv('/Users/brodyvogel/Desktop/ANLY 501/Group 5 Project 2/SPORTS_FINAL.csv')
stocks = pd.read_csv('/Users/brodyvogel/Desktop/ANLY 501/Group 5 Project 2/STOCKS_FINAL.csv', thousands = ',')

stocks = stocks[stocks.columns[2:]]
sports = sports[sports.columns[1:]]

### Date transition from Excel wasn't clean, so we have to fix the format
index = 0
for entry in stocks['Date']:
    stocks.iloc[index, 2] = entry.replace('"', '')
    index += 1
stdate=[]
for i in range(0,len(stocks['Date'])):
    stdate.append(datetime.datetime.strptime(stocks['Date'][i].strip('"'), '%b %d, %Y').strftime('%d-%b-%y'))

spdate=[]
for i in range(0,len(sports['Date'])):
    spdate.append(datetime.datetime.strptime(sports['Date'][i], '%d-%b-%y').strftime('%d-%b-%y'))
stocks['Date']=stdate
sports['Date']=spdate
closeCols = []
for column in stocks.columns:
    if 'Close' in column:
        closeCols.append(column)

### Keep pandas happy with consistent column lengths
neededLength = len(stocks.loc[:, 'Date'])

### Create the percentage change variables in the stocks data
for column in closeCols:
    i = 0
    closeChange = []
    closes = stocks.loc[:, column]
    closes = closes[pd.isnull(closes) == False]
    for price in closes[:-1]:
        change = (float(closes[i]) - float(closes[i + 1]))/float(closes[i + 1])
        closeChange.append(change)
        i += 1
    pos = stocks.columns.get_loc(column)
    newColName = column.split(' ')
    newColName = ' '.join(newColName[:-1])  + ' Percent Change'
    for x in range(len(closeChange),neededLength):
        closeChange.append(None)
    stocks.insert(pos + 1, newColName, closeChange)
    
### Get a list of the percentage change attributes
percentCols = []
for column in stocks.columns:
    if 'Percent' in column:
        percentCols.append(column)
stchangedata=stocks[['Date','NASDAQ Percent Change','DOW JONES Percent Change','S&P Percent Change','MARATHON Percent Change','CHEVRON Percent Change','EXXON Percent Change','FRANKLIN GOLD AND_PRECIOUS METALS Percent Change','FIRST EAGLE GOLD Percent Change','STURM & RUGER (GUNS) Percent Change','AMERICAN OUTDOOR (GUNS) Percent Change','ANHEUSER-BUSCH Percent Change','VICEX (SIN) Percent Change','RYDEX (SIN) Percent Change','UTILITIES Percent Change','HEALTHCARE Percent Change','IT Percent Change','BONDS Percent Change']]
spchangedata=sports[['Date','NFL Games','MLB Games','NBA Games','Total Games']]
result=pd.merge(stchangedata, spchangedata, on='Date', how='outer')
stweek=[]
for i in range(0,len(stocks['Date'])):
    stweek.append(datetime.datetime.strptime(stocks['Date'][i], '%d-%b-%y').strftime('%A'))
    
spweek=[]
for i in range(0,len(sports['Date'])):
    spweek.append(datetime.datetime.strptime(sports['Date'][i], '%d-%b-%y').strftime('%A'))
stchangedata['Weekday']=stweek
spchangedata['Weekday']=spweek
stchangedata=stchangedata[['Date','Weekday','NASDAQ Percent Change','DOW JONES Percent Change','S&P Percent Change','MARATHON Percent Change','CHEVRON Percent Change','EXXON Percent Change','FRANKLIN GOLD AND_PRECIOUS METALS Percent Change','FIRST EAGLE GOLD Percent Change','STURM & RUGER (GUNS) Percent Change','AMERICAN OUTDOOR (GUNS) Percent Change','ANHEUSER-BUSCH Percent Change','VICEX (SIN) Percent Change','RYDEX (SIN) Percent Change','UTILITIES Percent Change','HEALTHCARE Percent Change','IT Percent Change','BONDS Percent Change']]
spchangedata=spchangedata[['Date','Weekday','NFL Games','MLB Games','NBA Games','Total Games']]
newresult=pd.merge(stchangedata, spchangedata, on=['Date','Weekday'], how='outer')
sunnew=newresult.loc[newresult['Weekday']=='Sunday']
satnew=newresult.loc[newresult['Weekday']=='Saturday']
sunnew=sunnew.iloc[::-1]
satnew=satnew.iloc[::-1]
haharesult=newresult.copy()
haharesult=haharesult.drop('NFL Games', axis=1)
haharesult=haharesult.drop('MLB Games', axis=1)
haharesult=haharesult.drop('NBA Games', axis=1)
haharesult=haharesult.drop('Total Games', axis=1)
values = haharesult['NASDAQ Percent Change'].loc[haharesult['Weekday']=='Monday']
values=values.iloc[1:]
### Bin the percentage change variables by size (create a binned variable)
for column in percentCols:
    newColName = column.split(' ')
    newColName = ' '.join(newColName[:-2]) + ' Change Category'
    pos = stocks.columns.get_loc(column)
    jumps = []
    q0, q5, q20, q35, q50, q80, q95 = np.nanpercentile(stocks[column], [0, 5, 20, 35, 50, 80, 95])
    for entry in stocks[column]:
        if entry > q95:
            jumps.append('BIG JUMP')
        elif entry > q80:
            jumps.append('JUMP')
        elif entry > q50:
            jumps.append('LITTLE MOVEMENT')
        elif entry > q35:
            jumps.append('LITTLE MOVEMENT')
        elif entry > q5:
            jumps.append('DIP')
        elif entry > q0:
            jumps.append('BIG DIP')
        else:
            jumps.append(None)
    for x in range(len(jumps),neededLength):
        jumps.append(None)
    stocks.insert(pos + 1, newColName, jumps)

companys=['NASDAQ Percent Change','DOW JONES Percent Change','S&P Percent Change','MARATHON Percent Change','CHEVRON Percent Change','EXXON Percent Change','FRANKLIN GOLD AND_PRECIOUS METALS Percent Change','FIRST EAGLE GOLD Percent Change','STURM & RUGER (GUNS) Percent Change','AMERICAN OUTDOOR (GUNS) Percent Change','ANHEUSER-BUSCH Percent Change','VICEX (SIN) Percent Change','RYDEX (SIN) Percent Change','UTILITIES Percent Change','HEALTHCARE Percent Change','IT Percent Change','BONDS Percent Change']
for company in companys:
    values = haharesult[company].loc[haharesult['Weekday']=='Monday']
    values=values.iloc[1:]
    valuelist=values.tolist()
    nassun = sunnew[company].values
    i_custom = 0  # starting index on your iterator for your custom list
    for i in range(len(nassun)):
        if np.isnan(nassun[i]):
            nassun[i] = valuelist[i_custom]
            i_custom += 1  # increase the index
    sunnew[company]=nassun
for company in companys:
    values = haharesult[company].loc[haharesult['Weekday']=='Monday']
    values=values.iloc[1:]
    valuelist=values.tolist()
    nassun = satnew[company].values
    i_custom = 0  # starting index on your iterator for your custom list
    for i in range(len(nassun)):
        if np.isnan(nassun[i]):
            nassun[i] = valuelist[i_custom]
            i_custom += 1  # increase the index
    satnew[company]=nassun
newresult=newresult[0:10788]
newresult.isnull().sum()

df=newresult.head(n=3211)
df=df.dropna()
df.loc[:,'NASDAQ Percent Change'] *= 1000
df.loc[:,'S&P Percent Change'] *= 1000
df.loc[:,'DOW JONES Percent Change'] *= 1000
df['Date'] = pd.to_datetime(df['Date'])
trace_high = go.Scatter(
    x=df['Date'],
    y=df['NASDAQ Percent Change'],
    yaxis = 'percent change ',
    name = "NASDAQ Percent Change",
    line = dict(color = '#ad730f',width=2),
    opacity = 0.8)
    
trace_med = go.Scatter(
    x=df['Date'],
    y=df['S&P Percent Change'],
    name = 'S&P Percent Change',
    line = dict(color = '#6cd334',width=2),
    opacity = 0.8)
trace_ye = go.Scatter(
    x=df['Date'],
    y=df['DOW JONES Percent Change'],
    name = 'DOW JONES Percent Change',
    line = dict(color = '#9915c1',width=2),
    opacity = 0.8)
trace_low = go.Scatter(
    x=df['Date'],
    y=df['Total Games'],
    name = "Total Games",
    line = dict(color = '#00f7ea',width=4),
    opacity = 0.8)


data = [trace_high,trace_med,trace_ye,trace_low]
layout = dict(
    title='Time Series For Stocks %change versus Total Games',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1w',
                     step='week',
                     stepmode='backward'),
                dict(count=6,
                     label='6w',
                     step='week',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    ),
    yaxis=dict(
        title='Total Games'
    ),
    yaxis2=dict(
        title='Percent Change',
        titlefont=dict(
            color='#7F7F7F'
        ),
        tickfont=dict(
            color='#7F7F7F'
        ),
        overlaying='y',
        side='right'
    )
)

fig = dict(data=data, layout=layout)
py.plot(fig, filename = 'Time Series For Stocks %change versus Total Games')

df=stocks.head(n=800)
df['dif']=df['NASDAQ High']-df['NASDAQ Low']

df=df.dropna()
df['dif']=df['dif'].astype(int)
df.loc[:,'dif'] *= 100
trace1 = go.Scatter3d(
    x=df['NASDAQ Volume'],
    y=df['NASDAQ Change Category'],
    z=df['NASDAQ Close'],
    text=df['NASDAQ'],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        sizeref=300,
        size=df['dif'],
        color = df['NASDAQ Percent Change'],
        colorscale = 'Viridis',
        colorbar = dict(title = 'Percent Change'),
        line=dict(color='rgb(140, 140, 170)')
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='Discover Nasdaq Perent change to Nasdaq stocks movement')
fig=dict(data=data, layout=layout)
py.plot(fig, filename='3DBubble')