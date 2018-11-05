#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:34:09 2017

@author: Ye Zhang, Brody Vogel, Jinghao Yan, Zihao Li
"""

### This script scrapes data from Yahoo! Finance for certain stocks since 1975, then outputs the 
### information in .csv format

### import statements
import pandas as pd
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import json


### Instantiate the dataframe
STOCKS = pd.DataFrame()

### Create the webdriver (Use chromedriver - geckodriver is very finnicky)
driver = webdriver.Chrome(executable_path = '/Users/brodyvogel/Downloads/chromedriver')

### urls for the stocks for which we want data
urls = ['https://finance.yahoo.com/quote/%5EIXIC/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/%5EDJI/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/%5EGSPC/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/MRO/history?period1=104400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/CVX/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/XOM/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/FKRCX/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/SGGDX/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/RGR/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/AOBC/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/BUD/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/VICEX/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/RYLIX/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/FKUTX/history?period1=315637200&period2=1507176000&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/VHT/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/VGT/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d',
        'https://finance.yahoo.com/quote/FBNDX/history?period1=157784400&period2=1507089600&interval=1d&filter=history&frequency=1d']

### names of the stocks for which we want data
names = ['NASDAQ', 'DOW JONES', 'S&P', 'MARATHON', 'CHEVRON', 'EXXON', 'FRANKLIN GOLD AND_PRECIOUS METALS',
         'FIRST EAGLE GOLD', 'STURM & RUGER (GUNS)', 'AMERICAN OUTDOOR (GUNS)', 'ANHEUSER-BUSCH', 'VICEX (SIN)',
         'RYDEX (SIN)', 'UTILITIES', 'HEALTHCARE', 'IT', 'BONDS']

### used to go through the names of stocks later
indexer = 0
### used to make sure we only get one date variable
stopper = 1

### get the data
for url in urls:
    ### load the page
    driver.get(url)
    
    ### scroll to the bottom of the (very long) page
    for i in range(850):
        elm=driver.find_element_by_tag_name('body')
        elm.send_keys(Keys.END)


    ### snag the source data, in case we want to check it later
    html = driver.page_source

    ### make the soup and find the daily data
    soup = BeautifulSoup(html,"html.parser")
    tableData = soup.find("table", attrs={"class":"W(100%) M(0)"})
    rows = tableData.findAll('td',attrs={"class":"Py(10px) Ta(start)"})
    ### if there is a dividend payout or stock split, the script needs to ignore those entries
    helper = []
    for i in range(len(rows)-1):
        if 'Dividend' in rows[i + 1].text or 'Stock' in rows[i + 1].text:
            x = 'x'
        elif 'Dividend' in rows[i].text or 'Stock' in rows[i].text:
            x = 'x'
        else:
            helper.append(rows[i])
    rows = helper
    ### make our date variable
    date=[]
    for i in range(len(rows)):
        date.append(json.dumps(rows[i].text))
    if stopper == 1:
        STOCKS['Date'] = date
        stopper = 2
       
    ### again with the dividend and stock split corrections
    price=tableData.findAll('td',attrs={"class":"Py(10px)"})
    helper2 = []
    for i in range(len(price)-1):
        if 'Dividend' in price[i + 1].text or 'Stock' in price[i + 1].text:
            x = x
        elif 'Dividend' in price[i].text or 'Stock' in price[i].text:
            x = x
        else:
            helper2.append(price[i])
            
    ### the final list of data that we want
    price = helper2
    
    
    ### go through the data and create lists of the specifics, like open, high, close, etc.
    finalprice=[]
    for iq in range(len(price)):
        if 'Dividend' not in price[iq].text and 'Stock' not in price[iq].text:
            finalprice.append(json.dumps(price[iq].text))
    index=range(len(price))
    index[2::2]
    p_open=[]
    for i in index[1::7]:
        if 'Dividend' not in price[i].text and 'Stock' not in price[i].text:
            p_open.append(json.dumps(price[i].text))
    p_high=[]
    for i in index[2::7]:
        if 'Dividend' not in price[i].text and 'Stock' not in price[i].text:
            p_high.append(json.dumps(price[i].text))
    p_low=[]
    for i in index[3::7]:
        if 'Dividend' not in price[i].text and 'Stock' not in price[i].text:
            p_low.append(json.dumps(price[i].text))
    p_close=[]
    for i in index[4::7]:
        if 'Dividend' not in price[i].text and 'Stock' not in price[i].text:
            p_close.append(json.dumps(price[i].text))
    p_adjclose=[]
    for i in index[5::7]:
        if 'Dividend' not in price[i].text and 'Stock' not in price[i].text:
            p_adjclose.append(json.dumps(price[i].text))
    p_volumn=[]
    for i in index[6::7]:
        if 'Dividend' not in price[i].text and 'Stock' not in price[i].text:
            p_volumn.append(json.dumps(price[i].text))

    fdate=[item.replace('"', '') for item in date]

    fp_open=[item.replace('"', '') for item in p_open]

    fp_high=[item.replace('"', '') for item in p_high]

    fp_low=[item.replace('"', '') for item in p_low]

    fp_close=[item.replace('"', '') for item in p_close]

    fp_adjclose=[item.replace('"', '') for item in p_adjclose]

    fp_volumn=[item.replace('"', '') for item in p_volumn]


    ### make a dictionary from our lists of data
    d={'Date' : pd.Series(fdate),
       'Open' : pd.Series(fp_open),
       'High' : pd.Series(fp_high),
       'Low'  : pd.Series(fp_low),
       'Close' : pd.Series(fp_close),
       'AdjClose' : pd.Series(fp_adjclose),
       'Volumn': pd.Series(fp_volumn)
       }
    
    ### create the dataframe from the dictionary of data
    name = names[indexer]
    STOCKS[name] = name
    op = name + ' Open'
    STOCKS[op] = d['Open']
    high = name + ' High'
    STOCKS[high] = d['High']
    low = name + ' Low'
    STOCKS[low] = d['Low']
    close = name + ' Close'
    STOCKS[close] = d['Close']
    volume = name + ' Volume'
    STOCKS[volume] = d['Volumn']
    
    indexer += 1

### write everything to a csv
STOCKS.to_csv('STOCKS_RAW.csv')
