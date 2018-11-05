#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:59:26 2017

@author: Brody Vogel, Jinghao Yan, Ye Zhang, and Zihao Li
"""

import pandas as pd
import datetime
import math
import numpy as np
import plotly.plotly as ply
import plotly.graph_objs as go
import pandas_datareader.data as web
import networkx as nx
import matplotlib.pyplot as plt
import re

### Read in our data
sports = pd.read_csv('/Users/brodyvogel/Desktop/ANLY 501/Group 5 Project 2/SPORTS_FINAL.csv')
stocks = pd.read_csv('/Users/brodyvogel/Desktop/ANLY 501/Group 5 Project 2/STOCKS_FINAL.csv', thousands = ',')

### Duplicate index from previous cleaning needs fixed
stocks = stocks[stocks.columns[2:]]
sports = sports[sports.columns[1:]]

### Date transition from Excel wasn't clean, so we have to fix the format
index = 0
for entry in stocks['Date']:
    stocks.iloc[index, 0] = entry.replace('"', '')
    index += 1

### Creates the Outcomes variables denoting Wins, Losses, and No Games for the sports
    ### dataset 
for column in sports.columns[:-4]:
    result = []
    for entry in sports.loc[:, column]:
        if pd.isnull(entry) == True:
            result.append(0)
        elif str(entry)[0] == 'W':
            result.append(1)
        else:
            result.append(-1)
    newColName = column + ' Outcomes'
    pos = sports.columns.get_loc(column)
    sports.insert(pos + 1, newColName, result)
    
### Need a slightly different dataframe for the last viz
stdate=[]
for i in range(0,len(stocks['Date'])):
    stdate.append(datetime.datetime.strptime(stocks['Date'][i].strip('"'), '%b %d, %Y').strftime('%d-%b-%y'))

spdate=[]
for i in range(0,len(sports['Date'])):
    spdate.append(datetime.datetime.strptime(sports['Date'][i], '%d-%b-%y').strftime('%d-%b-%y'))

stocks1 = stocks.copy()
sports1 = sports.copy()

stocks1['Date']=stdate
sports1['Date']=spdate
    

### For building our new percentage change variable
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
        if pd.isnull(closes[i + 1]) == False:
            change = (float(closes[i]) - float(closes[i + 1]))/float(closes[i + 1])
            closeChange.append(change)
            i += 1
        else:
            ### dummy variable
            t = 0
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
    
stocks1 = stocks.copy()
sports1 = sports.copy()

stocks1['Date']=stdate
sports1['Date']=spdate


##Function 
def DataBuilder(teamOutcomes, stockChangePer):
  
    TeamWin = []
    TeamLose = []
    NoGame = []

    indexer = 0
    testTeam = []

    for day in sports['Date']:
        date = datetime.datetime.strptime(day, '%d-%b-%y')
        date = date + datetime.timedelta(days = 1)
        date2 = date + datetime.timedelta(days = 2)
        date3 = date + datetime.timedelta(days = 3)
        
        try:
            row = date.strftime('%b %d, %Y')
            row1 = date2.strftime('%b %d, %Y')
            row2 = date3.strftime('%b %d, %Y')

            if len(stocks.loc[stocks['Date'] == row]) != 0:  
                check = stocks.loc[stocks['Date'] == row]
                
                if sports[teamOutcomes][indexer] == 0:
                    testTeam.append(0)
                    NoGame.append(float(check[stockChangePer]))
                elif sports[teamOutcomes][indexer] == 1:
                    testTeam.append(1)
                    TeamWin.append(float(check[stockChangePer]))
                else:
                    testTeam.append(-1)
                    TeamLose.append(float(check[stockChangePer]))
                    
            elif len(stocks.loc[stocks['Date'] == row1]) != 0:
                check = stocks.loc[stocks['Date'] == row1]
                
                if sports[teamOutcomes][indexer] == 0:
                    testTeam.append(0)
                    NoGame.append(float(check[stockChangePer]))
                elif sports[teamOutcomes][indexer] == 1:
                    testTeam.append(1)
                    TeamWin.append(float(check[stockChangePer]))
                else:
                    testTeam.append(-1)
                    TeamLose.append(float(check[stockChangePer]))

            elif len(stocks.loc[stocks['Date'] == row2]) != 0:
                check = stocks.loc[stocks['Date'] == row2]
                
                if sports[teamOutcomes][indexer] == 0:
                    testTeam.append(0)
                    NoGame.append(float(check[stockChangePer]))
                elif sports[teamOutcomes][indexer] == 1:
                    testTeam.append(1)
                    TeamWin.append(float(check[stockChangePer]))
                else:
                    testTeam.append(-1)
                    TeamLose.append(float(check[stockChangePer]))

                
        except:
            continue

        indexer += 1
        
    ### Get rid of pesky NAs from when Lakers played but the stock did not yet exist
    TeamWin = [value for value in TeamWin if not math.isnan(value)]
    TeamLose = [value for value in TeamLose if not math.isnan(value)]
    NoGame = [value for value in NoGame if not math.isnan(value)]
    
    return([TeamWin, TeamLose, NoGame])
    
def DataBuilder1(teamOutcomes, stockChangeCat):
  
    TeamWin = []
    TeamLose = []
    NoGame = []

    indexer = 0
    testTeam = []

    for day in sports['Date']:
        date = datetime.datetime.strptime(day, '%d-%b-%y')
        date = date + datetime.timedelta(days = 1)
        date2 = date + datetime.timedelta(days = 2)
        date3 = date + datetime.timedelta(days = 3)
        
        try:
            row = date.strftime('%b %d, %Y')
            row1 = date2.strftime('%b %d, %Y')
            row2 = date3.strftime('%b %d, %Y')

            if len(stocks.loc[stocks['Date'] == row]) != 0:  
                check = stocks.loc[stocks['Date'] == row]
                
                if sports[teamOutcomes][indexer] == 0:
                    testTeam.append(0)
                    NoGame.append(list(check[stockChangeCat]))
                elif sports[teamOutcomes][indexer] == 1:
                    testTeam.append(1)
                    TeamWin.append(list(check[stockChangeCat]))
                else:
                    testTeam.append(-1)
                    TeamLose.append(list(check[stockChangeCat]))
                    
            elif len(stocks.loc[stocks['Date'] == row1]) != 0:
                check = stocks.loc[stocks['Date'] == row1]
                
                if sports[teamOutcomes][indexer] == 0:
                    testTeam.append(0)
                    NoGame.append(list(check[stockChangeCat]))
                elif sports[teamOutcomes][indexer] == 1:
                    testTeam.append(1)
                    TeamWin.append(list(check[stockChangeCat]))
                else:
                    testTeam.append(-1)
                    TeamLose.append(list(check[stockChangeCat]))

            elif len(stocks.loc[stocks['Date'] == row2]) != 0:
                check = stocks.loc[stocks['Date'] == row2]
                
                if sports[teamOutcomes][indexer] == 0:
                    testTeam.append(0)
                    NoGame.append(list(check[stockChangeCat]))
                elif sports[teamOutcomes][indexer] == 1:
                    testTeam.append(1)
                    TeamWin.append(list(check[stockChangeCat]))
                else:
                    testTeam.append(-1)
                    TeamLose.append(list(check[stockChangeCat]))

                
        except:
            continue

        indexer += 1
        
    ### Get rid of pesky NAs from when Lakers played but the stock did not yet exist
    #TeamWin = [value for value in TeamWin if not math.isnan(value)]
    #TeamLose = [value for value in TeamLose if not math.isnan(value)]
    #NoGame = [value for value in NoGame if not math.isnan(value)]
    
    return([TeamWin, TeamLose, NoGame])

    
### functions to make it easier to track different combinations
def GameCounter(stockChangeCat, gamesToTrack):
    ### creates lists of number of games on each day and the performance of a stock
    NFLGames = []
    MLBGames = []
    NBAGames = []
    TotalGames = []

    indexer = 0
    Outcomes = []

    for day in sports['Date']:
        date = datetime.datetime.strptime(day, '%d-%b-%y')
        date = date + datetime.timedelta(days = 1)
        date2 = date + datetime.timedelta(days = 2)
        date3 = date + datetime.timedelta(days = 3)
        
        row = date.strftime('%b %d, %Y')
        row1 = date2.strftime('%b %d, %Y')
        row2 = date3.strftime('%b %d, %Y')

        if len(stocks.loc[stocks['Date'] == row]) != 0:  
            check = stocks.loc[stocks['Date'] == row]
            NFLGames.append(sports['NFL Games'][indexer])
            MLBGames.append(sports['MLB Games'][indexer])
            NBAGames.append(sports['NBA Games'][indexer])
            TotalGames.append(sports['Total Games'][indexer])
            Outcomes.append(str(check[stockChangeCat]).split('\n')[0].split('    ')[1])


        elif len(stocks.loc[stocks['Date'] == row1]) != 0:
            check = stocks.loc[stocks['Date'] == row1]                  
            NFLGames.append(sports['NFL Games'][indexer])
            MLBGames.append(sports['MLB Games'][indexer])
            NBAGames.append(sports['NBA Games'][indexer])
            TotalGames.append(sports['Total Games'][indexer])
            Outcomes.append(str(check[stockChangeCat]).split('\n')[0].split('    ')[1])
        
        elif len(stocks.loc[stocks['Date'] == row2]) != 0:
            check = stocks.loc[stocks['Date'] == row2]
            NFLGames.append(sports['NFL Games'][indexer])
            MLBGames.append(sports['MLB Games'][indexer])
            NBAGames.append(sports['NBA Games'][indexer])
            TotalGames.append(sports['Total Games'][indexer])
            Outcomes.append(str(check[stockChangeCat]).split('\n')[0].split('    ')[1])
        
        else:
            continue

        indexer += 1

    ### Build a dataframe from the lists built above for analysis
    d = {'NFL Games':NFLGames, 'MLB Games':MLBGames, 'NBA Games':NBAGames, 'Total Games':TotalGames, 'Outcomes':Outcomes}
    testDF = pd.DataFrame(data = d)   
    ### Get rid of those pesky NAs from when the stock didn't exist yet
    testDF = testDF.dropna(thresh = 5)
    testDF = testDF.reset_index(drop = True)
    
    ### Make categorical buckets for the ANOVA test
    binn = []
    
    q25, q50, q75 = np.nanpercentile([value for value in testDF[gamesToTrack] if value != 0], [25, 50, 75])

    for x in range(len(testDF[gamesToTrack])):
        num = testDF[gamesToTrack][x]
        if num == 0:
            binn.append('Bucket 0')
        elif num > 0 and num <= q25:
            binn.append('Bucket 1')
        elif num > q25 and num <= q50:
            binn.append('Bucket 2')
        elif num > q50 and num <= q75:
            binn.append('Bucket 3')
        else:
            binn.append('Bucket 4')
        
    testDF.insert(5, 'Games Bin', binn)
    
    return(testDF)

def OutcomeCounter(team):
    wins = len(sports[team][sports[team] == 1])
    losses = len(sports[team][sports[team] == -1])
    nones = len(sports[team][sports[team] == 0])
    
    return([wins, losses, nones])
    
##############################
## ONE FINAL ANALYSIS: #######
###### NETWORKS ##############
##############################

print("Starting the final analysis...")

# Define function to strip once re.findall function finds data
def restrip(x):
    y = str(x).strip("['").strip("']")
    return y


# This finds the NFL Champions
nfl_df = pd.DataFrame(columns = ['Champion', 'Date', 'Combo'
                                 , 'Stock', 'Stock Date', 'Stock Percent Change', 'Stock Change Category'])
count = 0
for stockname in ['NASDAQ']:
    for i in range(1975,2018):
        index = sports['Date'][sports['Date'] == '1-Jan-'+ str(i)[2:]].index[0]
        dct = {}
        for j in range(index, index+61):
            if sports.loc[j, 'NFL Games'] == 1:
                for k in sports.columns[2:66]:
                    if re.search('Outcomes', k):
                        if sports.loc[j, k] != 0 and sports.loc[j, k] != -1:
                            #print(j, sports.loc[j, 'Date'], k, sports.loc[j, k])
                            dct[j] = list((sports.loc[j, 'Date'], k, sports.loc[j,k]))
        if dct != {}:
            vals = list(dct.values())
            sb = vals[len(vals)-1]
            #print(sb)
            date = sb[0]
            newdate = datetime.datetime.strptime(date, '%d-%b-%y').strftime('%d-%b-%y')
            dayafter = (datetime.datetime.strptime(newdate, '%d-%b-%y') + datetime.timedelta(days=1)).strftime('%d-%b-%y')
            dayafter_stocks = datetime.datetime.strptime(dayafter, '%d-%b-%y').strftime('%b %d, %Y')
            #Since Super Bowl always on Sunday's, the day after (Monday) should have stock data
            index_stock = stocks['Date'][stocks['Date'] == dayafter_stocks].index[0]
            if pd.isnull(stocks.loc[index_stock, stockname + ' Percent Change']) == False:
                nfl_df.loc[count, 'Stock Change Category'] = stocks.loc[index_stock, stockname + ' Change Category']
                nfl_df.loc[count, 'Champion'] = restrip(re.findall('(.*) Outcomes', sb[1]))
                nfl_df.loc[count, 'Date'] = datetime.datetime.strptime(newdate, '%d-%b-%y').strftime('%b %d, %Y')
                nfl_df.loc[count, 'Combo'] = nfl_df.loc[count,'Champion']+','+nfl_df.loc[count, 'Stock Change Category']
                nfl_df.loc[count, 'Stock'] = stockname
                nfl_df.loc[count, 'Stock Date'] = dayafter_stocks
                nfl_df.loc[count, 'Stock Percent Change'] = stocks.loc[index_stock, stockname + ' Percent Change']
                count += 1


# Network Analysis for Super Bowl champions

with open('nx_data.txt', 'w') as f:
    f.write('')

for l in range(len(nfl_df)):
    with open('nx_data.txt', 'a') as f:
        f.write(str(nfl_df.loc[l, 'Combo']) + ',' + str(nfl_df.loc[l, 'Stock Percent Change']) + '\n')
        
file=open("nx_data.txt", "rb")
nflnx=nx.read_edgelist(file, delimiter=",",create_using=nx.Graph(), nodetype=str,data=[("weight", float)])
file.close()
print("NFL NetworkX is:" ,nflnx.edges(data=True), "\n\n\n\n")
edge_labels = dict( ((u, v), d["weight"]) for u, v, d in nflnx.edges(data=True) )


##### Initial Metrics #####

# Prints summary information about the graph
print('Summary information:\n')
print(nx.info(nflnx))

# Prints density of graph
print('\nDensity:\n')
den = nx.density(nflnx)
print(den)

# Prints diameter of graph
print('\nDiameter:\n')
dia = nx.diameter(nflnx)
print(dia)

# Prints the degree of each node
print("\nNode Degree:\n")
for v in nflnx:
    print('%s %s' % (v,nflnx.degree(v)))

# Compute and print other stats
nbr_nodes = nx.number_of_nodes(nflnx)
nbr_edges = nx.number_of_edges(nflnx)
nbr_components = nx.number_connected_components(nflnx)

print("\nNumber of nodes:", nbr_nodes)
print("Number of edges:", nbr_edges)
print("Number of connected components:", nbr_components)

# Clustering coefficient
print("\nClustering coefficient:")
clust = nx.clustering(nflnx)
print(clust)

# Triangles
print("\nTriangles:")
tri = nx.triangles(nflnx)
print(tri)



# Compute betweenness and then store the value with each node in the networkx graph
betweenList = nx.betweenness_centrality(nflnx)
nx.set_node_attributes(nflnx, 'betweenness', betweenList)
print();
print("Betweenness of each node:\n")
print(betweenList)

# Graph based on betweenness
size = float(len(set(betweenList.values())))
pos = nx.spring_layout(nflnx)
count = 0.
for com in set(betweenList.values()) :
     count += 1.
     list_nodes = [nodes for nodes in betweenList.keys()
                                 if betweenList[nodes] == com]
     nx.draw_networkx_nodes(nflnx, pos, list_nodes, node_size = 50,
                                node_color = str(count/size), labels=list_nodes)
nx.draw_networkx_edges(nflnx,pos, alpha=0.5, with_labels=True)
nx.draw_networkx_labels(nflnx,pos,font_size=6)
plt.show()


# Create plot that colors nodes based on degree

degreelist = nx.degree_centrality(nflnx)
nx.set_node_attributes(nflnx, 'degree', degreelist)
print();
print("Degree of each node:\n")
print(degreelist)

size = float(len(set(degreelist.values())))
pos = nx.spring_layout(nflnx)
count = 0
for com in set(degreelist.values()) :
     count += 1
     list_nodes = [nodes for nodes in degreelist.keys()
                                 if degreelist[nodes] == com]
     nx.draw_networkx_nodes(nflnx, pos, list_nodes, node_size = 50,
                                node_color = str(list(degreelist.values())[count]), cmap=plt.cm.viridis,labels=list_nodes)
nx.draw_networkx_edges(nflnx,pos, alpha=0.5, with_labels=True)
nx.draw_networkx_labels(nflnx,pos,font_size=6)
plt.show()



# NBA champions
nba_df = pd.DataFrame(columns = ['Champion', 'Date', 'Combo'
                                 , 'Stock', 'Stock Date', 'Stock Percent Change', 'Stock Change Category'])
count = 0
for stockname in ['NASDAQ']:
    for i in range(1975,2018):
        index = sports['Date'][sports['Date'] == '1-May-'+ str(i)[2:]].index[0]
        dct = {}
        for j in range(index, index+61):
            if sports.loc[j, 'NBA Games'] == 1:
                for k in sports.columns[-66:]:
                    if re.search('Outcomes', k):
                        if sports.loc[j, k] != 0 and sports.loc[j, k] != -1:
                            #print(j, sports.loc[j, 'Date'], k, sports.loc[j, k])
                            dct[j] = list((sports.loc[j, 'Date'], k, sports.loc[j,k]))
        vals = list(dct.values())
        final_game = vals[len(vals)-1]
        #print(final_game)
        date = final_game[0]
        newdate = datetime.datetime.strptime(date, '%d-%b-%y').strftime('%d-%b-%y')
        dayafter = (datetime.datetime.strptime(newdate, '%d-%b-%y') + datetime.timedelta(days=1)).strftime('%d-%b-%y')
        dayafter_stocks = datetime.datetime.strptime(dayafter, '%d-%b-%y').strftime('%b %d, %Y')
        dayafter2 = (datetime.datetime.strptime(newdate, '%d-%b-%y') + datetime.timedelta(days=2)).strftime('%d-%b-%y')
        dayafter_stocks2 = datetime.datetime.strptime(dayafter2, '%d-%b-%y').strftime('%b %d, %Y')
        dayafter3 = (datetime.datetime.strptime(newdate, '%d-%b-%y') + datetime.timedelta(days=3)).strftime('%d-%b-%y')
        dayafter_stocks3 = datetime.datetime.strptime(dayafter3, '%d-%b-%y').strftime('%b %d, %Y')
        dayafter4 = (datetime.datetime.strptime(newdate, '%d-%b-%y') + datetime.timedelta(days=4)).strftime('%d-%b-%y')
        dayafter_stocks4 = datetime.datetime.strptime(dayafter4, '%d-%b-%y').strftime('%b %d, %Y')
        # If first day after game day in stocks data frame (i.e. Sundays)
        if dayafter_stocks in stocks['Date'].values:
            index_stock = stocks['Date'][stocks['Date'] == dayafter_stocks].index[0]
            if pd.isnull(stocks.loc[index_stock, stockname + ' Percent Change']) == False:
                nba_df.loc[count, 'Stock Change Category'] = stocks.loc[index_stock, stockname + ' Change Category']
                nba_df.loc[count, 'Champion'] = restrip(re.findall('(.*) Outcomes', final_game[1]))
                nba_df.loc[count, 'Date'] = datetime.datetime.strptime(newdate, '%d-%b-%y').strftime('%b %d, %Y')
                nba_df.loc[count, 'Combo'] = nba_df.loc[count, 'Champion']+','+nba_df.loc[count, 'Stock Change Category']
                nba_df.loc[count, 'Stock'] = stockname
                nba_df.loc[count, 'Stock Date'] = dayafter_stocks
                nba_df.loc[count, 'Stock Percent Change'] = stocks.loc[index_stock, stockname + ' Percent Change']
                count += 1
        # If second day after game day in stocks data frame (i.e. Saturdays)
        elif dayafter_stocks2 in stocks['Date'].values:
            index_stock2 = stocks['Date'][stocks['Date'] == dayafter_stocks2].index[0]
            if pd.isnull(stocks.loc[index_stock2, stockname + ' Percent Change']) == False:
                nba_df.loc[count, 'Stock Change Category'] = stocks.loc[index_stock, stockname + ' Change Category']
                nba_df.loc[count, 'Champion'] = restrip(re.findall('(.*) Outcomes', final_game[1]))
                nba_df.loc[count, 'Date'] = datetime.datetime.strptime(newdate, '%d-%b-%y').strftime('%b %d, %Y')
                nba_df.loc[count, 'Combo'] = nba_df.loc[count, 'Champion']+','+nba_df.loc[count, 'Stock Change Category']
                nba_df.loc[count, 'Stock'] = stockname
                nba_df.loc[count, 'Stock Date'] = dayafter_stocks
                nba_df.loc[count, 'Stock Percent Change'] = stocks.loc[index_stock, stockname + ' Percent Change']
                count += 1
        # If third day after game day in stocks data frame (i.e. Fridays)
        elif dayafter_stocks3 in stocks['Date'].values:
            index_stock3 = stocks['Date'][stocks['Date'] == dayafter_stocks3].index[0]
            if pd.isnull(stocks.loc[index_stock3, stockname + ' Percent Change']) == False:
                nba_df.loc[count, 'Stock Change Category'] = stocks.loc[index_stock, stockname + ' Change Category']
                nba_df.loc[count, 'Champion'] = restrip(re.findall('(.*) Outcomes', final_game[1]))
                nba_df.loc[count, 'Date'] = datetime.datetime.strptime(newdate, '%d-%b-%y').strftime('%b %d, %Y')
                nba_df.loc[count, 'Combo'] = nba_df.loc[count, 'Champion']+','+nba_df.loc[count, 'Stock Change Category']
                nba_df.loc[count, 'Stock'] = stockname
                nba_df.loc[count, 'Stock Date'] = dayafter_stocks
                nba_df.loc[count, 'Stock Percent Change'] = stocks.loc[index_stock, stockname + ' Percent Change']
                count += 1
        # If fourth day after game day in stocks data frame (i.e. Holiday)
        elif dayafter_stocks4 in stocks['Date'].values:
            index_stock4 = stocks['Date'][stocks['Date'] == dayafter_stocks4].index[0]
            if pd.isnull(stocks.loc[index_stock4, stockname + ' Percent Change']) == False:
                nba_df.loc[count, 'Stock Change Category'] = stocks.loc[index_stock, stockname + ' Change Category']
                nba_df.loc[count, 'Champion'] = restrip(re.findall('(.*) Outcomes', final_game[1]))
                nba_df.loc[count, 'Date'] = datetime.datetime.strptime(newdate, '%d-%b-%y').strftime('%b %d, %Y')
                nba_df.loc[count, 'Combo'] = nba_df.loc[count, 'Champion']+','+nba_df.loc[count, 'Stock Change Category']
                nba_df.loc[count, 'Stock'] = stockname
                nba_df.loc[count, 'Stock Date'] = dayafter_stocks
                nba_df.loc[count, 'Stock Percent Change'] = stocks.loc[index_stock, stockname + ' Percent Change']
                count += 1



# Network Analysis for NBA champions

with open('nx_data2.txt', 'w') as f:
    f.write('')

for l in range(len(nba_df)):
    with open('nx_data2.txt', 'a') as f:
        f.write(str(nba_df.loc[l, 'Combo']) + ',' + str(nba_df.loc[l, 'Stock Percent Change']) + '\n')
        
file2=open("nx_data2.txt", "rb")
nbanx=nx.read_edgelist(file2, delimiter=",",create_using=nx.Graph(), nodetype=str,data=[("weight", float)])
file2.close()
print("NBA NetworkX is:" ,nbanx.edges(data=True), "\n\n\n\n")
edge_labels = dict( ((u, v), d["weight"]) for u, v, d in nbanx.edges(data=True) )



##### Initial Metrics #####

# Prints summary information about the graph
print('Summary information:\n')
print(nx.info(nbanx))

# Prints density of graph
print('\nDensity:\n')
den = nx.density(nbanx)
print(den)

# Prints diameter of graph
print('\nDiameter:\n')
dia = nx.diameter(nbanx)
print(dia)

# Prints the degree of each node
print("\nNode Degree:\n")
for v in nbanx:
    print('%s %s' % (v,nbanx.degree(v)))

# Compute and print other stats
nbr_nodes = nx.number_of_nodes(nbanx)
nbr_edges = nx.number_of_edges(nbanx)
nbr_components = nx.number_connected_components(nbanx)

print("\nNumber of nodes:", nbr_nodes)
print("Number of edges:", nbr_edges)
print("Number of connected components:", nbr_components)

# Clustering coefficient
print("\nClustering coefficient:")
clust = nx.clustering(nbanx)
print(clust)

# Triangles
print("\nTriangles:")
tri = nx.triangles(nbanx)
print(tri)



# Compute betweenness and then store the value with each node in the networkx graph
betweenList = nx.betweenness_centrality(nbanx)
nx.set_node_attributes(nbanx, 'betweenness', betweenList)
print();
print("Betweenness of each node:\n")
print(betweenList)

# Graph based on betweenness
size = float(len(set(betweenList.values())))
pos = nx.spring_layout(nbanx)
count = 0.
for com in set(betweenList.values()) :
     count += 1.
     list_nodes = [nodes for nodes in betweenList.keys()
                                 if betweenList[nodes] == com]
     nx.draw_networkx_nodes(nbanx, pos, list_nodes, node_size = 50,
                                node_color = str(count/size), labels=list_nodes)
nx.draw_networkx_edges(nbanx,pos, alpha=0.5, with_labels=True)
nx.draw_networkx_labels(nbanx,pos,font_size=6)
plt.show()


# Create plot that colors nodes based on degree

degreelist = nx.degree_centrality(nbanx)
nx.set_node_attributes(nbanx, 'degree', degreelist)
print();
print("Degree of each node:\n")
print(degreelist)

size = float(len(set(degreelist.values())))
pos = nx.spring_layout(nbanx)
count = 0.
for com in set(degreelist.values()) :
     count += 1.
     list_nodes = [nodes for nodes in degreelist.keys()
                                 if degreelist[nodes] == com]
     nx.draw_networkx_nodes(nbanx, pos, list_nodes, node_size = 50,
                                node_color = str(count/size), labels=list_nodes)
nx.draw_networkx_edges(nbanx,pos, alpha=0.5, with_labels=True)
nx.draw_networkx_labels(nbanx,pos,font_size=6)
plt.show()
    

#####################################
########## VIS 1 ####################
#####################################

####
# Box and whisker plot examining the relationship between Lakers games and
# the market
####

test = DataBuilder('Los Angeles Lakers Outcomes', 'CHEVRON Percent Change')
test1 = DataBuilder('Los Angeles Lakers Outcomes', 'NASDAQ Percent Change')
test2 = DataBuilder('Los Angeles Lakers Outcomes', 'BONDS Percent Change')
test3 = DataBuilder('Los Angeles Lakers Outcomes', 'FRANKLIN GOLD AND_PRECIOUS METALS Percent Change')
test4 = DataBuilder('Los Angeles Lakers Outcomes', 'STURM & RUGER (GUNS) Percent Change')

trace0 = go.Box(x0 = 'CHEVRON (+Win)', y = [x * 100 for x in test[0]], boxmean = True, name = 'Lakers Win', marker = dict(color = 'red'))
trace1 = go.Box(x0 = 'CHEVRON (+Lose)', y = [x * 100 for x in test[1]], boxmean = True, name = 'Lakers Lose', marker = dict(color = 'blue'))
trace2 = go.Box(x0 = 'CHEVRON (+No Game)', y = [x * 100 for x in test[2]], boxmean = True, name = "Lakers Don't Play", marker = dict(color = 'green'))

trace3 = go.Box(x0 = 'NASDAQ (+Win)', y = [x * 100 for x in test1[0]], boxmean = True, name = 'Lakers Win', showlegend = False, marker = dict(color = 'red'))
trace4 = go.Box(x0 = 'NASDAQ (+Lose)', y = [x * 100 for x in test1[1]], boxmean = True, name = 'Lakers Lose', showlegend = False, marker = dict(color = 'blue'))
trace5 = go.Box(x0 = 'NASDAQ (+No Game)', y = [x * 100 for x in test1[2]], boxmean = True, name = "Lakers Don't Play", showlegend = False, marker = dict(color = 'green'))

trace6 = go.Box(x0 = 'BONDS (+Win)', y = [x * 100 for x in test2[0]], boxmean = True, name = 'Lakers Win', showlegend = False, marker = dict(color = 'red'))
trace7 = go.Box(x0 = 'BONDS (+Lose)', y = [x * 100 for x in test2[1]], boxmean = True, name = 'Lakers Lose', showlegend = False, marker = dict(color = 'blue'))
trace8 = go.Box(x0 = 'BONDS (+No Game)', y = [x * 100 for x in test2[2]], boxmean = True, name = "Lakers Don't Play", showlegend = False, marker = dict(color = 'green'))

trace9 = go.Box(x0 = 'FRANKLIN GPM (+Win)', y = [x * 100 for x in test3[0]], boxmean = True, name = 'Lakers Win', showlegend = False, marker = dict(color = 'red'))
trace10 = go.Box(x0 = 'FRANKLIN GPM (+Lose)', y = [x * 100 for x in test3[1]], boxmean = True, name = 'Lakers Lose', showlegend = False, marker = dict(color = 'blue'))
trace11 = go.Box(x0 = 'FRANKLIN GPM (+No Game)', y = [x * 100 for x in test3[2]], boxmean = True, name = "Lakers Don't Play", showlegend = False, marker = dict(color = 'green'))

trace12 = go.Box(x0 = 'STURM & RUGER (+Win)', y = [x * 100 for x in test4[0]], boxmean = True, name = 'Lakers Win', showlegend = False, marker = dict(color = 'red'))
trace13 = go.Box(x0 = 'STURM & RUGER (+Lose)', y = [x * 100 for x in test4[1]], boxmean = True, name = 'Lakers Lose', showlegend = False, marker = dict(color = 'blue'))
trace14 = go.Box(x0 = 'STURM & RUGER (+No Game)', y = [x * 100 for x in test4[2]], boxmean = True, name = "Lakers Don't Play", showlegend = False, marker = dict(color = 'green'))

data = [trace0, trace1, trace2, trace3, trace4, trace5,
        trace6, trace7, trace8, trace9, trace10, trace11,
        trace12, trace13, trace14]

updatemenus = list([dict(active = -1, buttons = list([
        dict(label = 'ALL',
             method = 'update',
             args = [{'visible': [True, True, True,True, True, True,True, True, True,True, True, True,True, True, True]},
                     {'title':'ALL'}]),
        dict(label = 'CHEVRON',
             method = 'update',
             args = [{'visible': [True, True, True, False, False,False, False,False, False,False, False,False, False,False, False]},
                     {'title':'CHEVRON'}]),
        dict(label = 'NASDAQ',
             method = 'update',
             args = [{'visible': [False, False, False, True, True, True, False, False, False, False, False, False, False, False, False]},
                     {'title':'NASDAQ'}]),
        dict(label = 'BONDS',
             method = 'update',
             args = [{'visible': [False, False, False, False, False, False, True, True, True, False, False, False, False, False, False]},
                     {'title':'BONDS'}]),
        dict(label = 'FRANKLIN GPM',
             method = 'update',
             args = [{'visible': [False, False, False, False, False, False, False, False, False, True, True, True, False, False, False]},
                     {'title':'FRANKLIN GPM'}]),
        dict(label = 'STRUM & RUGER',
             method = 'update',
             args = [{'visible': [False, False, False, False, False, False, False, False, False, False, False, False, True, True, True]},
                     {'title':'STURM & RUGER'}]),
                  ])
                  )]
                  )

myLayout = go.Layout(
        title = "The Market After Lakers Wins and Losses",
        xaxis=dict(
                title = 'Stocks'
	),
	yaxis=dict(
		title = 'Percentage Change in Market', range = [-5, 5]
	),
    updatemenus = updatemenus
    )
    

myFigure = go.Figure(data=data, layout=myLayout)


ply.plot(myFigure)

#################################
######### VIS 2 #################
#################################

####
# Simple scatter plot looking at the Yankees performance and 
# the market
####

test5 = DataBuilder('New York Yankees Outcomes', 'S&P Percent Change')
test6 = DataBuilder('New York Yankees Outcomes', 'ANHEUSER-BUSCH Percent Change')
test7 = DataBuilder('New York Yankees Outcomes', 'VICEX (SIN) Percent Change')
test8 = DataBuilder('New York Yankees Outcomes', 'EXXON Percent Change')
test9 = DataBuilder('New York Yankees Outcomes', 'FIRST EAGLE GOLD Percent Change')
test10 = DataBuilder('New York Yankees Outcomes', 'BONDS Percent Change')

wins = []
losses = []
nogames = []

for group in [test5, test6, test7, test8, test9, test10]:
    wins.append(np.mean(group[0]))
    losses.append(np.mean(group[1]))
    nogames.append(np.mean(group[2]))
    
trace15 = {'x': [x * 100 for x in wins], 'y': ['S&P', 'ANHEUSER-BUSCH',
                            'VICEX (SIN)', 'EXXON',
                            'FIRST EAGLE', 'BONDS'],
            'marker': {'color': 'green', 'size': 15},
            'mode': 'markers',
            'name': 'AFTER WINS',
            'type': 'scatter'}

trace16 = {'x': [x * 100 for x in losses], 'y': ['S&P', 'ANHEUSER-BUSCH',
                            'VICEX (SIN)', 'EXXON',
                            'FIRST EAGLE', 'BONDS'],
            'marker': {'color': 'red', 'size': 15},
            'mode': 'markers',
            'name': 'AFTER LOSSES',
            'type': 'scatter'}

trace17 = {'x': [x * 100 for x in nogames], 'y': ['S&P', 'ANHEUSER-BUSCH',
                            'VICEX (SIN)', 'EXXON',
                            'FIRST EAGLE', 'BONDS'],
            'marker': {'color': 'blue', 'size': 15},
            'mode': 'markers',
            'name': 'AFTER NO GAME',
            'type': 'scatter'}

data = go.Data([trace15, trace16, trace17])

layout = {'title': 'Market Performance After Yankee Games',
          'xaxis': {'title': 'Mean Percentage Change in Stock', 'range': [-.03, .1],},
          'yaxis': {'title': 'Stock',}
          }

fig = go.Figure(data = data, layout = layout)

ply.plot(fig)

##################################
######## VIS 3 ###################
##################################

####
# Stacked bar chart examining the # of games and the NASDAQ
####

helper = GameCounter('NASDAQ Change Category', 'Total Games')
helper = helper.iloc[1:, :]

bar0 = list(helper.loc[helper['Games Bin'] == 'Bucket 0']['Outcomes'])
bar1 = list(helper.loc[helper['Games Bin'] == 'Bucket 1']['Outcomes'])
bar2 = list(helper.loc[helper['Games Bin'] == 'Bucket 2']['Outcomes'])
bar3 = list(helper.loc[helper['Games Bin'] == 'Bucket 3']['Outcomes'])
bar4 = list(helper.loc[helper['Games Bin'] == 'Bucket 4']['Outcomes'])

trace18 = go.Bar(
        x = ['No Games', 'Few Games', 'Average # Games', 'A lot of Games', 'Most Games'],
        y = [len([x for x in bar0 if x == 'BIG DIP']), len([x for x in bar1 if x == 'BIG DIP']),
             len([x for x in bar2 if x == 'BIG DIP']), len([x for x in bar3 if x == 'BIG DIP']),
             len([x for x in bar4 if x == 'BIG DIP'])],
        name = 'BIG DIP',
        text = [str(round(100 * (len([x for x in bar0 if x == 'BIG DIP'])/len(bar0)), 1)) + '%',
                str(round(100 * (len([x for x in bar1 if x == 'BIG DIP'])/len(bar1)), 1)) + '%',
                str(round(100 * (len([x for x in bar2 if x == 'BIG DIP'])/len(bar2)), 1)) + '%',
                str(round(100 * (len([x for x in bar3 if x == 'BIG DIP'])/len(bar3)), 1)) + '%',
                str(round(100 * (len([x for x in bar4 if x == 'BIG DIP'])/len(bar4)), 1)) + '%']
        )

trace19 = go.Bar(
        x = ['No Games', 'Few Games', 'Average # Games', 'A lot of Games', 'Most Games'],
        y = [len([x for x in bar0 if x == 'DIP']), len([x for x in bar1 if x == 'DIP']),
             len([x for x in bar2 if x == 'DIP']), len([x for x in bar3 if x == 'DIP']),
             len([x for x in bar4 if x == 'DIP'])],
        name = 'DIP',
        text = [str(round(100 * (len([x for x in bar0 if x == 'DIP'])/len(bar0)), 1)) + '%',
                str(round(100 * (len([x for x in bar1 if x == 'DIP'])/len(bar1)), 1)) + '%',
                str(round(100 * (len([x for x in bar2 if x == 'DIP'])/len(bar2)), 1)) + '%',
                str(round(100 * (len([x for x in bar3 if x == 'DIP'])/len(bar3)), 1)) + '%',
                str(round(100 * (len([x for x in bar4 if x == 'DIP'])/len(bar4)), 1)) + '%']
        )

trace20 = go.Bar(
        x = ['No Games', 'Few Games', 'Average # Games', 'A lot of Games', 'Most Games'],
        y = [len([x for x in bar0 if x == 'LITTLE MOVEMENT']), len([x for x in bar1 if x == 'LITTLE MOVEMENT']),
             len([x for x in bar2 if x == 'LITTLE MOVEMENT']), len([x for x in bar3 if x == 'LITTLE MOVEMENT']),
             len([x for x in bar4 if x == 'LITTLE MOVEMENT'])],
        name = 'LITTLE MOVEMENT',
        text = [str(round(100 * (len([x for x in bar0 if x == 'LITTLE MOVEMENT'])/len(bar0)), 1)) + '%',
                str(round(100 * (len([x for x in bar1 if x == 'LITTLE MOVEMENT'])/len(bar1)), 1)) + '%',
                str(round(100 * (len([x for x in bar2 if x == 'LITTLE MOVEMENT'])/len(bar2)), 1)) + '%',
                str(round(100 * (len([x for x in bar3 if x == 'LITTLE MOVEMENT'])/len(bar3)), 1)) + '%',
                str(round(100 * (len([x for x in bar4 if x == 'LITTLE MOVEMENT'])/len(bar4)), 1)) + '%']
        )

trace21 = go.Bar(
        x = ['No Games', 'Few Games', 'Average # Games', 'A lot of Games', 'Most Games'],
        y = [len([x for x in bar0 if x == 'JUMP']), len([x for x in bar1 if x == 'JUMP']),
             len([x for x in bar2 if x == 'JUMP']), len([x for x in bar3 if x == 'JUMP']),
             len([x for x in bar4 if x == 'JUMP'])],
        name = 'JUMP',
        text = [str(round(100 * (len([x for x in bar0 if x == 'JUMP'])/len(bar0)), 1)) + '%',
                str(round(100 * (len([x for x in bar1 if x == 'JUMP'])/len(bar1)), 1)) + '%',
                str(round(100 * (len([x for x in bar2 if x == 'JUMP'])/len(bar2)), 1)) + '%',
                str(round(100 * (len([x for x in bar3 if x == 'JUMP'])/len(bar3)), 1)) + '%',
                str(round(100 * (len([x for x in bar4 if x == 'JUMP'])/len(bar4)), 1)) + '%']
        )

trace22 = go.Bar(
        x = ['No Games', 'Few Games', 'Average # Games', 'A lot of Games', 'Most Games'],
        y = [len([x for x in bar0 if x == 'BIG JUMP']), len([x for x in bar1 if x == 'BIG JUMP']),
             len([x for x in bar2 if x == 'BIG JUMP']), len([x for x in bar3 if x == 'BIG JUMP']),
             len([x for x in bar4 if x == 'BIG JUMP'])],
        name = 'BIG JUMP',
        text = [str(round(100 * (len([x for x in bar0 if x == 'BIG JUMP'])/len(bar0)), 1)) + '%',
                str(round(100 * (len([x for x in bar1 if x == 'BIG JUMP'])/len(bar1)), 1)) + '%',
                str(round(100 * (len([x for x in bar2 if x == 'BIG JUMP'])/len(bar2)), 1)) + '%',
                str(round(100 * (len([x for x in bar3 if x == 'BIG JUMP'])/len(bar3)), 1)) + '%',
                str(round(100 * (len([x for x in bar4 if x == 'BIG JUMP'])/len(bar4)), 1)) + '%']
        )

data = [trace18, trace19, trace20, trace21, trace22]
layout = go.Layout(barmode = 'stack',
                   title = 'NASDAQ Movement After Sports Games')

fig = go.Figure(data = data, layout = layout)

ply.plot(fig)

##########################
###### VIS 4 #############
##########################

####
# Customizable trend lines for the total-market indexes
####

df1 = web.DataReader('^IXIC', 'yahoo', datetime.datetime(1975, 1, 1), datetime.datetime(2017, 10, 1))
df2 = web.DataReader('^GSPC', 'yahoo', datetime.datetime(1975, 1, 1), datetime.datetime(2017, 10, 1))
df = web.DataReader('^DJI', 'yahoo', datetime.datetime(1985, 1, 1), datetime.datetime(2017, 10, 1))

trace23 = go.Scatter(x = df1.index, y = df1.Close, marker = {'color': 'red'}, name = 'NASDAQ')

trace24 = go.Scatter(x = df2.index, y = df2.Close, marker = {'color': 'green'}, name = 'S&P')

trace = go.Scatter(x = df.index, y = df.Close, marker = {'color': 'blue'}, name = 'DOW JONES')


data = [trace23, trace24, trace]

layout = dict(
    title='Index Growth Since 1975',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1 year',
                     step='year',
                     stepmode='backward'),
                dict(count=10,
                     label='10 years',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    ),
    yaxis = dict(title = 'Daily Closing Value')
)


fig = go.Figure(data = data, layout = layout)

ply.plot(fig)

####
# Customizable trend lines for the other, smaller stocks
####

df3 = web.DataReader('MRO', 'yahoo', datetime.datetime(1990, 1, 1), datetime.datetime(2017, 10, 1))
df4 = web.DataReader('CVX', 'yahoo', datetime.datetime(1975, 1, 1), datetime.datetime(2017, 10, 1))
df5 = web.DataReader('XOM', 'yahoo', datetime.datetime(1975, 1, 1), datetime.datetime(2017, 10, 1))
df6 = web.DataReader('FKRCX', 'yahoo', datetime.datetime(1985, 1, 1), datetime.datetime(2017, 10, 1))
df7 = web.DataReader('SGGDX', 'yahoo', datetime.datetime(1995, 1, 1), datetime.datetime(2017, 10, 1))
df8 = web.DataReader('RGR', 'yahoo', datetime.datetime(1985, 1, 1), datetime.datetime(2017, 10, 1))
df9 = web.DataReader('AOBC', 'yahoo', datetime.datetime(2000, 1, 1), datetime.datetime(2017, 10, 1))
df10 = web.DataReader('BUD', 'yahoo', datetime.datetime(2009, 1, 1), datetime.datetime(2017, 10, 1))
df11 = web.DataReader('VICEX', 'yahoo', datetime.datetime(2002, 1, 1), datetime.datetime(2017, 10, 1))
df12 = web.DataReader('RYCVX', 'yahoo', datetime.datetime(2004, 1, 1), datetime.datetime(2017, 10, 1))
df13 = web.DataReader('FKUTX', 'yahoo', datetime.datetime(1985, 1, 1), datetime.datetime(2017, 10, 1))
df14 = web.DataReader('VHT', 'yahoo', datetime.datetime(2004, 1, 1), datetime.datetime(2017, 10, 1))
df15 = web.DataReader('VGT', 'yahoo', datetime.datetime(2004, 1, 1), datetime.datetime(2017, 10, 1))
df16 = web.DataReader('FBNDX', 'yahoo', datetime.datetime(1985, 1, 1), datetime.datetime(2017, 10, 1))

trace25 = go.Scatter(x = df3.index, y = df3.Close, marker = {'color': 'pink'}, name = 'MARATHON OIL')
trace26 = go.Scatter(x = df4.index, y = df4.Close, marker = {'color': 'orange'}, name = 'CHEVRON OIL')
trace27 = go.Scatter(x = df5.index, y = df5.Close, marker = {'color': 'blue'}, name = 'EXXON OIL')
trace28 = go.Scatter(x = df6.index, y = df6.Close, marker = {'color': 'red'}, name = 'FRANKLIN GPM')
trace29 = go.Scatter(x = df7.index, y = df7.Close, marker = {'color': 'green'}, name = 'FIRST EAGLE')
trace30 = go.Scatter(x = df8.index, y = df8.Close, marker = {'color': 'purple'}, name = 'STURM & RUGER')
trace31 = go.Scatter(x = df9.index, y = df9.Close, marker = {'color': 'red'}, name = 'AMERICAN OUTDOOR')
trace32 = go.Scatter(x = df10.index, y = df10.Close, marker = {'color': 'black'}, name = 'ANHEUSER-BUSCH')
trace33 = go.Scatter(x = df11.index, y = df11.Close, marker = {'color': 'brown'}, name = 'VICEX')
trace34 = go.Scatter(x = df12.index, y = df12.Close, marker = {'color': 'gray'}, name = 'RYDEX')
trace35 = go.Scatter(x = df13.index, y = df13.Close, marker = {'color': 'brown'}, name = 'UTILITIES')
trace36 = go.Scatter(x = df14.index, y = df14.Close, marker = {'color': 'olive'}, name = 'HEALTHCARE')
trace37 = go.Scatter(x = df15.index, y = df15.Close, marker = {'color': 'skyblue'}, name = 'IT')
trace38 = go.Scatter(x = df16.index, y = df16.Close, marker = {'color': 'darkgreen'}, name = 'BONDS')

data = [trace25, trace26, trace27, trace28, trace29, trace30, trace31, trace32,
        trace33, trace34, trace35, trace36, trace37, trace38]

layout = dict(
    title='Single Fund Growth Since 1975 or Inception',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1 year',
                     step='year',
                     stepmode='backward'),
                dict(count=10,
                     label='10 years',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    ),
    yaxis = dict(title = 'Daily Closing Value')
)

fig = go.Figure(data = data, layout = layout)

ply.plot(fig)

#####################
#### VIS 5 ##########
#####################

####
# Plots of teams' performance, separated by league
####

outcomeCols = []
for column in sports.columns:
    if 'Outcomes' in column:
        outcomeCols.append(column)
        
outcomeCols = outcomeCols[1:95]
        
allTeams = []
for column in outcomeCols:
    allTeams.append(column[:-9])
    
NFLTeams = allTeams[0:32]
MLBTeams = allTeams[32:64]
NBATeams = allTeams[64:]

w_l = []
wins = []
losses = []
for column in outcomeCols:
    holder = OutcomeCounter(column)
    wins.append(holder[0])
    losses.append(holder[1])
    if holder[1] == 0:
        w_l.append(1)
    else:
        w_l.append(holder[0]/(holder[0] + holder[1]))
        
w_l = [round(x, 4) for x in w_l]

texts = []

mover = 0
for value in wins:
    texts.append(allTeams[mover] + '    ' + 'W/L%: ' + str(w_l[mover]) + '    ' + 'Wins: ' + str(value) + '    ' + 'Losses: ' + str(losses[mover]))
    mover += 1
        
NFLW_L = w_l[0:32]
NFLt = texts[0:32]


MLBW_L = w_l[32:64]
MLBt = texts[32:64]


NBAW_L = w_l[64:]
NBAt = texts[64:]


### NFL ###
    
trace39 = go.Scatter(x = NFLTeams, y = NFLW_L, mode = 'markers', text = NFLt,
                     marker = dict(size = '15',
                                   color = NFLW_L,
                                   colorscale = 'Portland',
                                   showscale = True))

layout = dict(title = 'NFL Teams W/L Stats',
              xaxis = dict(title = 'Team'),
              yaxis = dict(title = 'Win %'))

data = [trace39]

fig = go.Figure(data = data, layout = layout)

ply.plot(fig) 
 
### MLB ###

trace40 = go.Scatter(x = MLBTeams, y = MLBW_L, mode = 'markers', text = MLBt,
                     marker = dict(size = '15',
                                   color = MLBW_L,
                                   colorscale = 'Viridis',
                                   showscale = True))

layout = dict(title = 'MLB Teams W/L Stats',
              xaxis = dict(title = 'Team'),
              yaxis = dict(title = 'Win %'))

data = [trace40]

fig = go.Figure(data = data, layout = layout)

ply.plot(fig)

### NBA ###

trace41 = go.Scatter(x = NBATeams, y = NBAW_L, mode = 'markers', text = NBAt,
                     marker = dict(size = '15',
                                   color = NBAW_L,
                                   colorscale = 'Rainbow',
                                   showscale = True))

layout = dict(title = 'NBA Teams W/L Stats',
              xaxis = dict(title = 'Team'),
              yaxis = dict(title = 'Win %'))

data = [trace41]

fig = go.Figure(data = data, layout = layout)

ply.plot(fig)


#####################
###### VIS 6 ########
#####################

####
# Pie charts describing the Bond Index after Yankees games
####

result = DataBuilder1('New York Yankees Outcomes', 'BONDS Change Category')

bigJump1 = len([x for x in result[0] if str(x).strip('[').strip(']').strip("'") == 'BIG JUMP'])
jump1 = len([x for x in result[0] if str(x).strip('[').strip(']').strip("'") == 'JUMP'])
lm1 = len([x for x in result[0] if str(x).strip('[').strip(']').strip("'") == 'LITTLE MOVEMENT'])
dip1 = len([x for x in result[0] if str(x).strip('[').strip(']').strip("'") == 'DIP'])
bigDip1 = len([x for x in result[0] if str(x).strip('[').strip(']').strip("'") == 'BIG DIP'])

bigJump2 = len([x for x in result[1] if str(x).strip('[').strip(']').strip("'") == 'BIG JUMP'])
jump2 = len([x for x in result[1] if str(x).strip('[').strip(']').strip("'") == 'JUMP'])
lm2 = len([x for x in result[1] if str(x).strip('[').strip(']').strip("'") == 'LITTLE MOVEMENT'])
dip2 = len([x for x in result[1] if str(x).strip('[').strip(']').strip("'") == 'DIP'])
bigDip2 = len([x for x in result[1] if str(x).strip('[').strip(']').strip("'") == 'BIG DIP'])

bigJump3 = len([x for x in result[2] if str(x).strip('[').strip(']').strip("'") == 'BIG JUMP'])
jump3 = len([x for x in result[2] if str(x).strip('[').strip(']').strip("'") == 'JUMP'])
lm3 = len([x for x in result[2] if str(x).strip('[').strip(']').strip("'") == 'LITTLE MOVEMENT'])
dip3 = len([x for x in result[2] if str(x).strip('[').strip(']').strip("'") == 'DIP'])
bigDip3 = len([x for x in result[2] if str(x).strip('[').strip(']').strip("'") == 'BIG DIP'])

fig = dict(data = [dict(values = [bigJump1, jump1, lm1, dip1, bigDip1],
                        labels = ['Big Jump', 'Jump', 'Little Movement', 'Dip', 'Big Dip'],
                        domain = dict(x = [0, .3]),
                        name = 'Bonds After a Yankees Win',
                        hoverinfo = 'label'+'percent'+'name',
                        hole = .5,
                        text = 'Win',
                        textposition = 'inside',
                        type = 'pie'),
                    dict(values = [bigJump2, jump2, lm2, dip2, bigDip2],
                        labels = ['Big Jump', 'Jump', 'Little Movement', 'Dip', 'Big Dip'],
                        domain = dict(x = [.35, .65]),
                        name = 'Bonds After a Yankees Loss',
                        hoverinfo = 'label'+'percent'+'name',
                        hole = .5,
                        text = 'Loss',
                        textposition = 'inside',
                        type = 'pie'),
                    dict(values = [bigJump3, jump3, lm3, dip3, bigDip3],
                        labels = ['Big Jump', 'Jump', 'Little Movement', 'Dip', 'Big Dip'],
                        domain = dict(x = [.7, 1]),
                        name = 'Bonds After the Yankees Dont Play',
                        hoverinfo = 'label'+'percent'+'name',
                        hole = .5,
                        text = 'No Game',
                        textposition = 'inside',
                        type = 'pie'),

                    ],
    layout = dict(title = 'Movement in Bonds Based on Yankees Performance',
                  annotations = [dict(font = dict(size = 20),
                                      showarrow = False,
                                      text = 'Win',
                                      x = .12,
                                      y = .5),
                                dict(font = dict(size = 20),
                                     showarrow = False,
                                     text = 'Loss',
                                     x = .5,
                                     y = .5),
                                dict(font = dict(size = 20),
                                     showarrow = False,
                                     text = 'No Game',
                                     x = .91,
                                     y = .5)
                                ]
                  )
                  )

ply.plot(fig)


#######################
####### VIS 7 #########
#######################

####
# Box and whisker plot looking at Lakers games and industry-specific stocks
####

result2 = DataBuilder('Los Angeles Lakers Outcomes', 'CHEVRON Percent Change')
result3 = DataBuilder('Los Angeles Lakers Outcomes', 'EXXON Percent Change')
result4 = DataBuilder('Los Angeles Lakers Outcomes', 'MARATHON Percent Change')

result5 = DataBuilder('Los Angeles Lakers Outcomes', 'STURM & RUGER (GUNS) Percent Change')
result6 = DataBuilder('Los Angeles Lakers Outcomes', 'AMERICAN OUTDOOR (GUNS) Percent Change')

trace42 = go.Box(x0 = 'CHEVRON (+Win)', y = [x * 100 for x in result2[0]], boxmean = True, name = 'Lakers Win', marker = dict(color = 'red'))
trace43 = go.Box(x0 = 'CHEVRON (+Lose)', y = [x * 100 for x in result2[1]], boxmean = True, name = 'Lakers Lose', marker = dict(color = 'blue'))
trace44 = go.Box(x0 = 'CHEVRON (+No Game)', y = [x * 100 for x in result2[2]], boxmean = True, name = "Lakers Don't Play", marker = dict(color = 'green'))

trace45 = go.Box(x0 = 'EXXON (+Win)', y = [x * 100 for x in result3[0]], boxmean = True, name = 'Lakers Win', showlegend = False, marker = dict(color = 'red'))
trace46 = go.Box(x0 = 'EXXON (+Lose)', y = [x * 100 for x in result3[1]], boxmean = True, name = 'Lakers Lose', showlegend = False, marker = dict(color = 'blue'))
trace47 = go.Box(x0 = 'EXXON (+No Game)', y = [x * 100 for x in result3[2]], boxmean = True, name = "Lakers Don't Play", showlegend = False, marker = dict(color = 'green'))

trace48 = go.Box(x0 = 'MARATHON (+Win)', y = [x * 100 for x in result4[0]], boxmean = True, name = 'Lakers Win', showlegend = False, marker = dict(color = 'red'))
trace49 = go.Box(x0 = 'MARATHON (+Lose)', y = [x * 100 for x in result4[1]], boxmean = True, name = 'Lakers Lose', showlegend = False, marker = dict(color = 'blue'))
trace50 = go.Box(x0 = 'MARATHON (+No Game)', y = [x * 100 for x in result4[2]], boxmean = True, name = "Lakers Don't Play", showlegend = False, marker = dict(color = 'green'))

trace51 = go.Box(x0 = 'STURM & RUGER (+Win)', y = [x * 100 for x in result5[0]], boxmean = True, name = 'Lakers Win', showlegend = False, marker = dict(color = 'red'))
trace52 = go.Box(x0 = 'STURM & RUGER (+Lose)', y = [x * 100 for x in result5[1]], boxmean = True, name = 'Lakers Lose', showlegend = False, marker = dict(color = 'blue'))
trace53 = go.Box(x0 = 'STURM & RUGER (+No Game)', y = [x * 100 for x in result5[2]], boxmean = True, name = "Lakers Don't Play", showlegend = False, marker = dict(color = 'green'))

trace54 = go.Box(x0 = 'AMERICAN OUTDOOR (+Win)', y = [x * 100 for x in result6[0]], boxmean = True, name = 'Lakers Win', showlegend = False, marker = dict(color = 'red'))
trace55 = go.Box(x0 = 'AMERICAN OUTDOOR (+Lose)', y = [x * 100 for x in result6[1]], boxmean = True, name = 'Lakers Lose', showlegend = False, marker = dict(color = 'blue'))
trace56 = go.Box(x0 = 'AMERICAN OUTDOOR (+No Game)', y = [x * 100 for x in result6[2]], boxmean = True, name = "Lakers Don't Play", showlegend = False, marker = dict(color = 'green'))

updatemenus = list([dict(active = -1, buttons = list([
        dict(label = 'ALL',
             method = 'update',
             args = [{'visible': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]},
                     {'title':'ALL'}]),
        dict(label = 'WINS',
             method = 'update',
             args = [{'visible': [True, False, False, True, False, False, True, False, False, True, False, False, True, False, False]},
                     {'title':'WINS'}]),
        dict(label = 'LOSSES',
             method = 'update',
             args = [{'visible': [False, True, False, False, True, False, False, True, False, False, True, False, False, True, False]},
                     {'title':'LOSSES'}]),
        dict(label = 'WINS + LOSSES',
             method = 'update',
             args = [{'visible': [True, True, False, True, True, False, True, True, False, True, True, False, True, True, False]},
                     {'title':'WINS'}])
                  ])
                  )]
                  )
                  

myLayout = go.Layout(
        title = "Industry Stocks After Lakers Wins and Losses",
        xaxis=dict(
                title = 'Stock'
	),
	yaxis=dict(
		title = 'Percentage Change in Market', range = [-5, 5],
	),
    updatemenus = updatemenus
    )

data = [trace42, trace43, trace44, trace45, trace46, trace47, trace48,
        trace49, trace50, trace51, trace52, trace53, trace54, trace55, 
        trace56]

fig = go.Figure(layout = myLayout, data = data)

ply.plot(fig)

##################
##### VIS 8 ######
##################

####
# Plots showing the various statisical results of the
# exploratory analyses
####

allStocks = ['NASDAQ', 'DOW JONES', 'S&P', 'MARATHON', 'CHEVRON', 'EXXON', 'FRANKLIN GPM',
             'FIRST EAGLE', 'STURM & RUGER', 'AMERICAN OUTDOOR', 'ANHEUSER-BUSCH', 'VICEX',
             'RYDEX', 'UTILITIES', 'HEALTHCARE', 'IT', 'BONDS']

means = [.00051, .00041, .00039, .00031, .00045, .00044, .00026, .00023, .00073, .00197,
         .00066, .00035, .00029, .00019, .00038, .00041, 0]

medians = [.00111, .00055, .0005, 0, 0, 0, 0, 0, 0, 0, .00027, .00070, .00065, 0, .00069, .00103, 0]

stdevs = [.01258, .01097, .01068, .02222, .02229, .01445, .01895, .01700, .02641, .05445,
          .01456, .00986, .01427, .00872, .01033, .01277, .00306]

### Means and Medians ###

trace57 = go.Scatter(name = 'MEAN', x = allStocks, y = means, mode = 'markers',
                     marker = dict(size = '15',
                                   color = 'Blue'))

trace58 = go.Scatter(name = 'MEDIAN', x = allStocks, y = medians, mode = 'markers',
                     marker = dict(size = '15',
                                   color = 'Green'))

updatemenus = list([dict(active = -1, buttons = list([
        dict(label = 'BOTH',
             method = 'update',
             args = [{'visible': [True, True, True]},
                     {'title':'ALL'}]),
        dict(label = 'MEANS',
             method = 'update',
             args = [{'visible': [True, False, False]},
                     {'title':'MEANS'}]),
        dict(label = 'MEDIANS',
             method = 'update',
             args = [{'visible': [False, True, False]},
                     {'title':'MEDIANS'}])
                  ])
                  )]
                  )

myLayout = go.Layout(
        title = "Volatility of Stocks Tested",
        xaxis=dict(
                title = 'Stock'
	),
	yaxis=dict(
		title = 'Summary Stats of Daily % Change', range = [-.0002, .0012],
	),
    updatemenus = updatemenus
    )
    
data = [trace57, trace58]
    
myFigure = go.Figure(data = data, layout = myLayout)

ply.plot(myFigure)

### Standard deviations ###

trace59 = go.Scatter(name = 'STANDARD DEVIATION', x = allStocks, y = stdevs, mode = 'markers',
                     marker = dict(size = '25',
                                   color = 'Red'))

myLayout = go.Layout(
        title = "Volatility of Stocks Tested",
        xaxis=dict(
                title = 'Stock'
	),
	yaxis=dict(
		title = 'Standard Deviation of Daily % Change', range = [0, .06],
	)
    )
    
data = [trace59]

myFigure = go.Figure(data = data, layout = myLayout)

ply.plot(myFigure)

####################
##### VIS 9 ########
####################

####
# Trend lines showing the performance of the market and the number
# of professional games, by day
####

stchangedata=stocks1[['Date','NASDAQ Percent Change','DOW JONES Percent Change','S&P Percent Change','MARATHON Percent Change','CHEVRON Percent Change','EXXON Percent Change','FRANKLIN GOLD AND_PRECIOUS METALS Percent Change','FIRST EAGLE GOLD Percent Change','STURM & RUGER (GUNS) Percent Change','AMERICAN OUTDOOR (GUNS) Percent Change','ANHEUSER-BUSCH Percent Change','VICEX (SIN) Percent Change','RYDEX (SIN) Percent Change','UTILITIES Percent Change','HEALTHCARE Percent Change','IT Percent Change','BONDS Percent Change']]
spchangedata=sports1[['Date','NFL Games','MLB Games','NBA Games','Total Games']]
result=pd.merge(stchangedata, spchangedata, on='Date', how='outer')
stweek=[]
for i in range(0,len(stocks1['Date'])):
    stweek.append(datetime.datetime.strptime(stocks1['Date'][i], '%d-%b-%y').strftime('%A'))
    
spweek=[]
for i in range(0,len(sports1['Date'])):
    spweek.append(datetime.datetime.strptime(sports1['Date'][i], '%d-%b-%y').strftime('%A'))

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

dfx=newresult.head(n=3211)
dfx=dfx.dropna()
dfx.loc[:,'NASDAQ Percent Change'] *= 1000
dfx.loc[:,'S&P Percent Change'] *= 1000
dfx.loc[:,'DOW JONES Percent Change'] *= 1000
dfx['Date'] = pd.to_datetime(dfx['Date'])
trace_high = go.Scatter(
    x=dfx['Date'],
    y=dfx['NASDAQ Percent Change'],
    yaxis = 'percent change ',
    name = "NASDAQ Percent Change",
    line = dict(color = '#ad730f',width=2),
    opacity = 0.8)
    
trace_med = go.Scatter(
    x=dfx['Date'],
    y=dfx['S&P Percent Change'],
    name = 'S&P Percent Change',
    line = dict(color = '#6cd334',width=2),
    opacity = 0.8)
trace_ye = go.Scatter(
    x=dfx['Date'],
    y=dfx['DOW JONES Percent Change'],
    name = 'DOW JONES Percent Change',
    line = dict(color = '#9915c1',width=2),
    opacity = 0.8)
trace_low = go.Scatter(
    x=dfx['Date'],
    y=dfx['Total Games'],
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
ply.plot(fig, filename = 'Time Series For Stocks %change versus Total Games')

######################
###### VIS 10 ########
######################

####
# In-depth 3D bubble chart that plots various aspects of the NASDAQ's performance
####

dfz=stocks.head(n=800)
dfz['dif']=dfz['NASDAQ High']-dfz['NASDAQ Low']

dfz=dfz.dropna()
dfz['dif']=dfz['dif'].astype(int)
dfz.loc[:,'dif'] *= 100
trace1 = go.Scatter3d(
    x=dfz['NASDAQ Volume'],
    y=dfz['NASDAQ Change Category'],
    z=dfz['NASDAQ Close'],
    text=dfz['NASDAQ'],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        sizeref=300,
        size=dfz['dif'],
        color = dfz['NASDAQ Percent Change'],
        colorscale = 'Viridis',
        colorbar = dict(title = 'Percent Change'),
        line=dict(color='rgb(140, 140, 170)')
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='Discover Nasdaq Perent change to Nasdaq stocks movement')
fig=dict(data=data, layout=layout)
ply.plot(fig, filename='3DBubble')






