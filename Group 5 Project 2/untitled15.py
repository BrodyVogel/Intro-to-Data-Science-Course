#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:31:30 2017

@author: vogebr01
"""

import networkx as nx
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import datetime
import numpy as np

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



# Sets path to current folder where file exists
working_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_path)


# Define function to strip once re.findall function finds data
def restrip(x):
    y = str(x).strip("['").strip("']")
    return y


# NFL Champions
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
