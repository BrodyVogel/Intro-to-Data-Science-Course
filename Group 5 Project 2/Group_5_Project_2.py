#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:55:36 2017

@author: Brody Vogel, Jinghao Yan, Zihao Li, Ye Zhang
"""

### Import statements
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import apyori
import datetime 
import os
from sklearn.cluster import AgglomerativeClustering as agg
from sklearn.cluster import DBSCAN as db
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats as ss
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import math
from sklearn.metrics import roc_curve
from sklearn.neighbors import kneighbors_graph
from sklearn import decomposition
import re
from sklearn.ensemble import RandomForestClassifier as rf
import os
import statsmodels.formula.api as smf



### Read in our data
sports = pd.read_csv('/Users/brodyvogel/Desktop/Group 5 Project 2/SPORTS_FINAL.csv')
stocks = pd.read_csv('/Users/brodyvogel/Desktop/Group 5 Project 2/STOCKS_FINAL.csv', thousands = ',')

### Duplicate index from previous cleaning needs fixed
stocks = stocks[stocks.columns[2:]]
sports = sports[sports.columns[1:]]

### Date transition from Excel wasn't clean, so we have to fix the format
index = 0
for entry in stocks['Date']:
    stocks.iloc[index, 0] = entry.replace('"', '')
    index += 1

######################################
######## New Variables Creation ######    
######################################

print('\n##################################')    
print('Working on Variable Creation....\n')
print('##################################')
    
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
                
#########################################
####### Basic Statistical Analysis ######        
#########################################   
    
print('\n####################################################')
print('Starting the Basic Statistical Analyses....(Part 1)')
print('####################################################')
    
### Function to make things easier; have to account for a lot of NaNs because of 
    ### the different amounts of data available for the varying stocks
def StatsSummary(list, name):
    print('\nThe mean of', name,  'is: ', np.nanmean(list))
    print('The median of', name,  'is: ', np.nanmedian(list))
    print('The standard deviation of', name, 'is: ', np.nanstd(list))
        
### Summarize the 17 percentage change variables, since they're the most important to our project
for column in percentCols:
    StatsSummary(stocks[column], column)
    
#######################################
########## Outlier Discovery ##########
#######################################
    
print('\n####################')
print('Finding Outliers....')
print('####################')    
    
### Function that finds outliers
def OutlierFinder(list, name):
    outList = []
    top, bottom = np.nanpercentile(list, [75 ,25])
    IQR = top - bottom
    for entry in list:
        ### Notice we had to really change the IQR detection method because the stock market is so volatile
        if entry > (5 * IQR + top) or entry < (bottom - 5 * IQR):
            outList.append(entry)
    return(outList)    

### Find and number the outliers
totalOuts = []
for column in percentCols:
    #print(OutlierFinder(stocks[column], column))
    totalOuts.append(len(OutlierFinder(stocks[column], column)))
    
print('\nThere are', sum(totalOuts), 'outliers in the data\n')
    
##################################
############## Plotting ##########
##################################

print('\n########################')
print('Building Plots...(Part 2)')
print('##########################')

### Create histograms of the percentage changes
for column in percentCols:
    data = stocks[column][pd.isnull(stocks[column]) == False]
    plt.hist(data, bins = 40)
    plt.title(column)
    plt.show()
    
###################
#Correlation Check#    
###################
    
print('\n############################')
print('Correlation Calculations...')
print('############################')
    
### Examine the correlation between 3 (hopefully) disparate stock percentage changes
pairedCols = ['NASDAQ Percent Change', 'BONDS Percent Change', 'VICEX (SIN) Percent Change', 'UTILITIES Percent Change', 'AMERICAN OUTDOOR (GUNS) Percent Change']

### Correlation check function to make life easier
def CorrCheck(one, two):
    one1 = stocks[one]
    two1 = stocks[two]
    one1 = one1[pd.isnull(one1) == False]
    two1 = two1[pd.isnull(two1) == False]
    smallest = min(len(one1), len(two1))
    first = one1[:smallest-1]
    second = two1[:smallest-1]
    print('\nThe correlation coefficient between', one, 'and', two, 'is: ', np.correlate(first, second)[0])

    plt.scatter(first, second)
    title = 'Correlation Between ' + one + ' (x-axis) and ' + two + '(y-axis)'
    plt.title(title)
    plt.show()

CorrCheck(pairedCols[0], pairedCols[1])
CorrCheck(pairedCols[0], pairedCols[2])
CorrCheck(pairedCols[0], pairedCols[3])
CorrCheck(pairedCols[0], pairedCols[4])
CorrCheck(pairedCols[1], pairedCols[2])
CorrCheck(pairedCols[1], pairedCols[3])
CorrCheck(pairedCols[1], pairedCols[4])
CorrCheck(pairedCols[2], pairedCols[3])
CorrCheck(pairedCols[2], pairedCols[4])
CorrCheck(pairedCols[3], pairedCols[4])

############################################
############### CLUSTERING #################
############################################

print('###################################')
print('Starting the Clustering....(Part 3)')
print('###################################')

### Percent change columns
percentcols = percentCols

### Create separate data frame for clustering purposes
stocks_clust = stocks.copy()

### Replace nan's with mean's of each column
for i in stocks_clust[percentcols]:
    stocks_clust[i] = stocks_clust[i].fillna(np.nanmean(stocks_clust[i]))

# Sets path to current folder where file exists
working_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_path)

##### Hierarchical clustering #####

### Define function for plotting hierarchical clustering
def hierplot (k, dataframe):
    # Normalize data
    x = dataframe.values
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    # Cluster labels
    #knn = kneighbors_graph(normalizedDataFrame, 30)
    hier= agg(linkage='ward', connectivity = None, n_clusters=k)
    cluster_labels = hier.fit_predict(normalizedDataFrame)
    
    # Convert high dimensional data into 2 dimensions
    pca2D = decomposition.PCA(2)

    # Turn data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    
    # Plot using a scatter plot and shade by cluster label
    plt.clf()
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("Ward Linkage")
    plt.show()
    # Saves plot - comment out if preferred
    plt.savefig(working_path + '\\' + 'Hier_' + str(k) + '.png')

### Plot for different values of k
for i in range(2,6):
    hierplot(i, stocks_clust[percentcols])


### Define function for measuring cluster quality using Silhouette procedure
def sil_avg (k, dataframe):
    # Normalize data
    x = dataframe.values
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    # Cluster labels
    #knn = kneighbors_graph(normalizedDataFrame, 30)
    hier= agg(linkage='ward', connectivity = None, n_clusters=k)
    cluster_labels = hier.fit_predict(normalizedDataFrame)
    
    # Calculate Silhouette average
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "the average silhouette_score is :", silhouette_avg)

### Measure cluster quality
for i in range(2,6):
    sil_avg(i, stocks_clust[percentcols])
# n=2 gives best silhouette average for Ward linkage - 0.465445 (ok indicator of clustering)

##### K-means clustering #####

### Define function for plotting k-means clustering
def kmeanplot (k, dataframe):
    # Normalize data
    x = dataframe.values
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    # Cluster labels
    kmeans= KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Convert high dimensional data into 2 dimensions
    pca2D = decomposition.PCA(2)

    # Turn data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    
    # Plot using a scatter plot and shade by cluster label
    plt.clf()
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("K-means Clustering")
    plt.show()
    # Saves plot - comment out if preferred
    plt.savefig(working_path + '\\' + 'KMean_' + str(k) + '.png')

### Plot for different values of k
for i in range(2,6):
    kmeanplot(i, stocks_clust[percentcols])


### Define function for measuring cluster quality using Silhouette procedure
def sil_avg (k, dataframe):
    # Normalize data
    x = dataframe.values
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    # Cluster labels
    kmeans= KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    
    # Calculate Silhouette average
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "the average silhouette_score is :", silhouette_avg)

### Measure cluster quality
for i in range(2,6):
    sil_avg(i, stocks_clust[percentcols])
# n=3 gives best silhouette average for k-means - 0.282306 (still not good indicator of clustering)


##### DBScan clustering #####

### Define function for plotting dbscan clustering
def dbplot (e, dataframe):
    # Normalize data
    x = dataframe.values
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    # Cluster labels
    dbscan= db(eps=e, min_samples=20)
    cluster_labels = dbscan.fit_predict(normalizedDataFrame)
    
    # Convert high dimensional data into 2 dimensions
    pca2D = decomposition.PCA(2)

    # Turn data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    
    # Plot using a scatter plot and shade by cluster label
    plt.clf()
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("DBSCAN Clustering")
    plt.show()
    # Saves plot - comment out if preferred
    plt.savefig(working_path + '\\' + 'DB_' + str(e) + '.png')

### Plot for different values of k
for i in (0.5, 1.5, 3.5, 5.5):
    dbplot(i, stocks_clust[percentcols])


### Define function for measuring cluster quality using Silhouette procedure
def sil_avg (e, dataframe):
    # Normalize data
    x = dataframe.values
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    # Cluster labels
    dbscan=db(eps=e, min_samples=20)
    cluster_labels = dbscan.fit_predict(normalizedDataFrame)
    
    # Calculate Silhouette average
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For eps =", e, "the average silhouette_score is :", silhouette_avg)

### Measure cluster quality
for i in (0.5, 1.5, 3.5, 5.5):
    sil_avg(i, stocks_clust[percentcols])
# eps=5.5 gives best silhouette average for dbscan - -0.71538234 (decent indicator of clustering)


### Slightly different clustering - specifc to the NASDAQ Index, this tracks that
    ### stock's activity
fstocks=pd.DataFrame(stocks)

nasdaq=pd.DataFrame(stocks,columns=['NASDAQ Open', 'NASDAQ High', 'NASDAQ Low', 'NASDAQ Close','NASDAQ Volume'])
nasdaq.isnull().any()
nasdaq = nasdaq[np.isfinite(nasdaq['NASDAQ Volume'])]
nasdaq.dtypes

naopen=[]
naopen=nasdaq['NASDAQ Open']
naclose=[]
naclose=nasdaq['NASDAQ Close']
nasdaq['NASDAQ OCdiff']=nasdaq['NASDAQ Close']-nasdaq['NASDAQ Open']

nasdaq=nasdaq.drop('NASDAQ Open', axis=1)
nasdaq=nasdaq.drop('NASDAQ Close', axis=1)
nasdaq=nasdaq.drop('NASDAQ Volume', axis=1)
dc = db(eps=50, min_samples=5,metric="euclidean").fit(nasdaq)
km =  KMeans(n_clusters=3, random_state=0).fit(nasdaq)
labels = dc.labels_
kmlabels = km.labels_
nasdaq['labels'] = labels
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(nasdaq['NASDAQ High'], nasdaq['NASDAQ Low'], nasdaq['NASDAQ OCdiff'],cmap='seismic', s=50)
ax.set_xlabel('NASDAQ High')
ax.set_ylabel('NASDAQ Low')
ax.set_zlabel('NASDAQ OCdiff')
plt.show()
plt.clf()
plt.cla()
plt.close()
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(nasdaq['NASDAQ High'], nasdaq['NASDAQ Low'], nasdaq['NASDAQ OCdiff'],c=nasdaq['labels'],cmap='Set1', s=50)
ax.set_xlabel('NASDAQ High')
ax.set_ylabel('NASDAQ Low')
ax.set_zlabel('NASDAQ OCdiff')
plt.show()
plt.clf()
plt.cla()
plt.close()


nasdaq['kmlabels'] = km.labels_
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(nasdaq['NASDAQ High'], nasdaq['NASDAQ Low'], nasdaq['NASDAQ OCdiff'],c=nasdaq['kmlabels'],cmap='Set1', s=50)
ax.set_xlabel('NASDAQ High')
ax.set_ylabel('NASDAQ Low')
ax.set_zlabel('NASDAQ OCdiff')
plt.show()
plt.clf()
plt.cla()
plt.close()
nasdaq=nasdaq.drop('labels', axis=1)
nasdaq=nasdaq.drop('kmlabels', axis=1)
ward = agg(n_clusters=3, linkage='ward').fit(nasdaq)
wlabel = ward.labels_
nasdaq['wlabels'] = ward.labels_

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(nasdaq['NASDAQ High'], nasdaq['NASDAQ Low'], nasdaq['NASDAQ OCdiff'],c=nasdaq['wlabels'],cmap='Set1', s=50)
ax.set_xlabel('NASDAQ High')
ax.set_ylabel('NASDAQ Low')
ax.set_zlabel('NASDAQ OCdiff')
plt.show()
plt.clf()
plt.cla()
plt.close()
nasdaq=nasdaq.drop('wlabels', axis=1)
km_labels = km.fit_predict(nasdaq)
silhouette_avg = silhouette_score(nasdaq, km_labels)
print("For n_clusters =", 3, "The average silhouette_score of kmean is :", silhouette_avg)
db_labels = dc.fit_predict(nasdaq)
dbsilhouette_avg = silhouette_score(nasdaq, db_labels)
print("For n_clusters =", 5, "The average silhouette_score of dbscan is :", dbsilhouette_avg)
w_labels = ward.fit_predict(nasdaq)
wsilhouette_avg = silhouette_score(nasdaq, w_labels)
print("For n_clusters =", 3, "The average silhouette_score of ward is :", wsilhouette_avg)

#########################################
########## Association Rule Mining ######
######################################### 

print('\n################################################')
print('Starting the Association Rule Mining....(Part 4)')
print('################################################')

### Grab a list of the outcome variables from the sports data
outcomeCols = []
for column in sports.columns:
    if 'Outcomes' in column:
        outcomeCols.append(column)
        
### Get the binned variables from the stocks data
changeCols = []
for column in stocks.columns:
    if 'Change Category' in column:
        changeCols.append(column)       

### Create test sets for the association rule mining
        
### These are the test values we actually used, but 
    ### it'll jack the runtime up over an hour

#testCols = ['NASDAQ Change Category', 'EXXON Change Category', 'FRANKLIN GOLD AND_PRECIOUS METALS Change Category', 'STURM & RUGER (GUNS) Change Category', 'ANHEUSER-BUSCH Change Category', 'VICEX (SIN) Change Category', 'UTILITIES Change Category', 'HEALTHCARE Change Category', 'IT Change Category', 'BONDS Change Category']
#testTeams = ['Dallas Cowboys Outcomes', 'New York Yankees Outcomes', 'Los Angeles Lakers Outcomes']

testCols = ['BONDS Change Category']
testTeams = ['Los Angeles Lakers Outcomes']


### Function that creates a list of possible associations and then runs the Apriori 
    ### Algorithm on that list, creating possible association rules
def apr(colName1, colName2):
    associations = []
    for day in sports['Date']:
        small = sports.loc[sports['Date'] == day]
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
                for column in [colName1]:
                    for entry in small[column]:
                        if entry == -1:
                            for col in [colName2]:
                                for ent in check[col]:
                                    if pd.isnull(ent) == False:
                                        string1 = column + ' Loss'
                                        string2 = col + ' ' + ent  
                                        associations.append([string1, string2])
                        elif entry == 1:
                            for col in [colName2]:
                                for ent in check[col]:
                                    if pd.isnull(ent) == False:
                                        string1 = column + ' Win'
                                        string2 = col + ' ' + ent
                                        associations.append([string1, string2])
                                        
            elif len(stocks.loc[stocks['Date'] == row1]) != 0:
                check = stocks.loc[stocks['Date'] == row1]
                for column in [colName1]:
                    for entry in small[column]:
                        if entry == -1:
                            for col in [colName2]:
                                for ent in check[col]:
                                    if pd.isnull(ent) == False:
                                        string1 = column + ' Loss'
                                        string2 = col + ' ' + ent  
                                        associations.append([string1, string2])
                        elif entry == 1:
                            for col in [colName2]:
                                for ent in check[col]:
                                    if pd.isnull(ent) == False:
                                        string1 = column + ' Win'
                                        string2 = col + ' ' + ent
                                        associations.append([string1, string2])
                                        
            elif len(stocks.loc[stocks['Date'] == row2]) != 0:
                check = stocks.loc[stocks['Date'] == row2]
                for column in [colName1]:
                    for entry in small[column]:
                        if entry == -1:
                            for col in [colName2]:
                                for ent in check[col]:
                                    if pd.isnull(ent) == False:
                                        string1 = column + ' Loss'
                                        string2 = col + ' ' + ent  
                                        associations.append([string1, string2])
                        elif entry == 1:
                            for col in [colName2]:
                                for ent in check[col]:
                                    if pd.isnull(ent) == False:
                                        string1 = column + ' Win'
                                        string2 = col + ' ' + ent
                                        associations.append([string1, string2])


            
        except:
            continue
    
    return(list(apyori.apriori(associations)))


###Snag the list of all possible association rules
possiblySignificant = []
for col in testTeams:
    for column in testCols:
        out = apr(col, column)
        for num in range(len(out)):
            if len((out)[num][0]) == 2:
                print(out[num][0])
                possiblySignificant.append(out[num])
                
###Prune down the list of possible association rules to only those with relatively
    ### significant support values (emphasis on relatively)
point2 = []
point25 = []
point28 = []

for value in possiblySignificant:
    sup = value[1]
    if sup > .28:
        point2.append(value)
        point25.append(value)
        point28.append(value)
    elif sup > .25:
        point2.append(value)
        point25.append(value)
    elif sup > .2:
        point2.append(value)
        
### Spit out the possibly useful association rules with their numbers
strngs = ['.2', '.25', '.28']
i = 0
for bunch in [point2, point25, point28]:
    print('\nAssociations with >', strngs[i], 'Support: ')
    for entry in bunch:
        print('\nRelation: ', entry[0])
        print('support: ', entry[1])
        print('confidence: ', entry[2][1][2])
    i += 1


####################
#Hypothesis Testing#
####################
    

### Hypothesis 1 ###
## Does the performance of the New York Yankees, 
    ## one of baseball's biggest payroll and market-size teams, affect any 
        ## of the stocks/indices? ##
        
print('\n######################################')
print('# Beginning Hypothesis One...(Part 4) #')
print('\n######################################')



# Define function to strip once re.findall function finds data
def restrip(x):
    y = str(x).strip("['").strip("']")
    return y

# Stock we are testing on
for sn in [ 'S&P', 'ANHEUSER-BUSCH', 'VICEX (SIN)', 'EXXON', 'FIRST EAGLE GOLD', 'BONDS']:
    stockname = sn
    print('\nFor '+ stockname + '...')
    yanks = 'New York Yankees'
    date = 'Date'
    # New data frame containing necessary columns for hypothesis testing
    yanks_df = pd.DataFrame(columns = ['Team', 'Date', 'W/L', 'W/L_Cat', 'Score Difference', 'Stock Date', 'Stock', 'Stock Percent Change', 'Stock Change Category', 'Stock Performance'])
    count = 0
    for i in range(0,len(sports[yanks])):
        if pd.isnull(sports.loc[i, yanks]) == False:
            # Find column attributes, such as Win/Loss, difference in score
            # Find subsequent days after game day, up to 4 days
            W_L = sports.loc[i, yanks][:sports.loc[i, yanks].find('/')]
            diff = int(restrip(re.findall('[A-Z]/([0-9]+)/', sports.loc[i, yanks]))) - int(sports.loc[i, yanks][sports.loc[i, yanks].rfind('/')+1:])
            newdate = datetime.datetime.strptime(sports.loc[i, date], '%d-%b-%y').strftime('%d-%b-%y')
            dayafter = (datetime.datetime.strptime(newdate, '%d-%b-%y') + datetime.timedelta(days=1)).strftime('%d-%b-%y')
            dayafter_stocks = datetime.datetime.strptime(dayafter, '%d-%b-%y').strftime('%b %d, %Y')
            dayafter2 = (datetime.datetime.strptime(newdate, '%d-%b-%y') + datetime.timedelta(days=2)).strftime('%d-%b-%y')
            dayafter_stocks2 = datetime.datetime.strptime(dayafter2, '%d-%b-%y').strftime('%b %d, %Y')
            dayafter3 = (datetime.datetime.strptime(newdate, '%d-%b-%y') + datetime.timedelta(days=3)).strftime('%d-%b-%y')
            dayafter_stocks3 = datetime.datetime.strptime(dayafter3, '%d-%b-%y').strftime('%b %d, %Y')
            dayafter4 = (datetime.datetime.strptime(newdate, '%d-%b-%y') + datetime.timedelta(days=4)).strftime('%d-%b-%y')
            dayafter_stocks4 = datetime.datetime.strptime(dayafter4, '%d-%b-%y').strftime('%b %d, %Y')
            # If first day after game day in stock data frame (i.e. Sundays)
            if dayafter_stocks in stocks[date].values:
                index_stock = stocks[date][stocks[date] == dayafter_stocks].index[0]
                if pd.isnull(stocks.loc[index_stock, date]) == False and pd.isnull(stocks.loc[index_stock, stockname + ' Percent Change']) == False:
                    count = count + 1
                    yanks_df.loc[count, 'Team'] = yanks
                    yanks_df.loc[count, 'Date'] = newdate
                    yanks_df.loc[count, 'W/L'] = W_L
                    if W_L == 'W':
                        yanks_df.loc[count, 'W/L_Cat'] = 1
                    else:
                        yanks_df.loc[count, 'W/L_Cat'] = 0
                    yanks_df.loc[count, 'Score Difference'] = diff
                    yanks_df.loc[count, 'Stock Date'] = dayafter_stocks
                    yanks_df.loc[count, 'Stock'] = stockname
                    yanks_df.loc[count, 'Stock Percent Change'] = stocks.loc[index_stock, stockname + ' Percent Change']
                    yanks_df.loc[count, 'Stock Change Category'] = stocks.loc[index_stock, stockname + ' Change Category']
                    if stocks.loc[index_stock, stockname + ' Percent Change'] > 0:
                            yanks_df.loc[count, 'Stock Performance'] = 1
                    else:
                        yanks_df.loc[count, 'Stock Performance'] = 0
            # If second day after game day in stock data frame (i.e. Saturdays)
            elif dayafter_stocks2 in stocks[date].values:
                index_stock2 = stocks[date][stocks[date] == dayafter_stocks2].index[0]
                if pd.isnull(stocks.loc[index_stock2, date]) == False and pd.isnull(stocks.loc[index_stock2, stockname + ' Percent Change']) == False:
                    count = count + 1
                    yanks_df.loc[count, 'Team'] = yanks
                    yanks_df.loc[count, 'Date'] = newdate
                    yanks_df.loc[count, 'W/L'] = W_L
                    if W_L == 'W':
                        yanks_df.loc[count, 'W/L_Cat'] = 1
                    else:
                        yanks_df.loc[count, 'W/L_Cat'] = 0
                    yanks_df.loc[count, 'Score Difference'] = diff
                    yanks_df.loc[count, 'Stock Date'] = dayafter_stocks2
                    yanks_df.loc[count, 'Stock'] = stockname
                    yanks_df.loc[count, 'Stock Percent Change'] = stocks.loc[index_stock2, stockname + ' Percent Change']
                    yanks_df.loc[count, 'Stock Change Category'] = stocks.loc[index_stock2, stockname + ' Change Category']
                    if stocks.loc[index_stock2, stockname + ' Percent Change'] > 0:
                        yanks_df.loc[count, 'Stock Performance'] = 1
                    else:
                        yanks_df.loc[count, 'Stock Performance'] = 0
            # If third day after game day in stock data frame (i.e. Fridays)
            elif dayafter_stocks3 in stocks[date].values:
                index_stock3 = stocks[date][stocks[date] == dayafter_stocks3].index[0]
                if pd.isnull(stocks.loc[index_stock3, date]) == False and pd.isnull(stocks.loc[index_stock3, stockname + ' Percent Change']) == False:
                    count = count + 1
                    yanks_df.loc[count, 'Team'] = yanks
                    yanks_df.loc[count, 'Date'] = newdate
                    yanks_df.loc[count, 'W/L'] = W_L
                    if W_L == 'W':
                        yanks_df.loc[count, 'W/L_Cat'] = 1
                    else:
                        yanks_df.loc[count, 'W/L_Cat'] = 0
                    yanks_df.loc[count, 'Score Difference'] = diff
                    yanks_df.loc[count, 'Stock Date'] = dayafter_stocks3
                    yanks_df.loc[count, 'Stock'] = stockname
                    yanks_df.loc[count, 'Stock Percent Change'] = stocks.loc[index_stock3, stockname + ' Percent Change']
                    yanks_df.loc[count, 'Stock Change Category'] = stocks.loc[index_stock3, stockname + ' Change Category']
                    if stocks.loc[index_stock3, stockname + ' Percent Change'] > 0:
                        yanks_df.loc[count, 'Stock Performance'] = 1
                    else:
                        yanks_df.loc[count, 'Stock Performance'] = 0
            # If fourth day after game day in stock data frame (i.e. Fridays of Labor Day weekends)
            elif dayafter_stocks4 in stocks[date].values:
                index_stock4 = stocks[date][stocks[date] == dayafter_stocks4].index[0]
                if pd.isnull(stocks.loc[index_stock4, date]) == False and pd.isnull(stocks.loc[index_stock4, stockname + ' Percent Change']) == False:
                    count = count + 1
                    yanks_df.loc[count, 'Team'] = yanks
                    yanks_df.loc[count, 'Date'] = newdate
                    yanks_df.loc[count, 'W/L'] = W_L
                    if W_L == 'W':
                        yanks_df.loc[count, 'W/L_Cat'] = 1
                    else:
                        yanks_df.loc[count, 'W/L_Cat'] = 0
                    yanks_df.loc[count, 'Score Difference'] = diff
                    yanks_df.loc[count, 'Stock Date'] = dayafter_stocks4
                    yanks_df.loc[count, 'Stock'] = stockname
                    yanks_df.loc[count, 'Stock Percent Change'] = stocks.loc[index_stock4, stockname + ' Percent Change']
                    yanks_df.loc[count, 'Stock Change Category'] = stocks.loc[index_stock4, stockname + ' Change Category']
                    if stocks.loc[index_stock4, stockname + ' Percent Change'] > 0:
                        yanks_df.loc[count, 'Stock Performance'] = 1
                    else:
                        yanks_df.loc[count, 'Stock Performance'] = 0
    
    
    # Sets path to current folder where file exists
    working_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(working_path)
    
    roc_folder = 'ROC_Plots'
    rf_folder = 'RandomForest'
    nb_folder = 'NaiveBayes'
    svm_folder = 'SVM'
    
    # Create folders for ROC plots - comment out if preferred
    if not os.path.isdir(os.path.join(working_path, roc_folder)):
        os.makedirs(os.path.join(working_path, roc_folder))
        os.makedirs(os.path.join(working_path, roc_folder, rf_folder))
        os.makedirs(os.path.join(working_path, roc_folder, nb_folder))
        os.makedirs(os.path.join(working_path, roc_folder, svm_folder))
    
    
    # Separate training and final validation data set. First remove class
    # label from data (X). Setup target class (Y)
    # Then make the validation set 20% of the entire
    # set of labeled data (X_validate, Y_validate)
    valueArray = yanks_df.values
    X = valueArray[:,[3,4]]
    # Normalize data
    X_norm = preprocessing.normalize(X, norm='l2')
    Y = valueArray[:,9]
    test_size = 0.20
    seed = 8
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X_norm, Y, test_size=test_size, random_state=seed)
    
    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    num_folds = 10
    num_instances = len(X_train)
    seed = 8
    scoring = 'accuracy'
    
    ######################################################
    # Use different algorithms to build models
    ######################################################
    
    # Add each algorithm and its name to the model array
    models = []
    models.append(('RandomForest', rf(n_estimators=100))) #100 trees in forest
    models.append(('NaiveBayes', GaussianNB()))
    models.append(('SVM', SVC()))
    
    # Evaluate each model
    # Print accuracy results
    # ROC plots
    results = []
    names = []
    for name, model in models:
        print('\nCalculating ' + name + '...')
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train.astype('int'), cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s %s: %f (%f)" % (stockname, name, cv_results.mean(), cv_results.std())
        model.fit(X_train, Y_train.astype('int'))
        predictions = model.predict(X_validate)
        print()
        print(msg)
        print(accuracy_score(Y_validate.astype('int'), predictions))
        print(confusion_matrix(Y_validate.astype('int'), predictions))
        fpr, tpr, _ = roc_curve(Y_validate.astype('int'), predictions)
        plt.clf()
        plt.plot(fpr, tpr)
        plt.title(stockname + '_' + name + ' ROC Analysis')
        # Saves plot - comment out if preferred
        plt.savefig(os.path.join(working_path, roc_folder, name) + '\\' + stockname +'_' + name + '_ROC' + '.png')
        plt.close()    

#################################################################################   
### Hypothesis 2: The market reacts to the Lakers' performance ##################
#################################################################################
        
print('\n#############################')
print('# Beginning Hypothesis Two... #')
print('\n#############################')

##Function to make it easier to test different stocks according to the Lakers' performance
def Hyp2(teamOutcomes, stockChangeCat, stockChangePer):
### This all creates lists of the stock percentage change the day after the Lakers
    ### won, lost, or didn't play - kind of a nasty loop but the runtime isn't bad    
    TeamWin = []
    TeamLose = []
    NoGame = []

    indexer = 0
    testTeam = []
    testOutcomes = []

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
                    testOutcomes.append(list(check[stockChangeCat]))
                    NoGame.append(float(check[stockChangePer]))
                elif sports[teamOutcomes][indexer] == 1:
                    testTeam.append(1)
                    testOutcomes.append(list(check[stockChangeCat]))
                    TeamWin.append(float(check[stockChangePer]))
                else:
                    testTeam.append(-1)
                    testOutcomes.append(list(check[stockChangeCat]))
                    TeamLose.append(float(check[stockChangePer]))
                    
            elif len(stocks.loc[stocks['Date'] == row1]) != 0:
                check = stocks.loc[stocks['Date'] == row1]
                
                if sports[teamOutcomes][indexer] == 0:
                    testTeam.append(0)
                    testOutcomes.append(list(check[stockChangeCat]))
                    NoGame.append(float(check[stockChangePer]))
                elif sports[teamOutcomes][indexer] == 1:
                    testTeam.append(1)
                    testOutcomes.append(list(check[stockChangeCat]))
                    TeamWin.append(float(check[stockChangePer]))
                else:
                    testTeam.append(-1)
                    testOutcomes.append(list(check[stockChangeCat]))
                    TeamLose.append(float(check[stockChangePer]))

            elif len(stocks.loc[stocks['Date'] == row2]) != 0:
                check = stocks.loc[stocks['Date'] == row2]
                
                if sports[teamOutcomes][indexer] == 0:
                    testTeam.append(0)
                    testOutcomes.append(list(check[stockChangeCat]))
                    NoGame.append(float(check[stockChangePer]))
                elif sports[teamOutcomes][indexer] == 1:
                    testTeam.append(1)
                    testOutcomes.append(list(check[stockChangeCat]))
                    TeamWin.append(float(check[stockChangePer]))
                else:
                    testTeam.append(-1)
                    testOutcomes.append(list(check[stockChangeCat]))
                    TeamLose.append(float(check[stockChangePer]))

                
        except:
            continue

        indexer += 1
        
    ### Get rid of pesky NAs from when Lakers played but the stock did not yet exist
    TeamWin = [value for value in TeamWin if not math.isnan(value)]
    TeamLose = [value for value in TeamLose if not math.isnan(value)]
    NoGame = [value for value in NoGame if not math.isnan(value)]
    
    print('\nWin Mean: ', np.nanmean(TeamWin), 'Lose Mean: ', np.nanmean(TeamLose), 'No Game Mean: ', np.nanmean(NoGame))
        
    ### Perform the t-test and one-way ANOVA - not particularly good numbers
    print('\nANOVA for', teamOutcomes, 'and', stockChangePer, ss.f_oneway(TeamWin, TeamLose, NoGame))
    print('\nT-test for difference in stocks after Win/Loss in', teamOutcomes, ss.ttest_ind(TeamWin, TeamLose))

    ### Decision Tree for Hyptothesis 2
    count = 0
    for i in testOutcomes:
        if i == []:
            testOutcomes[count] = ' '
        else:
            testOutcomes[count] = i[0]
        count += 1

    ### Make the stock change integers for ease
    count = 0    
    for j in testOutcomes:
        if j == 'BIG DIP':
            testOutcomes[count] = 1
        elif j == 'DIP':
            testOutcomes[count] = 2
        elif j == 'LITTLE MOVEMENT':
            testOutcomes[count] = 3
        elif j == 'JUMP':
            testOutcomes[count] = 4
        elif j == 'BIG JUMP':
            testOutcomes[count] = 5
        else:
            testOutcomes[count] = 0
        
        count += 1
    
    ### Get rid of empty outcomes, like before
    d = {'Team Outcomes':testTeam, 'Stock Changes':testOutcomes}
    testDF = pd.DataFrame(data = d)
    testDF = testDF.dropna(thresh = 2)
    testDF = testDF.reset_index(drop = True)


    ### Create training and test data
    valueArray = testDF.values
    X1 = valueArray[:, 1]
    Y1 = valueArray[:,0]
    Y1 = Y1.astype(int)
    test_size = 0.20
    seed = 6

    X_train1, X_validate1, Y_train1, Y_validate1 = cross_validation.train_test_split(X1, Y1, test_size=test_size, random_state=seed)

    X_train1 = X_train1[:, None]
    X_validate1 = X_validate1[:, None]

    ### Create decision tree classifier
    DT = DecisionTreeClassifier()
    DT = DT.fit(X_train1, Y_train1)
    y_pred = DT.predict(X_validate1)

    ### Get results via confusion matrix and accuracy score - both very poor, as a rule
    print('\nDecision Tree Confusion Matrix for', teamOutcomes, 'and', stockChangeCat, '\n', confusion_matrix(Y_validate1, y_pred))

    print('Related Accuracy Score: ', accuracy_score(Y_validate1, y_pred))

    ### K-Nearest Neighbors test on hypothesis one

    ### Create classifier
    KNN = KNeighborsClassifier(n_neighbors = 3)
    KNN = KNN.fit(X_train1, Y_train1)
    y_pred1 = KNN.predict(X_validate1)

    ### Get results via confusion matrix and accuracy score - somehow even worse
    print('\nKNN Decision Tree Confusion Matrix for', teamOutcomes, 'and', stockChangeCat, '\n', confusion_matrix(Y_validate1, y_pred1))

    print('Related Accuracy Score: ', accuracy_score(Y_validate1, y_pred1))
    

Hyp2('Los Angeles Lakers Outcomes', 'BONDS Change Category', 'BONDS Percent Change')
#Hyp2('Los Angeles Lakers Outcomes', 'NASDAQ Change Category', 'NASDAQ Percent Change')
#Hyp2('Los Angeles Lakers Outcomes', 'CHEVRON Change Category', 'CHEVRON Percent Change')
#Hyp2('Los Angeles Lakers Outcomes', 'FRANKLIN GOLD AND_PRECIOUS METALS Change Category', 'FRANKLIN GOLD AND_PRECIOUS METALS Percent Change')
#Hyp2('Los Angeles Lakers Outcomes', 'STURM & RUGER (GUNS) Change Category', 'STURM & RUGER (GUNS) Percent Change')

### to test if the Chevron and Sturm & Ruger (Guns) numbers were typical
#Hyp2('Los Angeles Lakers Outcomes', 'AMERICAN OUTDOOR (GUNS) Change Category', 'AMERICAN OUTDOOR (GUNS) Percent Change')
#Hyp2('Los Angeles Lakers Outcomes', 'EXXON Change Category', 'EXXON Percent Change')
#Hyp2('Los Angeles Lakers Outcomes', 'MARATHON Change Category', 'MARATHON Percent Change')

################################################################################
### Hypothesis 3: The Stock Market Reacts to the Number of Professional Sports #
###### Games Played the Day Before #############################################
################################################################################

print('\n#############################')
print('# Beginning Hypothesis Three... #')
print('\n#############################')

### functions to make it easier to track different combinations
def Hyp3(stockChangePer, gamesToTrack):
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
            Outcomes.append(float(check[stockChangePer]))


        elif len(stocks.loc[stocks['Date'] == row1]) != 0:
            check = stocks.loc[stocks['Date'] == row1]                  
            NFLGames.append(sports['NFL Games'][indexer])
            MLBGames.append(sports['MLB Games'][indexer])
            NBAGames.append(sports['NBA Games'][indexer])
            TotalGames.append(sports['Total Games'][indexer])
            Outcomes.append(float(check[stockChangePer]))
        
        elif len(stocks.loc[stocks['Date'] == row2]) != 0:
            check = stocks.loc[stocks['Date'] == row2]
            NFLGames.append(sports['NFL Games'][indexer])
            MLBGames.append(sports['MLB Games'][indexer])
            NBAGames.append(sports['NBA Games'][indexer])
            TotalGames.append(sports['Total Games'][indexer])
            Outcomes.append(float(check[stockChangePer]))
        
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
    
    test0 = testDF[testDF['Games Bin'] == 'Bucket 0']['Outcomes']
    test1 = testDF[testDF['Games Bin'] == 'Bucket 1']['Outcomes']
    test2 = testDF[testDF['Games Bin'] == 'Bucket 2']['Outcomes']
    test3 = testDF[testDF['Games Bin'] == 'Bucket 3']['Outcomes']
    test4 = testDF[testDF['Games Bin'] == 'Bucket 4']['Outcomes']

    ### Perform an ANOVA test for significance in difference between bucket means
    print('ANOVA results for difference between means based on number of games: ', ss.f_oneway(test0, test1, test2, test3, test4))

    ### Try regression analysis 
    linear = testDF.copy()

    linear.columns = ['MLBGames', 'NBAGames', 'NFLGames', 'Outcomes', 'TotalGames', 'GamesBin']

    ### Make the 5 potentially interesting linear models for each combination
    lm = smf.ols(formula = 'Outcomes~TotalGames', data = linear).fit() 
    lm1 = smf.ols(formula = 'Outcomes~MLBGames', data = linear).fit()
    lm2 = smf.ols(formula = 'Outcomes~NBAGames', data = linear).fit()
    lm3 = smf.ols(formula = 'Outcomes~NFLGames', data = linear).fit()
    lm4 = smf.ols(formula = 'Outcomes~NFLGames+NBAGames+MLBGames', data = linear).fit()
    
    ### Get the linear model summaries
    print(lm.summary()) 
    print(lm1.summary())
    print(lm2.summary()) 
    print(lm3.summary()) 
    print(lm4.summary())
    
    
Hyp3('NASDAQ Percent Change', 'Total Games')
#Hyp3('NASDAQ Percent Change', 'NBA Games')
#Hyp3('NASDAQ Percent Change', 'NFL Games')
#Hyp3('NASDAQ Percent Change', 'MLB Games')
    

