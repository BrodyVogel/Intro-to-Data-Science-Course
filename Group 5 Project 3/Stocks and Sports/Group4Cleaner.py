#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:19:48 2017

@author: Brody Vogel, Jinghao Yan, Ye Zhang, and Zihao Li
"""

### This script cleans the sports and stock data and outputs a metric that scores the 
### ORIGINAL datasets based on cleanliness

### Import statements
import pandas as pd
import string

### Get the csvs as dataframes
SP = pd.read_csv('Sports_Consolidated_RAW.csv')
ST = pd.read_csv('STOCKS_RAW.csv')

### Instantiate the issues counter for both datasets
Problems = 0
Problems1 = 0

Total = len(SP.columns) * len(SP.index)
Total1 = len(ST.columns) * len(ST.index)

####################
#####Stocks#########
####################

### This is part of a shortcut for our metric calculation
badVols = ['FRANKLIN GOLD AND_PRECIOUS METALS Volume', 'FIRST EAGLE GOLD Volume', 'VICEX (SIN) Volume', 'RYDEX (SIN) Volume', 'UTILITIES Volume', 'BONDS Volume']
Problems2 = 0
for bad in badVols:
    for z in ST.index:
            if isinstance(ST.loc[z, bad], float) == False:
                ### This will come in handy during the metric calculation
                Problems2 += 1

### Check all the columns that could be troublesome for problem data
for col in ST.columns[2:]:         
    for x in ST.index:
        ### A lot of columns have missing values for various New York Stock Exchange-related reasons.
        ### These turned up in the dataset as '-'
        if ST.loc[x, col] == '-':
            Problems1 += 1
            ### Change the missing data to 'nan's 
            ST.loc[x, col] = float('nan')
    ### Check the columns that have numeric values to be sure each entry is, in fact, numeric
    if 'Open' in col or 'High' in col or 'Low' in col or 'Close' in col or 'Volume' in col:
       for x in ST.index:
           if 'nan' in str(ST.loc[x, col]):
               z = 'z'
           ### The data includes commas, so we must remove those while checking for problem data    
           elif ',' in str(ST.loc[x, col]):
               if type(eval(str(ST.loc[x, col].replace(',' , '')))) != float and type(eval(str(ST.loc[x, col].replace(',' , '')))) != int:
                   Problems1 += 1
                   print('Problem: ', ST.iloc[x, 1], ST.loc[x, col])
           else:
               if type(eval(str(ST.loc[x, col]))) != float and type(eval(str(ST.loc[x, col]))) != int:
                   Problems1 += 1
                   print('Problem: ', ST.iloc[x, 1], ST.loc[x, col])

### Some stocks don't have volume data, and so it makes no sense to keep those columns
for bad in badVols:    
    del(ST[bad])
    

######################
#######SPORTS#########
######################

### Define the function to help deal with teams that have moved
def link(team1, team2):
    for x in SP.index:
        if pd.isnull(SP.loc[x, team2]) == False:
            SP.loc[x, team1] = SP.loc[x, team2]
        
### Create lists of the teams that have moved and their current names/locations
SamesNFL = [['Indianapolis Colts', 'Baltimore Colts'], ['Arizona Cardinals', 'St. Louis Cardinals (Football)'], ['Arizona Cardinals', 'Phoenix Cardinals'],
         ['Los Angeles Rams', 'St. Louis Rams'], ['Los Angeles Chargers', 'San Diego Chargers'], ['Tennessee Titans', 'Tennessee Oilers'],
         ['Tennessee Titans', 'Houston Oilers'], ['Oakland Raiders', 'Los Angeles Raiders']]
        
SamesMLB = [['LA Angels of Anaheim', 'California Angels'], ['Miami Marlins', 'Florida Marlins'],
            ['Tampa Bay Rays', 'Tampa Bay Devil Rays'], ['LA Angels of Anaheim', 'Los Angeles Angels']]

SamesNBA = [['Los Angeles Clippers', 'Buffalo Braves'], ['Sacramento Kings', 'Kansas City-Omaha Kings'],
            ['Washington Wizards', 'Washington Bullets'], ['Oklahoma City Thunder', 'Seattle SuperSonics'],
            ['Utah Jazz', 'New Orleans Jazz'], ['Sacramento Kings', 'Kansas City Kings'], ['Brooklyn Nets', 'New York Nets'],
            ['Brooklyn Nets', 'New Jersey Nets'], ['Los Angeles Clippers', 'San Diego Clippers'], ['Charlotte Hornets', 'New Orleans Hornets'],
            ['Memphis Grizzlies', 'Vancouver Grizzlies'], ['Charlotte Hornets', 'Charlotte Bobcats'], ['Charlotte Hornets', 'New Orleans/Oklahoma City Hornets']]

### Merge all the columns of deprecated teams with their new names/locations
for pair in SamesNFL:
    link(pair[0], pair[1])
    del(SP[pair[1]])
    
    
for pair in SamesMLB:
    link(pair[0], pair[1])
    del(SP[pair[1]])  
    
for pair in SamesNBA:
    link(pair[0], pair[1])
    del(SP[pair[1]]) 
    
### One special and sui generis case that was causing problems because of the '/'
SP = SP.replace({'New Orleans/Oklahoma City Hornets': 'Charlotte Hornets'}, regex=True)

### Check all the cells for potential problems
for x in SP.index:
    for y in SP.columns[1:]:
        if pd.isnull(SP.loc[x,y]) == False:
            ### split the cells into their 4 components of interest
            cell = SP.loc[x,y].split('/')
            ### NFL Games can end in ties - this fixes their listing in the dataset
            ### This also makes sure winners never scored fewer points than their losing counterparts
            if cell[0] == 'W' and eval(cell[1]) <= eval(cell[3]):
                Problems += 1
                SP.loc[x,y] = 'Tie/' + cell[1] + '/' + cell[2] + '/' + cell[3]
            if cell[0] == 'L' and eval(cell[1]) >= eval(cell[3]):
                Problems += 1
                SP.loc[x,y] = 'Tie/' + cell[1] + '/' + cell[2] + '/' + cell[3]
            ### if their are more than 4 values in a cell, something has gone wrong
            if len(cell) < 4 or len(cell) > 4:
                Problems += 1
                print('Problem: ', cell)
            ### if either team has a score that is not a number, something has gone wrong
            if isinstance(eval(cell[1]), int) == False or isinstance(eval(cell[3]), int) == False:
                Problems += 1
                print('Problem: ', cell)
            ### if the outcome variable (W, L, or Tie) is not one of those parenthetical things, something has gone wrong
            if cell[0][0] not in string.ascii_lowercase and cell[0][0] not in string.ascii_uppercase:
                Problems += 1
                print('Problem: ', cell)
            ### if the name of the opponent is not a string, something has gone wrong
            if cell[2][0] not in string.ascii_uppercase:
                Problems += 1
                print('Problem: ', cell)
            ### if there are no winners, losers, or ties, something has gone wrong
            if cell[2] in SP.columns:
                if cell[0] == 'W' and SP.loc[x, cell[2]][0] != 'L' and SP.loc[x, cell[2]][0] != 'T':
                    Problems += 1
                    print('Problem: ', SP.iloc[x, ])
            ### opponents shouldn't have 1-letter names
            if len(cell[2]) < 2:
                Problems += 1
                print('Problem: ', cell)

print('The sports dataframe, before cleaning, was:', (Total-Problems)/Total, 'percent clean. That means', (Total-Problems)/Total, 'percent of the data was correct.')
print('After cleaning, the sports dataframe is:', (Total-Problems)/(Total-Problems), 'percent clean. That means', (Total - Problems)/(Total - Problems), 'percent of the data is correct, by our measure.')

print('The stocks dtaframe, before cleaning, was:', (Total1-Problems1)/Total1, 'percent clean, using the same metric as that for the sports data.')
print('The stocks dataframe, after cleaning, is:', (Total1 - Problems2 - (Problems1 - Problems2))/(Total1 - Problems2), 'percent correct. The nas are stil being counted as problematic.')            
    
### create variables for all the teams left in the dataset  
#NFLTeams = ['Washington Redskins', 'Pittsburgh Steelers', 'Cincinnati Bengals',
#       'New York Jets','Chicago Bears', 'Los Angeles Rams', 'Minnesota Vikings',
#       'Miami Dolphins','Oakland Raiders', 'Dallas Cowboys', 
#       'New England Patriots','Kansas City Chiefs', 'Detroit Lions', 'San Francisco 49ers',
#       'Green Bay Packers', 'Denver Broncos', 'New York Giants',
#       'Buffalo Bills', 'Atlanta Falcons', 'Cleveland Browns',
#       'Philadelphia Eagles', 'New Orleans Saints', 'Seattle Seahawks',
#       'Tampa Bay Buccaneers', 'Indianapolis Colts',
#       'Arizona Cardinals',
#       'Carolina Panthers', 'Jacksonville Jaguars', 'Baltimore Ravens',
#       'Tennessee Titans', 'Houston Texans', 'Los Angeles Chargers']

#MLBTeams = ['Houston Astros', 'Boston Red Sox',
#       'Atlanta Braves', 'Oakland Athletics', 'Milwaukee Brewers',
#       'Cincinnati Reds', 'Chicago White Sox', 'Minnesota Twins',
#       'Philadelphia Phillies', 'San Francisco Giants', 'Texas Rangers',
#       'Los Angeles Dodgers', 'St. Louis Cardinals', 'San Diego Padres',
#       'Baltimore Orioles', 'Cleveland Indians', 'Detroit Tigers',
#       'Pittsburgh Pirates', 'Kansas City Royals', 'Chicago Cubs',
#       'Montreal Expos', 'New York Yankees', 'New York Mets',
#       'Seattle Mariners', 'Toronto Blue Jays', 'Colorado Rockies',
#       "Arizona D'Backs", 'LA Angels of Anaheim', 'Washington Nationals',
#       'Tampa Bay Rays', 'Miami Marlins']

#NBATeams = ['New York Knicks', 'Los Angeles Lakers',
#       'Philadelphia 76ers', 'Detroit Pistons', 'Boston Celtics',
#       'Golden State Warriors', 'Cleveland Cavaliers',
#       'Portland Trail Blazers', 'Atlanta Hawks', 'Houston Rockets',
#       'Phoenix Suns', 'Chicago Bulls',
#       'Milwaukee Bucks','San Antonio Spurs', 'Denver Nuggets', 'Indiana Pacers',
#       'Utah Jazz',
#       'Dallas Mavericks', 'Los Angeles Clippers', 'Sacramento Kings',
#       'Charlotte Hornets', 'Miami Heat', 'Minnesota Timberwolves',
#       'Orlando Magic', 'Toronto Raptors','Washington Wizards', 'Memphis Grizzlies', 
#       'Oklahoma City Thunder', 'Brooklyn Nets', 'New Orleans Pelicans']


### create our new feature, which counts how many games were played each day
#helperStrings = ['NFL Games', 'MLB Games', 'NBA Games', 'Total Games']

#for phrase in helperStrings:
    #SP[string] = ''


#indexer1 = 0
#for League in [NFLTeams, MLBTeams, NBATeams]:
    #string = helperStrings[indexer1]
    #for date in SP.index:
        #counter = 0
        #for team in League:
            #if pd.isnull(SP.loc[date, team]) == False:
                #counter += 1
        #SP.loc[date, string] = counter/2
    
    #indexer1 += 1
 
               
### Check to make sure the number of daily games are feasible
#for x in SP.index:
    #SP.loc[x, 'Total Games'] = SP.loc[x, 'NFL Games'] + SP.loc[x, 'MLB Games'] + SP.loc[x, 'NBA Games']
    #if SP.loc[x, 'NFL Games'] > 16 or SP.loc[x, 'MLB Games'] > 17 or SP.loc[x, 'NBA Games'] > 15:
        #print(SP.iloc[x, ])
        #Problems += 1

SP.to_csv('SPORTS_FINAL.csv')
ST.to_csv('STOCKS_FINAL.csv')