#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jinghao Yan, Ye Zhang, Brody Vogel, and Zihao Li
ANLY 501
October 5th, 2017
"""
### This script grabs the results from every game from the MLB, NBA, and NFL since 1975, then
### outputs this information in .csv format


### import statements
import pandas as pd
from urllib.request import urlopen
import re
from datetime import datetime

### Instantiate dataframe
SPORTS = pd.DataFrame()

### Create the dates that'll become the index and format them appropriately
datelist = pd.date_range(pd.datetime(1975, 1, 1), pd.datetime(2017, 9, 29)).tolist()
dates = []
for date in datelist:
   date = datetime.strptime(str(date)[0:10], '%Y-%m-%d').strftime('%-d-%b-%y')
   dates.append(date)

### All the columns we'll need
ALLTEAMS = ['Date', 'Washington Redskins', 'Pittsburgh Steelers', 'Cincinnati Bengals',
       'Baltimore Colts', 'New York Jets', 'St. Louis Cardinals (Football)',
       'Chicago Bears', 'Los Angeles Rams', 'Minnesota Vikings',
       'San Diego Chargers', 'Houston Oilers', 'Miami Dolphins',
       'Oakland Raiders', 'Dallas Cowboys', 'New England Patriots',
       'Kansas City Chiefs', 'Detroit Lions', 'San Francisco 49ers',
       'Green Bay Packers', 'Denver Broncos', 'New York Giants',
       'Buffalo Bills', 'Atlanta Falcons', 'Cleveland Browns',
       'Philadelphia Eagles', 'New Orleans Saints', 'Seattle Seahawks',
       'Tampa Bay Buccaneers', 'Los Angeles Raiders', 'Indianapolis Colts',
       'Phoenix Cardinals', 'Arizona Cardinals', 'St. Louis Rams',
       'Carolina Panthers', 'Jacksonville Jaguars', 'Baltimore Ravens',
       'Tennessee Oilers', 'Tennessee Titans', 'Houston Texans', 'Los Angeles Chargers',
       'California Angels', 'Houston Astros', 'Boston Red Sox',
       'Atlanta Braves', 'Oakland Athletics', 'Milwaukee Brewers',
       'Cincinnati Reds', 'Chicago White Sox', 'Minnesota Twins',
       'Philadelphia Phillies', 'San Francisco Giants', 'Texas Rangers',
       'Los Angeles Dodgers', 'St. Louis Cardinals', 'San Diego Padres',
       'Baltimore Orioles', 'Cleveland Indians', 'Detroit Tigers',
       'Pittsburgh Pirates', 'Kansas City Royals', 'Chicago Cubs',
       'Montreal Expos', 'New York Yankees', 'New York Mets',
       'Seattle Mariners', 'Toronto Blue Jays', 'Colorado Rockies',
       'Florida Marlins', 'Anaheim Angels', 'Tampa Bay Devil Rays',
       "Arizona D'Backs", 'LA Angels of Anaheim', 'Washington Nationals',
       'Tampa Bay Rays', 'Miami Marlins', 'Los Angeles Angels',
       'New York Knicks', 'Buffalo Braves', 'Los Angeles Lakers',
       'Philadelphia 76ers', 'Detroit Pistons', 'Boston Celtics',
       'Golden State Warriors', 'Kansas City-Omaha Kings',
       'Washington Bullets', 'Seattle SuperSonics', 'Cleveland Cavaliers',
       'Portland Trail Blazers', 'Atlanta Hawks', 'Houston Rockets',
       'Phoenix Suns', 'New Orleans Jazz', 'Chicago Bulls',
       'Milwaukee Bucks', 'Kansas City Kings', 'New York Nets',
       'San Antonio Spurs', 'Denver Nuggets', 'Indiana Pacers',
       'New Jersey Nets', 'San Diego Clippers', 'Utah Jazz',
       'Dallas Mavericks', 'Los Angeles Clippers', 'Sacramento Kings',
       'Charlotte Hornets', 'Miami Heat', 'Minnesota Timberwolves',
       'Orlando Magic', 'Vancouver Grizzlies', 'Toronto Raptors',
       'Washington Wizards', 'Memphis Grizzlies', 'New Orleans Hornets',
       'Charlotte Bobcats', 'New Orleans/Oklahoma City Hornets',
       'Oklahoma City Thunder', 'Brooklyn Nets', 'New Orleans Pelicans']

### Instantiate all the columns
for entry in ALLTEAMS:
    SPORTS[entry] = ''

### Re-index for ease
SPORTS['Date'] = dates
SPORTS = SPORTS.set_index('Date')


# Set delimiter for writing out to file
sep = '|'

# Define function to strip once re.findall function finds data
def restrip(x):
    y = str(x).strip("['").strip("']")
    return y



###########################
######## NFL Data #########
###########################

# Filenames for NFL data
NFLFile = 'NFL_data_raw.txt'
timedout_file = 'NFL_timedout.txt'
error_file = 'NFL_error.txt'

# NFL data file, writing the first row
with open(NFLFile, 'w') as file:
    file.write('Winner' + sep + 'Loser' + sep + 'Date' + sep + 'Score' + '\n')

# Any time out connection when using urlopen
with open(timedout_file, 'w') as file:
    file.write('')

# Error file containing errored out URLs (most likely that week does not exist in the season for older seasons)
with open(error_file, 'w') as file:
    file.write('')

# Loop through all weeks for all years from 1975 to the present
for year in range(1975, 2018):
    for week in range(1,22):
        
        ### create the 
        baseurl = "https://www.pro-football-reference.com/years/" + str(year) + "/week_"
        url = baseurl + str(week) + ".htm" 
        try:
            page = urlopen(url)
        except:
            # Write any timed out URLs to file
            with open(timedout_file, 'a') as file:
                file.write(url + '\n')
        data = page.read()
        encoding = page.info().get_content_charset()
        decoded = data.decode(encoding)
    
        # Write any error URLs to file
        if re.search('404 error', decoded):
            with open(error_file, 'a') as file:
                file.write(url + '\n')
        # Split HTML source code based on game summaries
        else:
            lines = re.split('class="game_summary', decoded)
        
            for i in lines[0:len(lines)]:
                # Look for indicator the section contains actual game summaries, not just source code
                if re.search('expanded nohover', i):
                    dct={}
                    # In cases of a tie game (tie games are possible during NFL regular season)
                    if re.search('class="draw"', i):
                        # Split into identifiable pieces
                        strings = re.split('(?<!</a>)</td>', i)
                        for j in strings:
                            # Add both team names to a list value in dictionary and add score
                            if re.search('class="draw"', j):
                                if 'Draw' not in dct:
                                    dct['Draw'] = []
                                if 'Draw' in dct:
                                    dct['Draw'].append(restrip(re.findall('">(.*)</a>', j)))
                                    dct['Scoretemp'] = restrip(re.findall('class="right">(.*)', j))
                            # Find date
                            elif re.search('/boxscores/', j) and re.search('>Final</a>', j):
                                dct['Date'] = restrip(re.findall('/boxscores/([0-9]{8})', j))
                        dct['Winner'] = dct['Draw'][0]
                        # Clarify this is football team due to same city/team names across sports
                        if dct['Winner'] == 'St. Louis Cardinals':
                            dct['Winner'] = 'St. Louis Cardinals (Football)'
                        dct['Loser'] = dct['Draw'][1]
                        if dct['Loser'] == 'St. Louis Cardinals':
                            dct['Loser'] = 'St. Louis Cardinals (Football)'
                        # Set score
                        dct['Score'] = str(dct['Scoretemp']) + '-' + str(dct['Scoretemp'])
                        #print(dct)
                        # Write to file
                        with open(NFLFile, 'a') as file:
                            file.write(dct['Winner'] + sep + dct['Loser'] + sep + dct['Date'] + sep + dct['Score'] +'\n')
                    # Most games have winner and loser and split into identifiable pieces
                    else:
                        strings = re.split('(?<!</a>)</td>', i)
                        for j in strings:
                            # Find losing team and corresponding points
                            if re.search('class="loser"', j):
                                dct['Loser'] = restrip(re.findall('">(.*)</a>', j))
                                dct['Losing_pts'] = restrip(re.findall('class="right">(.*)', j))
                            # Find winning team and corresponding points
                            elif re.search('class="winner"', j):
                                dct['Winner'] = restrip(re.findall('">(.*)</a>', j))
                                dct['Winning_pts'] = restrip(re.findall('class="right">(.*)', j))
                            # Find date
                            elif re.search('/boxscores/', j) and re.search('>Final</a>', j):
                                dct['Date'] = restrip(re.findall('/boxscores/([0-9]{8})', j))
                        dct['Score'] = str(dct['Winning_pts']) + '-' + str(dct['Losing_pts'])
                        # Clarify this is football team due to same city/team names across sports
                        if dct['Winner'] == 'St. Louis Cardinals':
                            dct['Winner'] = 'St. Louis Cardinals (Football)'
                        if dct['Loser'] == 'St. Louis Cardinals':
                            dct['Loser'] = 'St. Louis Cardinals (Football)'
                        #print(dct)
                        # Write to file
                        with open(NFLFile, 'a') as file:
                            file.write(dct['Winner'] + sep + dct['Loser'] + sep + dct['Date'] + sep + dct['Score'] +'\n')




###########################
######## NBA Data #########
###########################

# Filename for NBA data							
NBAFile = 'NBA_data_raw.txt'

with open(NBAFile, 'w') as file:
    file.write('Winner' + sep + 'Loser' + sep + 'Date' + sep + 'Score' + '\n')

# Loop through each NBA month for year range 1975 to current year
for year in range(1975, 2018):
    for m in ('october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june'):
        baseurl = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) +'_games-'
        url = baseurl + m + '.html'
        try:
            page = urlopen(url)
        except:
            continue
        data = page.read()
        encoding = page.info().get_content_charset()
        decoded = data.decode(encoding)
    
        # Split HTML source code based on games
        lines = re.split('date_game', decoded)
    
        for i in lines[0:len(lines)]:
            # Look for indicator the section contains actual game summaries, not just source code
            if re.search('csk', i):
                dct = {}
                # Split into identifiable pieces
                strings = re.split('/td', i)
                # For newer seasons that have a game_start_time attribute
                if re.search('game_start_time', strings[0]):
                    for j in strings:
                        # Find date, visiting team, home team, and their corresponding points
                        if re.search('visitor_team_name', j):
                            dct['Visiting'] = restrip(re.findall('.html">(.*)</a>', j))
                        elif re.search('home_team_name', j):
                            dct['Home'] = restrip(re.findall('.html">(.*)</a>', j))
                        elif re.search('visitor_pts', j):
                            dct['Visit_pts'] = int(restrip(re.findall('>([0-9]+)<', j)))
                        elif re.search('home_pts', j):
                            dct['Home_pts'] = int(restrip(re.findall('>([0-9]+)<', j)))
                        elif re.search('/boxscores/', j) and re.search('csk', j):
                            dct['Date'] = restrip(re.findall('year=([0-9]+)', j)) + restrip(re.findall('month=([0-9]+)&', j)).zfill(2) + restrip(re.findall('day=([0-9]+)&', j)).zfill(2)
                    # Identify winning team, losing team, and score
                    if dct['Home_pts'] > dct['Visit_pts']:
                        dct['Winner'] = dct['Home']
                        dct['Loser'] = dct['Visiting']
                        dct['Score'] = str(dct['Home_pts']) + '-' + str(dct['Visit_pts'])
                    else:
                        dct['Winner'] = dct['Visiting']
                        dct['Loser'] = dct['Home']
                        dct['Score'] = str(dct['Visit_pts']) + '-' + str(dct['Home_pts'])
                    #print(dct)
                    # Write to file
                    with open(NBAFile, 'a') as file:
                        file.write(dct['Winner'] + sep + dct['Loser'] + sep + dct['Date'] + sep + dct['Score'] +'\n')
                # For older seasons
                else:
                    for j in strings:
                        # Find date, visiting team, home team, and their corresponding points
                        if re.search('visitor_team_name', j):
                            dct['Visiting'] = restrip(re.findall('.html">(.*)</a>', j))
                            dct['Date'] = restrip(re.findall('year=([0-9]+)', j)) + restrip(re.findall('month=([0-9]+)&', j)).zfill(2) + restrip(re.findall('day=([0-9]+)&', j)).zfill(2)
                        elif re.search('home_team_name', j):
                            dct['Home'] = restrip(re.findall('.html">(.*)</a>', j))
                        elif re.search('visitor_pts', j):
                            dct['Visit_pts'] = int(restrip(re.findall('>([0-9]+)<', j)))
                        elif re.search('home_pts', j):
                            dct['Home_pts'] = int(restrip(re.findall('>([0-9]+)<', j)))
                    # Identify winning team, losing team, and score
                    if dct['Home_pts'] > dct['Visit_pts']:
                        dct['Winner'] = dct['Home']
                        dct['Loser'] = dct['Visiting']
                        dct['Score'] = str(dct['Home_pts']) + '-' + str(dct['Visit_pts'])
                    else:
                        dct['Winner'] = dct['Visiting']
                        dct['Loser'] = dct['Home']
                        dct['Score'] = str(dct['Visit_pts']) + '-' + str(dct['Home_pts'])
                    #print(dct)
                    # Write to file
                    with open(NBAFile, 'a') as file:
                        file.write(dct['Winner'] + sep + dct['Loser'] + sep + dct['Date'] + sep + dct['Score'] +'\n')


            
###########################
######## MLB Data #########
###########################

# Filename for MLB data				
filename = 'MLB_data_raw.txt'

with open(filename, 'w') as file:
    file.write('Winner' + sep + 'Loser' + sep + 'Date' + sep + 'Score' + '\n')

# Loop through for year range 1975 to current year
for year in range(1975, 2018):
    url = 'https://www.baseball-reference.com/leagues/MLB/' + str(year)+ '-schedule.shtml'
    page = urlopen(url)
    data = page.read()
    encoding = page.info().get_content_charset()
    decoded = data.decode(encoding)

    # Split HTML source code based on game summaries
    lines = re.split('<.*class.*game.*>', decoded)

    for i in lines[0:len(lines)]:
        # Look for indicator the section contains actual game summaries, not just source code
        if re.search('<strong>', i) and re.search('Boxscore', i):
            dct = {}
            # Split into identifiable pieces
            strings = re.split('</.*>', i)
            # For cases where winning team appears before losing team in boxscore
            if re.search('<strong>', strings[0]):
                for j in strings:
                    # Find date, winning team, losing team, and corresponding scores
                    if re.search('<strong>', j) and re.search('/teams/', j):
                        dct['winner'] = j[j.rfind('>')+1:]
                    elif re.search('/teams/', j) and re.search('\.shtml', j):
                        dct['loser'] = j[j.rfind('>')+1:]
                    elif re.search('[0-9]{8}.*\.shtml', j):
                        dct['date'] = restrip(re.findall('[0-9]{8}', j))
                        dct['l_score'] = restrip(re.findall('\n \((.*)\)', j))
                    elif re.search('\n \([0-9]+\)', j):
                        dct['w_score'] = restrip(re.findall('\n \((.*)\)', j))
                dct['score'] = str(dct['w_score']) + '-' + str(dct['l_score'])
                #print(dct)
                # Write to file
                with open(filename, 'a') as file:
                    file.write(dct['winner'] + sep + dct['loser'] + sep + dct['date'] + sep + dct['score'] +'\n')
            # For cases where losing team appears before winning team in boxscore
            else:
                for j in strings:
                    # Find date, winning team, losing team, and corresponding scores
                    if re.search('<strong>', j) and re.search('/teams/', j):
                        dct['winner'] = j[j.rfind('>')+1:]
                        dct['l_score'] = restrip(re.findall('\n \((.*)\)', j))
                    elif re.search('/teams/', j) and re.search('\.shtml', j):
                        dct['loser'] = j[j.rfind('>')+1:]
                    elif re.search('[0-9]{8}.*\.shtml', j):
                        dct['date'] = restrip(re.findall('[0-9]{8}', j))
                    elif re.search('\n \([0-9]+\)', j):
                        dct['w_score'] = restrip(re.findall('\n \((.*)\)', j))
                dct['score'] = str(dct['w_score']) + '-' + str(dct['l_score'])
                #print(dct)
                # Write to file
                with open(filename, 'a') as file:
                    file.write(dct['winner'] + sep + dct['loser'] + sep + dct['date'] + sep + dct['score'] +'\n')

#######################
#######Reformat########
#######################                    

### Get the data into the dataframe, first for the MLB, then NBA, and finally NFL
                    
### count the lines in the file                    
with open(filename, 'r') as file:
    counter = 0
    for line in file:
       counter += 1

### Find the date, winner, loser, winning points, and losing points for every game and add that information
          ### to the dataframe in the appropriate position
with open(filename, 'r') as file:
    file.readline()
    for x in range(counter-1):
        Next = file.readline()
        Next = Next.strip().split('|')
        #print(Next)
        winner = Next[0]
        loser = Next[1]
        date = Next[2][4:6] + '-' + Next[2][6:] + '-' + Next[2][2:4]
        date = str(date)
        date = datetime.strptime(date, '%m-%d-%y').strftime('%-d-%b-%y')
        if date in SPORTS.index:
            WPoints = Next[3].split('-')[0]
            WPoints = str(WPoints)
            LPoints = Next[3].split('-')[1]
            LPoints = str(LPoints)
            SPORTS.loc[date, winner] = ['W', WPoints, loser, LPoints]
            SPORTS.loc[date, loser] = ['L', LPoints, winner, WPoints]


### Next two sections are identical to the previous, with new files
with open(NBAFile, 'r') as file:
    counter = 0
    for line in file:
        counter += 1
with open(NBAFile, 'r') as file:
    file.readline()
    for line in range(counter-1):
        Next = file.readline()
        Next = Next.strip().split('|')
        #print(Next)
        winner = Next[0]
        loser = Next[1]
        date = Next[2][4:6] + '-' + Next[2][6:] + '-' + Next[2][2:4]
        date = str(date)
        date = datetime.strptime(date, '%m-%d-%y').strftime('%-d-%b-%y')
        if date in SPORTS.index:
            WPoints = Next[3].split('-')[0]
            WPoints = str(WPoints)
            LPoints = Next[3].split('-')[1]
            LPoints = str(LPoints)
            SPORTS.loc[date, winner] = ['W', WPoints, loser, LPoints]
            SPORTS.loc[date, loser] = ['L', LPoints, winner, WPoints]
            
with open(NFLFile, 'r') as file:
    counter = 0
    for line in file:
        counter += 1
with open(NFLFile, 'r') as file:
    file.readline()
    for line in range(counter-1):
        Next = file.readline()
        #print(Next)
        Next = Next.strip().split('|')
        winner = Next[0]
        loser = Next[1]
        date = Next[2][4:6] + '-' + Next[2][6:] + '-' + Next[2][2:4]
        date = str(date)
        date = datetime.strptime(date, '%m-%d-%y').strftime('%-d-%b-%y')
        if date in SPORTS.index:
            WPoints = Next[3].split('-')[0]
            WPoints = str(WPoints)
            LPoints = Next[3].split('-')[1]
            LPoints = str(LPoints)
            SPORTS.loc[date, winner] = 'W' + '/' + WPoints + '/' + loser + '/' + LPoints
            SPORTS.loc[date, loser] = 'L' + '/' + LPoints + '/' + winner + '/' + WPoints

### Output the csv
SPORTS.to_csv('Sports_Consolidated_RAW.csv')