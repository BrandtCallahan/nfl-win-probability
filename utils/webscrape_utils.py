from datetime import datetime, date

import numpy as np
from unidecode import unidecode

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from logzero import logger
from urllib.request import urlopen

from utils.beautiful_soup_helper import get_soup_from_url

## Create a playoff team sheet/dict

# scrape gamelog data
def read_gamelog(url):

    # gamelog data
    data = pd.read_html(url)

    tm_data = data[0]

    # make sure to get regular season
    if data[1].shape[0] > 7:
        opp_data = data[1]
        tm_data_post = pd.DataFrame()
        opp_data_post = pd.DataFrame()
    else:
        opp_data = data[2]
        tm_data_post = data[1]
        opp_data_post = data[3]

    # df column names
    tm_columns = tm_data.columns.droplevel(0)
    opp_columns = opp_data.columns.droplevel(0)

    if tm_data_post.empty:
        pass
    else:
        tm_data_post.columns = tm_columns
        opp_data_post.columns = opp_columns

    tm_data.columns = tm_columns
    opp_data.columns = opp_columns

    tm_index_mapping = {
        0: "Rank",
        1: "G",
        2: "Week",
        3: "Date",
        4: "Day",
        5: "Location",
        6: "Opp",
        7: "W/L",
        8: "Tm Pts",
        9: "Opp Pts",
        10: "OT",
        11: "Pass Cmp",
        12: "Pass Att",
        13: "Pass Cmp %",
        14: "Pass Yds",
        15: "Pass TD",
        16: "Pass Y/A",
        17: "Pass Adj Y/A",  # (Yds+20*TD-45*Int)/Att
        18: "Pass Rate",
        19: "Sacks",
        20: "Sack Yds",      
        21: "Rush Att",
        22: "Rush Yds",
        23: "Rush Y/A",
        24: "Rush TD",
        25: "Tot Plays",
        26: "Tot Yds",
        27: "Avg Yds",
        28: "FGA",
        29: "FGM",
        30: "XPA",
        31: "XPM",
        32: "Punt Att",
        33: "Punt Yds",
        34: "1st Dwn Pass",  # 1st Downs reached by Passing
        35: "1st Dwn Rush",  # 1st Downs reached by Rushing
        36: "1st Dwn Penalty",  # 1st Downs reached by Opp Penalty
        37: "Tot 1st Dwn", # Total 1st Downs
        38: "3rd Dwn Conv",
        39: "3rd Dwn Att",
        40: "4th Dwn Conv",
        41: "4th Dwn Att",
        42: "Pen",
        43: "Pen Yds",
        44: "Fmbl",
        45: "Int",
        46: 'Tot TO',
        47: 'ToP',
    }
    tm_data.columns = [
        tm_index_mapping.get(i, col) for i, col in enumerate(tm_data.columns)
    ]
    tm_data['Playoffs'] = 0
    # team postseason (if applicable)
    tm_data_post.columns = [
        tm_index_mapping.get(i, col) for i, col in enumerate(tm_data_post.columns)
    ]
    tm_data_post['Playoffs'] = 1
    tm_data = pd.concat([tm_data, tm_data_post])

    opp_index_mapping = {
        0: "Rank",
        1: "G",
        2: "Week",
        3: "Date",
        4: "Day",
        5: "Location",
        6: "Opp",
        7: "W/L",
        8: "Tm Pts",
        9: "Opp Pts",
        10: "OT",
        11: "Opp Pass Cmp",
        12: "Opp Pass Att",
        13: "Opp Pass Cmp %",
        14: "Opp Pass Yds",
        15: "Opp Pass TD",
        16: "Opp Pass Y/A",
        17: "Opp Pass Adj Y/A",  # (Yds+20*TD-45*Int)/Att
        18: "Opp Pass Rate",
        19: "Opp Sacks",
        20: "Opp Sack Yds",
        21: "Opp Rush Att",
        22: "Opp Rush Yds",
        23: "Opp Rush Y/A",
        24: "Opp Rush TD",
        25: "Opp Tot Plays",
        26: "Opp Tot Yds",
        27: "Opp Avg Yds",
        28: "Opp FGA",
        29: "Opp FGM",
        30: "Opp XPA",
        31: "Opp XPM",
        32: "Opp Punt Att",
        33: "Opp Punt Yds",
        34: "Opp 1st Dwn Pass",  # 1st Downs reached by Passing
        35: "Opp 1st Dwn Rush",  # 1st Downs reached by Rushing
        36: "Opp 1st Dwn Penalty",  # 1st Downs reached by Opp Penalty
        37: "Opp Tot 1st Dwn",  # Total 1st Downs
        38: "Opp 3rd Dwn Conv",
        39: "Opp 3rd Dwn Att",
        40: "Opp 4th Dwn Conv",
        41: "Opp 4th Dwn Att",
        42: "Opp Pen",
        43: "Opp Pen Yds",
        44: "Opp Fmbl",
        45: "Opp Int",
        46: "Opp Tot TO",
        47: "Opp ToP",
    }
    opp_data.columns = [opp_index_mapping.get(i, col) for i, col in enumerate(opp_data.columns)]
    opp_data = opp_data.iloc[:, 11:]
    opp_data['Opp Playoffs'] = 0
    # team postseason (if applicable)
    opp_data_post.columns = [
        opp_index_mapping.get(i, col) for i, col in enumerate(opp_data_post.columns)
    ]
    opp_data_post = opp_data_post.iloc[:, 11:]
    opp_data_post["Opp Playoffs"] = 1
    opp_data = pd.concat([opp_data, opp_data_post])

    data_df = pd.concat([tm_data, opp_data], axis=1)
    data_df = data_df[data_df['Opp'].notnull()].reset_index(drop=True)

    return data_df
