import cmath
from datetime import datetime
import os

os.chdir(f"C:/Users/{os.getlogin()}/personal-github/nfl-win-probability")

"""
Python Predictive Model imports
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import rpy2.robjects as robjects

from logzero import logger
from math import sqrt
from tqdm import tqdm
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from utils.webscrape_utils import (
    read_gamelog,
)

from utils.team_dict import get_teamnm

"""
    Team Stats by Game
"""

###############################################################################
"""
    HELPER FUNCTIONS
        - used when pulling down and saving off data
        - also used for data manipulation/transformation
"""


def get_team_stats(season_year, team_url, team_name):
    team_df = get_teamnm()

    conference_dict = {
        "ARI": "NFC West",
        "ATL": "NFC South",
        "BAL": "AFC North",
        "BUF": "AFC East",
        "CAR": "NFC South",
        "CHI": "NFC North",
        "CIN": "AFC North",
        "CLE": "AFC North",
        "DAL": "NFC East",
        "DEN": "AFC West",
        "DET": "NFC North",
        "GNB": "NFC North",
        "HOU": "AFC South",
        "IND": "AFC South",
        "JAX": "AFC South",
        "KAN": "AFC West",
        "LAC": "AFC West",
        "LAR": "NFC West",
        "LVR": "AFC West",
        "MIA": "AFC East",
        "MIN": "NFC North",
        "NWE": "AFC East",
        "NOR": "NFC South",
        "NYG": "NFC East",
        "NYJ": "AFC East",
        "PHI": "NFC East",
        "PIT": "AFC North",
        "SEA": "NFC West",
        "SFO": "NFC West",
        "TAM": "NFC South",
        "TEN": "AFC South",
        "WAS": "NFC East",
        "OAK": "AFC West",
        "STL": "NFC West",
        "SDG": "AFC West",
    }

    team_df = team_df[(team_df["Gamelog Name"] == team_url)]
    url = (
        f"https://www.pro-football-reference.com/teams/{team_url}/{season_year}/gamelog"
    )
    # check out boxscores for player game by game stats

    # team gamelog
    team_gamelog = read_gamelog(url)
    team_gamelog = team_gamelog[
        ~(team_gamelog["Opp"].isin(["", "Opponent", "Opp"]))
        & ~(team_gamelog["W/L"].isin(["", np.nan, pd.NA]))
        & ~(team_gamelog["G"].isin(["", np.nan, pd.NA]))
    ].reset_index(drop=True)

    # drop games with no data
    team_gamelog = team_gamelog[
        (team_gamelog["Date"].astype("datetime64[ns]"))
        < datetime.now().strftime("%Y-%m-%d")
    ].reset_index(drop=True)

    # label team
    team_gamelog["Tm"] = team_name

    # label team conference/division
    team_gamelog["Tm Div"] = conference_dict[f"{team_name}"]

    # label opp conference
    team_gamelog["Opp Div"] = np.nan
    team_gamelog["Opp Div"] = team_gamelog["Opp Div"].astype(str)
    for n, opp in enumerate(team_gamelog["Opp"]):
        try:
            team_gamelog.loc[n, "Opp Div"] = conference_dict[f"{opp}"]
            # team_gamelog['Opp Conference'][n] = [conference_dict[f'{opp}']]
        except:
            team_gamelog.loc[n, "Opp Div"] = np.nan

    team_gamelog.loc[team_gamelog["Tm Div"] == team_gamelog["Opp Div"], "Div"] = 1
    team_gamelog["Div"] = team_gamelog["Div"].fillna(0)

    # replace team movement
    team_gamelog["Opp"] = (
        team_gamelog["Opp"]
        .replace("SDG", "LAC")
        .replace("STL", "LAR")
        .replace("OAK", "LVR")
    )

    return team_gamelog


def tm_elo_rating(season_year, today):

    # read out gamelog
    team_gamelog = pd.read_csv(
        f"~/personal-github/nfl-win-probability/csv_files/{season_year}/season{season_year}_tm_gamelogs.csv"
    )
    # only games less than "today"
    team_gamelog = team_gamelog[
        (team_gamelog["Date"].astype("datetime64[ns]") <= pd.to_datetime(today))
    ].reset_index(drop=True)

    # set initial Elo rating
    team_gamelog.loc[team_gamelog["Week"] == 1, "Tm Elo"] = 1500
    team_gamelog.loc[team_gamelog["Week"] == 1, "Opp Elo"] = 1500

    # K factor
    K = 100

    # list of possible weeks
    week_list = team_gamelog["Week"].unique().tolist()
    week_list.sort()
    week_list = week_list[:-1]

    team_elo = pd.DataFrame()
    for g in week_list:

        tmp_gm = team_gamelog[team_gamelog["Week"] == g].reset_index(drop=True)

        tmp_gm.loc[(tmp_gm["Location"].isna()), "hm_field"] = 50
        tmp_gm["hm_field"] = tmp_gm["hm_field"].fillna(0)

        tmp_gm.loc[tmp_gm["W/L"] == "L", "score_neg"] = -1
        tmp_gm["score_neg"] = tmp_gm["score_neg"].fillna(1)

        tmp_gm["elo_diff"] = tmp_gm["Tm Elo"] - tmp_gm["Opp Elo"]
        tmp_gm["elo_margin"] = (
            tmp_gm["score_neg"]
            * (abs(tmp_gm["Tm Pts"] - tmp_gm["Opp Pts"] + tmp_gm["hm_field"]) ** 0.8)
        ) / (7.5 + (0.006 * (tmp_gm["elo_diff"])))

        tmp_gm.loc[tmp_gm["W/L"] == "W", "actual"] = 1
        tmp_gm["actual"] = tmp_gm["actual"].fillna(0)

        tmp_gm["tm_elo_adj"] = 1 / (
            1 + 10 ** ((tmp_gm["Opp Elo"] - tmp_gm["Tm Elo"]) / 400)
        )
        tmp_gm["tm_elo"] = tmp_gm["Tm Elo"] + K * (
            (tmp_gm["actual"] * 1) - (tmp_gm["tm_elo_adj"])
        )

        tmp_df = (
            tmp_gm[["Tm", "tm_elo"]]
            .reset_index(drop=True)
            .rename(columns={"tm_elo": "Tm Elo2"})
        )

        # accomodate for bye week
        if team_elo.empty:
            team_elo = tmp_df.copy()
        else:
            team_elo = pd.concat([team_elo, tmp_df]).reset_index(drop=True)

        team_elo = team_elo.drop_duplicates(subset=["Tm"], keep="last")

        if (season_year == 2017) & (g == 1):
            team_elo = pd.concat(
                [
                    team_elo,
                    pd.DataFrame(data={"Tm": ["TAM", "MIA"], "Tm Elo2": [1500, 1500]}),
                ]
            )

        tmp_gamelog = team_gamelog[team_gamelog["Week"] == (g + 1)]

        tmp_gamelog = tmp_gamelog.merge(team_elo, how="left", on=["Tm"])
        tmp_gamelog["Tm Elo"] = tmp_gamelog["Tm Elo"].fillna(tmp_gamelog["Tm Elo2"])
        tmp_gamelog = tmp_gamelog.drop(columns=["Tm Elo2"])

        # put the new opponent elo in for the next week
        tmp_gamelog = (
            tmp_gamelog.merge(team_elo, how="left", left_on=["Opp"], right_on=["Tm"])
            .drop(columns={"Tm_y"})
            .rename(columns={"Tm_x": "Tm"})
        )
        tmp_gamelog["Opp Elo"] = tmp_gamelog["Opp Elo"].fillna(tmp_gamelog["Tm Elo2"])
        tmp_gamelog = tmp_gamelog.drop(columns=["Tm Elo2"])

        team_gamelog = (
            pd.concat([team_gamelog, tmp_gamelog])
            .drop_duplicates(subset=["Tm", "Opp", "Week", "Date"], keep="last")
            .sort_values(by=["Tm", "G", "Week"])
            .reset_index(drop=True)
        )

        team_gamelog["Date"] = pd.to_datetime(team_gamelog["Date"])
        team_gamelog = team_gamelog.drop_duplicates(["Tm", "Opp", "Date"]).reset_index(
            drop=True
        )

    elo_df = team_gamelog[["Week", "Date", "Tm", "Tm Elo", "Opp", "Opp Elo"]]

    return elo_df


def tm_lg_ranking(season_year, today):

    # read out gamelog
    team_gamelog = pd.read_csv(
        f"~/personal-github/nfl-win-probability/csv_files/{season_year}/season{season_year}_tm_gamelogs.csv"
    )
    # only games less than "today"
    team_gamelog = team_gamelog[
        (team_gamelog["Date"].astype("datetime64[ns]") <= pd.to_datetime(today))
    ].reset_index(drop=True)

    # set initial lg rankings
    team_gamelog.loc[team_gamelog["Week"] == 1, "Tm Rnk"] = 1
    team_gamelog.loc[team_gamelog["Week"] == 1, "Opp Rnk"] = 1

    team_gamelog.loc[team_gamelog["Week"] == 1, "Tm W"] = 0
    team_gamelog.loc[team_gamelog["Week"] == 1, "Opp W"] = 0

    # list of possible weeks
    week_list = team_gamelog["Week"].unique().tolist()
    week_list.sort()
    week_list = week_list[:-1]

    team_rnk = pd.DataFrame()
    for g in week_list:

        tmp_gm = team_gamelog[team_gamelog["Week"] == g].reset_index(drop=True)

        # Tm Win/Loss
        tmp_gm.loc[tmp_gm["W/L"] == "W", "Tm W"] = (
            tmp_gm.loc[tmp_gm["W/L"] == "W"]["Tm W"] + 1
        )
        tmp_gm.loc[tmp_gm["W/L"] == "L", "Tm W"] = tmp_gm.loc[tmp_gm["W/L"] == "L"][
            "Tm W"
        ]

        # Opp Win/Loss
        tmp_gm.loc[tmp_gm["W/L"] == "W", "Opp W"] = tmp_gm.loc[tmp_gm["W/L"] == "W"][
            "Opp W"
        ]
        tmp_gm.loc[tmp_gm["W/L"] == "L", "Opp W"] = (
            tmp_gm.loc[tmp_gm["W/L"] == "L"]["Opp W"] + 1
        )

        # Tm W%
        tmp_gm["W%"] = tmp_gm["Tm W"] / tmp_gm["G"]

        tmp_df = (
            tmp_gm[["Tm", "Tm W", "W%"]]
            .reset_index(drop=True)
            .rename(
                columns={
                    "W%": "W% 2",
                    "Tm W": "Tm W 2",
                }
            )
        )
        tmp_df["Lg Rnk"] = tmp_df["W% 2"].rank(method="min", ascending=False)
        tmp_df = tmp_df.drop(columns=["W% 2"])

        # accomodate for bye week
        if team_rnk.empty:
            team_rnk = tmp_df.copy()
        else:
            team_rnk = pd.concat([team_rnk, tmp_df]).reset_index(drop=True)

        team_rnk = team_rnk.drop_duplicates(subset=["Tm"], keep="last")

        # Tampa Bay and Miami had a Week 1 bye in 2017
        if (season_year == 2017) & (g == 1):
            team_rnk = pd.concat(
                [
                    team_rnk,
                    pd.DataFrame(
                        data={"Tm": ["TAM", "MIA"], "Tm W 2": [0, 0], "Lg Rnk": [1, 1]}
                    ),
                ]
            )

        tmp_gamelog = team_gamelog[team_gamelog["Week"] == (g + 1)]

        tmp_gamelog = tmp_gamelog.merge(team_rnk, how="left", on=["Tm"])
        tmp_gamelog["Tm W"] = tmp_gamelog["Tm W"].fillna(tmp_gamelog["Tm W 2"])
        tmp_gamelog = tmp_gamelog.drop(columns=["Tm W 2"])
        tmp_gamelog["Tm Rnk"] = tmp_gamelog["Tm Rnk"].fillna(tmp_gamelog["Lg Rnk"])
        tmp_gamelog = tmp_gamelog.drop(columns=["Lg Rnk"])

        # put the new opponent elo in for the next week
        tmp_gamelog = (
            tmp_gamelog.merge(team_rnk, how="left", left_on=["Opp"], right_on=["Tm"])
            .drop(columns={"Tm_y"})
            .rename(columns={"Tm_x": "Tm"})
        )
        tmp_gamelog["Opp W"] = tmp_gamelog["Opp W"].fillna(tmp_gamelog["Tm W 2"])
        tmp_gamelog = tmp_gamelog.drop(columns=["Tm W 2"])
        tmp_gamelog["Opp Rnk"] = tmp_gamelog["Opp Rnk"].fillna(tmp_gamelog["Lg Rnk"])
        tmp_gamelog = tmp_gamelog.drop(columns=["Lg Rnk"])

        team_gamelog = (
            pd.concat([team_gamelog, tmp_gamelog])
            .drop_duplicates(subset=["Tm", "Opp", "Week", "Date"], keep="last")
            .sort_values(by=["Tm", "G", "Week"])
            .reset_index(drop=True)
        )

        team_gamelog["Date"] = pd.to_datetime(team_gamelog["Date"])
        team_gamelog = team_gamelog.drop_duplicates(["Tm", "Opp", "Date"]).reset_index(
            drop=True
        )

    rnk_df = team_gamelog[["Week", "Date", "Tm", "Tm Rnk", "Opp", "Opp Rnk"]]

    return rnk_df


def nfl_odds(season_year):
    csv_df = pd.read_csv(
        f"C:/Users/{os.getlogin()}/personal-github/nfl-win-probability/csv_files/nfl_game_results.csv",
    )
    csv_df = csv_df.astype(
        {
            "gameday": "datetime64[ns]",
        }
    )

    # map team abbreviations
    fastr_dict = {
        "ARI": "ARI",
        "ATL": "ATL",
        "BAL": "BAL",
        "BUF": "BUF",
        "CAR": "CAR",
        "CHI": "CHI",
        "CIN": "CIN",
        "CLE": "CLE",
        "DAL": "DAL",
        "DEN": "DEN",
        "DET": "DET",
        "GB": "GNB",
        "HOU": "HOU",
        "IND": "IND",
        "JAX": "JAX",
        "KC": "KAN",
        "LA": "LAR",
        "LAC": "LAC",
        "LV": "LVR",
        "MIA": "MIA",
        "MIN": "MIN",
        "NE": "NWE",
        "NO": "NOR",
        "NYG": "NYG",
        "NYJ": "NYJ",
        "OAK": "LVR",
        "PHI": "PHI",
        "PIT": "PIT",
        "SD": "LAC",
        "SEA": "SEA",
        "SF": "SFO",
        "STL": "LAR",
        "TB": "TAM",
        "TEN": "TEN",
        "WAS": "WAS",
    }

    csv_df["away_team"] = csv_df["away_team"].map(fastr_dict)
    csv_df["home_team"] = csv_df["home_team"].map(fastr_dict)

    csv_df.loc[:, "matchup"] = (
        csv_df.loc[:, "away_team"] + " vs. " + csv_df.loc[:, "home_team"]
    )

    # limit to 2013 (for now)
    nfl_df = csv_df[(csv_df["season"].astype("int16") == season_year)]

    return nfl_df


def gamelog_setup(season, tm_name, gm_date):

    # make sure 'gm_date' is a date and not string
    gm_date = pd.to_datetime(gm_date)

    team_df = get_teamnm()
    tm_df = team_df[team_df["Tm Abbrv"] == tm_name].reset_index(drop=True)

    # pull data from csv_files folder
    gamelog = pd.read_csv(
        f"~/personal-github/nfl-win-probability/csv_files/{season}/season{season}_tm_gamelogs.csv",
    )
    team_gamelog = gamelog[gamelog["Tm"] == tm_df["Tm Abbrv"][0]]

    team_gamelog = team_gamelog[
        (
            ~(team_gamelog["Opp"].isin(["", "Opponent", "Opp"]))
            & ~(team_gamelog["W/L"].isin(["", np.nan, pd.NA]))
        )
    ].reset_index(drop=True)

    # add season column
    team_gamelog["Season"] = season

    """
        Building SOS and SOR metrics
        - SOS: looking at your opponents records (sum wins for all opponents divide by total games)
        - SOS_2: looking at your opponents opponents records (2*OR+OOR / 3)
    """
    # SOS (Opponent Win %) (at time of game)
    opp_sos_df = pd.DataFrame()
    for week in range(len(team_gamelog)):
        opponent = team_gamelog["Opp"][week]
        gm_day = team_gamelog["Date"][week]

        # read in opponent data
        log = pd.read_csv(
            f"~/personal-github/nfl-win-probability/csv_files/{season}/season{season}_tm_gamelogs.csv",
        )
        opp_log = log[log["Tm"] == opponent]

        opp_log = opp_log[
            (
                ~(opp_log["Opp"].isin(["", "Opponent", "Opp"]))
                & ~(opp_log["W/L"].isin(["", np.nan, pd.NA]))
            )
        ].reset_index(drop=True)

        opp_log = opp_log[opp_log["Date"] < gm_day]

        try:
            # friendly win/loss
            opp_log.loc[opp_log["W/L"].str.contains("W"), "W/L Flag"] = 1
            opp_log["W/L Flag"] = opp_log["W/L Flag"].fillna(0)

            opp_log["W/L"] = opp_log["W/L"].str[0]

            # running sum W
            opp_log["Opp W"] = opp_log.groupby(["Tm"])["W/L Flag"].cumsum()
            opp_log["Opp W%"] = opp_log["Opp W"] / opp_log["G"].astype(int)

            opp_log["Gm Date"] = gm_day

            if opp_sos_df.empty:
                opp_sos_df = opp_log.iloc[opp_log.index.max() :, :][
                    ["Gm Date", "Tm", "Opp W%"]
                ].rename(columns={"Tm": "Opp", "Gm Date": "Date"})
            else:
                opp_sos_df = pd.concat(
                    [
                        opp_sos_df,
                        opp_log.iloc[opp_log.index.max() :, :][
                            ["Gm Date", "Tm", "Opp W%"]
                        ].rename(columns={"Tm": "Opp", "Gm Date": "Date"}),
                    ]
                ).reset_index(drop=True)
        except ValueError:
            pass

    opp_sos_df["Tm"] = tm_name

    opp_sos = pd.DataFrame()
    for i in range(len(opp_sos_df)):
        opp_df = (
            opp_sos_df.iloc[: i + 1]
            .groupby(["Tm"])
            .agg(
                last_gm_date=("Date", "last"),
                opp_sos=("Opp W%", "mean"),
                Opp=("Opp", "last"),
            )
            .reset_index()
        )

        if opp_sos.empty:
            opp_sos = opp_df[["Tm", "last_gm_date", "Opp", "opp_sos"]].rename(
                columns={"last_gm_date": "Date", "opp_sos": "Opp SOS"}
            )
        else:
            opp_sos = pd.concat(
                [
                    opp_sos,
                    opp_df[["Tm", "last_gm_date", "Opp", "opp_sos"]].rename(
                        columns={"last_gm_date": "Date", "opp_sos": "Opp SOS"}
                    ),
                ]
            ).reset_index(drop=True)

    # merge in Opp SOS
    team_gamelog = team_gamelog.merge(opp_sos, how="left", on=["Tm", "Date", "Opp"])
    team_gamelog["Opp SOS"] = team_gamelog["Opp SOS"].fillna(0.5)

    # 3rd Down Conversion
    team_gamelog["3Dwn Conv %"] = (
        team_gamelog["3rd Dwn Conv"] / team_gamelog["3rd Dwn Att"]
    )
    team_gamelog["Opp 3Dwn Conv %"] = (
        team_gamelog["Opp 3rd Dwn Conv"] / team_gamelog["Opp 3rd Dwn Att"]
    )

    # FG%
    team_gamelog["FG%"] = team_gamelog["FGM"] / team_gamelog["FGA"]
    team_gamelog["Opp FG%"] = team_gamelog["Opp FGM"] / team_gamelog["Opp FGA"]

    # TO +/-
    team_gamelog["TO +/-"] = team_gamelog["Opp Tot TO"] - team_gamelog["Tot TO"]

    # friendly win/loss
    team_gamelog.loc[team_gamelog["W/L"].str.contains("W"), "W/L Flag"] = 1
    team_gamelog["W/L Flag"] = team_gamelog["W/L Flag"].fillna(0)

    team_gamelog["W/L"] = team_gamelog["W/L"].str[0]

    # running sum W
    team_gamelog["Total W"] = team_gamelog.groupby(["Tm"])["W/L Flag"].cumsum()
    team_gamelog["Total W%"] = team_gamelog["Total W"] / team_gamelog["G"].astype(int)

    # streaks
    team_gamelog.loc[(team_gamelog["W/L"] == "L"), "Streak Value"] = -1
    team_gamelog["Streak Value"] = team_gamelog["Streak Value"].fillna(1)

    team_gamelog["Start Streak"] = team_gamelog["Streak Value"].ne(
        team_gamelog["Streak Value"].shift()
    )
    team_gamelog["Streak Id"] = team_gamelog["Start Streak"].cumsum()
    team_gamelog["Running Streak"] = team_gamelog.groupby("Streak Id").cumcount() + 1

    # W Streak == +, L Streak == -, Tie == 0
    team_gamelog.loc[team_gamelog["W/L"] == "W", "Streak +/-"] = team_gamelog[
        "Running Streak"
    ]
    team_gamelog.loc[team_gamelog["W/L"] == "L", "Streak +/-"] = (
        team_gamelog["Running Streak"] * -1
    )
    team_gamelog.loc[team_gamelog["W/L"] == "T", "Streak +/-"] = 0
    team_gamelog["Streak +/-"] = team_gamelog["Streak +/-"].astype(int)
    team_gamelog = team_gamelog.drop(
        columns=["Streak Value", "Start Streak", "Streak Id", "Running Streak"]
    )

    # possessions
    team_gamelog["Poss"] = (
        team_gamelog["Tot TO"].astype(int)
        + team_gamelog["Punt Att"].astype(int)
        + team_gamelog["Pass TD"].astype(int)
        + team_gamelog["Rush TD"].astype(int)
        + team_gamelog["FGA"].astype(int)
        + (
            team_gamelog["4th Dwn Att"].astype(int)
            - team_gamelog["4th Dwn Conv"].astype(int)
        )
    )
    team_gamelog["Opp Poss"] = (
        team_gamelog["Opp Tot TO"].astype(int)
        + team_gamelog["Opp Punt Att"].astype(int)
        + team_gamelog["Opp Pass TD"].astype(int)
        + team_gamelog["Opp Rush TD"].astype(int)
        + team_gamelog["Opp FGA"].astype(int)
        + (
            team_gamelog["Opp 4th Dwn Att"].astype(int)
            - team_gamelog["Opp 4th Dwn Conv"].astype(int)
        )
    )

    # offense ratings
    team_gamelog["Pass Off Eff"] = team_gamelog["Pass Rate"].astype(float)
    team_gamelog["Rush Off Eff"] = team_gamelog["Rush Y/A"].astype(float)
    # compare to college avg. (passer rating avg. is 100)
    team_gamelog["Adj Pass Off Eff"] = team_gamelog["Pass Off Eff"] - 100
    team_gamelog["Adj Rush Off Eff"] = (
        team_gamelog["Rush Off Eff"] - gamelog["Rush Y/A"].median()
    )
    # offense efficency
    # defined as Points per Possession
    team_gamelog["Off Eff"] = team_gamelog["Tm Pts"] / team_gamelog["Poss"]

    # defense ratings
    team_gamelog["Pass Def Eff"] = team_gamelog["Opp Pass Rate"].astype(float)
    team_gamelog["Rush Def Eff"] = team_gamelog["Opp Rush Y/A"].astype(float)
    # compare to college avg.
    team_gamelog["Adj Pass Def Eff"] = team_gamelog["Pass Def Eff"] - 100
    team_gamelog["Adj Rush Def Eff"] = (
        team_gamelog["Rush Def Eff"] - gamelog["Opp Rush Y/A"].median()
    )
    # defense efficency
    team_gamelog["Def Eff"] = team_gamelog["Opp Pts"] / team_gamelog["Opp Poss"]

    # tm effieciency rating
    team_gamelog["Tm Eff"] = team_gamelog["Off Eff"] - team_gamelog["Def Eff"]

    # margin of victory
    team_gamelog["Margin Victory"] = team_gamelog["Tm Pts"].astype(int) - team_gamelog[
        "Opp Pts"
    ].astype(int)

    # game luck
    team_gamelog["Tm Luck"] = (
        (team_gamelog["Tm Pts"] / team_gamelog["Opp Pts"].replace(0, 1)) ** 2.37
    ) / (((team_gamelog["Tm Pts"] / team_gamelog["Opp Pts"].replace(0, 1)) ** 2.37) + 1)

    # flag conference games (regular season)
    team_gamelog.loc[
        (team_gamelog["Tm Div"] == team_gamelog["Opp Div"]), "Div Game"
    ] = 1
    team_gamelog["Div Game"] = team_gamelog["Div Game"].fillna(0)

    # rename Home/Away/Neutral to 2/1/0
    location_dict = {
        "@": 1,
        "N": 0,
    }
    team_gamelog["Location"] = (team_gamelog["Location"].map(location_dict)).fillna(2)
    team_gamelog["Location"] = team_gamelog["Location"].astype(int)

    # limit data to only show up to defined date (not including)
    team_gamelog = team_gamelog[
        (team_gamelog["Date"].astype("datetime64[ns]") < gm_date)
    ].reset_index(drop=True)

    return team_gamelog


def rolling_gamedata(season, hm_tm, aw_tm, gm_date):
    # pull both team's gamelogs up to game
    hm_gm_log = gamelog_setup(season, hm_tm, gm_date)
    aw_gm_log = gamelog_setup(season, aw_tm, gm_date)

    gm_log = pd.concat([hm_gm_log, aw_gm_log])

    # calculate scoring plays (Pass TD, Rush TD, FGM)
    gm_log["ScoringPlays"] = gm_log["Pass TD"] + gm_log["Rush TD"] + gm_log["FGM"]
    gm_log["OppScoringPlays"] = (
        gm_log["Opp Pass TD"] + gm_log["Opp Rush TD"] + gm_log["Opp FGM"]
    )

    # group gamelog stats (season avg.)
    gm_df = (
        gm_log.groupby(["Tm"], observed=True)
        .agg(
            G=("G", "last"),
            W=("W/L Flag", "sum"),
            Wpct=("Total W%", "last"),
            Poss=("Poss", "mean"),
            TotPoss=("Poss", "sum"),
            OppPoss=("Opp Poss", "mean"),
            TotOppPoss=("Opp Poss", "sum"),
            PassOffEff=("Adj Pass Off Eff", "mean"),
            RushOffEff=("Adj Rush Off Eff", "mean"),
            PassDefEff=("Adj Pass Def Eff", "mean"),
            RushDefEff=("Adj Rush Def Eff", "mean"),
            TmOffEff=("Off Eff", "mean"),
            TmDefEff=("Def Eff", "mean"),
            TmEff=("Tm Eff", "mean"),
            TmLuckW=("Tm Luck", "sum"),
            Pts=("Tm Pts", "mean"),
            TotPts=("Tm Pts", "sum"),
            TotScorePlays=("ScoringPlays", "sum"),
            OppPts=("Opp Pts", "mean"),
            TotOppPts=("Opp Pts", "sum"),
            TotOppScorePlays=("OppScoringPlays", "sum"),
            TmDiv=("Tm Div", "first"),
            WStreak=("Streak +/-", "last"),
            AvgMoV=("Margin Victory", "mean"),
            TotPtDiff=("Margin Victory", "sum"),
            OppSOS=("Opp SOS", "last"),
            # start boxscore stats
            PassCmppct=("Pass Cmp %", "mean"),
            PassAdjYdsAtt=("Pass Adj Y/A", "mean"),
            RushYdsAtt=("Rush Y/A", "mean"),
            PassTDG=("Pass TD", "mean"),
            RushTDG=("Rush TD", "mean"),
            FGpct=("FG%", "mean"),
            PenYdsG=("Pen Yds", "mean"),
            TOVG=("Tot TO", "mean"),
            TO=("TO +/-", "sum"),
            Dwn3Conv=("3Dwn Conv %", "mean"),
            # opponent boxscore stats
            OppPassCmppct=("Opp Pass Cmp %", "mean"),
            OppPassAdjYdsAtt=("Opp Pass Adj Y/A", "mean"),
            OppRushYdsAtt=("Opp Rush Y/A", "mean"),
            OppPassTDG=("Opp Pass TD", "mean"),
            OppRushTDG=("Opp Rush TD", "mean"),
            OppFGpct=("Opp FG%", "mean"),
            OppPenYdsG=("Opp Pen Yds", "mean"),
            OppTOVG=("Opp Tot TO", "mean"),
            OppDwn3Conv=("3Dwn Conv %", "mean"),
        )
        .reset_index()
    )

    # TmLuck aims to calculate how many more/less games a team has won versus what their Pts (scored & allowed) accounts for
    ## + value: more wins than should be (lucky)
    ## - value: less wins than should be (unlucky)
    gm_df["TmLuck"] = gm_df["W"] - gm_df["TmLuckW"]

    gm_df["Game Date"] = gm_date
    gm_df["Season"] = season

    # adjusted efficiencies
    ## aims to show % of possessions end in points (either TD or FG)
    ## final adjustment made for "Luck" team is currently experiencing
    gm_df["OffEffAdj"] = ((gm_df["TotScorePlays"] / gm_df["TotPoss"])) * (
        1 + (gm_df["TmLuck"] / gm_df["G"])
    )
    gm_df["DefEffAdj"] = ((gm_df["TotOppScorePlays"] / gm_df["TotOppPoss"])) * (
        1 - ((1 - gm_df["TmLuck"]) / gm_df["G"])
    )

    # transform gm_df into single line game for game results df
    hm = (
        gm_df[gm_df["Tm"] == hm_tm]
        .add_prefix("Hm_")
        .rename(columns={"Hm_Game Date": "Game Date"})
    )
    aw = (
        gm_df[gm_df["Tm"] == aw_tm]
        .add_prefix("Aw_")
        .rename(columns={"Aw_Game Date": "Game Date"})
    )

    full_gm_df = (aw.merge(hm, how="outer", on=["Game Date"])).reset_index(drop=True)
    full_gm_df["Matchup"] = full_gm_df["Aw_Tm"] + " vs. " + full_gm_df["Hm_Tm"]

    return full_gm_df


###############################################################################
"""
    FUNCTIONS USED FOR SAVING DATA
"""


def save_nfl_odds():

    robjects.r(
        f"""
        # Packages
        library(nflfastR)
        library(nflplotR)
        library(nflreadr)
        library(nflverse)
        library(dplyr)
        library(tidyverse)
        library(readr)

        options(scipen=999)

        past_schedules <-load_schedules(seasons = TRUE)
    
        # all NFL games (week by week)
        games <- past_schedules %>%
        select(
            game_id,
            season,
            gameday,
            weekday,
            away_team,
            away_score,
            home_team,
            home_score,
            total,
            spread_line,
            home_spread_odds,
            away_spread_odds,
            total_line,
            home_moneyline,
            away_moneyline,
            div_game,
        ) %>% 
        mutate(
            across(c(spread_line), ~ . * -1)
        ) %>%
        rename(
            hm_spread = spread_line,
            total_score = total,
        )
    """
    )

    nfl_df = robjects.globalenv["games"]

    with localconverter(robjects.default_converter + pandas2ri.converter) as cv:
        nfl_df_pd = robjects.conversion.get_conversion().rpy2py(nfl_df)

    # fix null values with replacement
    nfl_df_pd = nfl_df_pd.replace(-2147483648, np.nan)

    # write the data frame to a CSV file
    nfl_df_pd.to_csv(
        f"C:\\Users\\{os.getlogin()}\\personal-github\\nfl-win-probability\\csv_files\\nfl_game_results.csv",
        index=False,
    )

    return


def save_team_stats(season_year, team_name_list, today):
    teamnm_df = get_teamnm()

    for n, tm in enumerate(team_name_list):
        logger.info(f"Adding {n+1}/{len(team_name_list)}: {tm}")
        team_df = teamnm_df[(teamnm_df["Tm Abbrv"] == tm)].reset_index(drop=True)

        # pull season gamelog
        try:
            team_gamelog = get_team_stats(season_year, team_df["Gamelog Name"][0], tm)
            team_gamelog = team_gamelog[
                (team_gamelog["Date"].astype("datetime64[ns]"))
                < pd.to_datetime(today).strftime("%Y-%m-%d")
            ]

            team_gamelog = team_gamelog[
                [
                    "G",
                    "Week",
                    "Date",
                    "Day",
                    "Location",
                    "Opp",
                    "W/L",
                    "Tm Pts",
                    "Opp Pts",
                    "OT",
                    "Pass Cmp",
                    "Pass Att",
                    "Pass Cmp %",
                    "Pass Yds",
                    "Pass TD",
                    "Pass Y/A",
                    "Pass Adj Y/A",
                    "Pass Rate",
                    "Sacks",
                    "Sack Yds",
                    "Rush Att",
                    "Rush Yds",
                    "Rush Y/A",
                    "Rush TD",
                    "Tot Plays",
                    "Tot Yds",
                    "Avg Yds",
                    "FGA",
                    "FGM",
                    "XPA",
                    "XPM",
                    "Punt Att",
                    "Punt Yds",
                    "1st Dwn Pass",
                    "1st Dwn Rush",
                    "1st Dwn Penalty",
                    "Tot 1st Dwn",
                    "3rd Dwn Conv",
                    "3rd Dwn Att",
                    "4th Dwn Conv",
                    "4th Dwn Att",
                    "Pen",
                    "Pen Yds",
                    "Fmbl",
                    "Int",
                    "Tot TO",
                    "ToP",
                    "Opp Pass Cmp",
                    "Opp Pass Att",
                    "Opp Pass Cmp %",
                    "Opp Pass Yds",
                    "Opp Pass TD",
                    "Opp Pass Y/A",
                    "Opp Pass Adj Y/A",
                    "Opp Pass Rate",
                    "Opp Sacks",
                    "Opp Sack Yds",
                    "Opp Rush Att",
                    "Opp Rush Yds",
                    "Opp Rush Y/A",
                    "Opp Rush TD",
                    "Opp Tot Plays",
                    "Opp Tot Yds",
                    "Opp Avg Yds",
                    "Opp FGA",
                    "Opp FGM",
                    "Opp XPA",
                    "Opp XPM",
                    "Opp Punt Att",
                    "Opp Punt Yds",
                    "Opp 1st Dwn Pass",
                    "Opp 1st Dwn Rush",
                    "Opp 1st Dwn Penalty",
                    "Opp Tot 1st Dwn",
                    "Opp 3rd Dwn Conv",
                    "Opp 3rd Dwn Att",
                    "Opp 4th Dwn Conv",
                    "Opp 4th Dwn Att",
                    "Opp Pen",
                    "Opp Pen Yds",
                    "Opp Fmbl",
                    "Opp Int",
                    "Opp Tot TO",
                    "Opp ToP",
                    "Playoffs",
                    "Tm",
                    "Tm Div",
                    "Opp Div",
                ]
            ].astype(
                {
                    "OT": "object",
                    "Opp Div": "object",
                    "Location": "object",
                }
            )
        except:
            team_gamelog = pd.DataFrame(
                columns=[
                    "G",
                    "Week",
                    "Date",
                    "Day",
                    "Location",
                    "Opp",
                    "W/L",
                    "Tm Pts",
                    "Opp Pts",
                    "OT",
                    "Pass Cmp",
                    "Pass Att",
                    "Pass Cmp %",
                    "Pass Yds",
                    "Pass TD",
                    "Pass Y/A",
                    "Pass Adj Y/A",
                    "Pass Rate",
                    "Sacks",
                    "Sack Yds",
                    "Rush Att",
                    "Rush Yds",
                    "Rush Y/A",
                    "Rush TD",
                    "Tot Plays",
                    "Tot Yds",
                    "Avg Yds",
                    "FGA",
                    "FGM",
                    "XPA",
                    "XPM",
                    "Punt Att",
                    "Punt Yds",
                    "1st Dwn Pass",
                    "1st Dwn Rush",
                    "1st Dwn Penalty",
                    "Tot 1st Dwn",
                    "3rd Dwn Conv",
                    "3rd Dwn Att",
                    "4th Dwn Conv",
                    "4th Dwn Att",
                    "Pen",
                    "Pen Yds",
                    "Fmbl",
                    "Int",
                    "Tot TO",
                    "ToP",
                    "Opp Pass Cmp",
                    "Opp Pass Att",
                    "Opp Pass Cmp %",
                    "Opp Pass Yds",
                    "Opp Pass TD",
                    "Opp Pass Y/A",
                    "Opp Pass Adj Y/A",
                    "Opp Pass Rate",
                    "Opp Sacks",
                    "Opp Sack Yds",
                    "Opp Rush Att",
                    "Opp Rush Yds",
                    "Opp Rush Y/A",
                    "Opp Rush TD",
                    "Opp Tot Plays",
                    "Opp Tot Yds",
                    "Opp Avg Yds",
                    "Opp FGA",
                    "Opp FGM",
                    "Opp XPA",
                    "Opp XPM",
                    "Opp Punt Att",
                    "Opp Punt Yds",
                    "Opp 1st Dwn Pass",
                    "Opp 1st Dwn Rush",
                    "Opp 1st Dwn Penalty",
                    "Opp Tot 1st Dwn",
                    "Opp 3rd Dwn Conv",
                    "Opp 3rd Dwn Att",
                    "Opp 4th Dwn Conv",
                    "Opp 4th Dwn Att",
                    "Opp Pen",
                    "Opp Pen Yds",
                    "Opp Fmbl",
                    "Opp Int",
                    "Opp Tot TO",
                    "Opp ToP",
                    "Playoffs",
                    "Tm",
                    "Tm Div",
                    "Opp Div",
                ]
            ).astype(
                {
                    "OT": object,
                    "Opp Div": object,
                    "Location": object,
                }
            )

        # restructure Cmp%
        team_gamelog["Pass Cmp %"] = team_gamelog["Pass Cmp %"] / 100
        team_gamelog["Opp Pass Cmp %"] = team_gamelog["Opp Pass Cmp %"] / 100

        # recalculate Rush Y/A
        team_gamelog["Rush Y/A"] = team_gamelog["Rush Yds"] / team_gamelog["Rush Att"]
        team_gamelog["Opp Rush Y/A"] = (
            team_gamelog["Opp Rush Yds"] / team_gamelog["Opp Rush Att"]
        )

        # pull out .csv
        try:
            season_gamelogs = pd.read_csv(
                f"~/personal-github/nfl-win-probability/csv_files/{season_year}/season{season_year}_tm_gamelogs.csv",
            )
            season_gamelogs = season_gamelogs.astype(
                {
                    "OT": "object",
                    "Opp Div": "object",
                    "Location": "object",
                }
            )
        except:
            season_gamelogs = pd.DataFrame()

        add_tm = (
            pd.concat(
                [
                    season_gamelogs,
                    team_gamelog.astype(season_gamelogs.dtypes),
                ]
            )
            .drop_duplicates(subset=["Opp", "Date"], keep="last")
            .reset_index(drop=True)
        )

        add_tm.to_csv(
            f"~/personal-github/nfl-win-probability/csv_files/{season_year}/season{season_year}_tm_gamelogs.csv",
            index=False,
        )

        # sleep 10 seconds after each data pull
        time.sleep(10)

    return print(f"{season_year} gamelogs saved to .csv")


# TODO: edit game_results() for neutral games with ratings (tm_elo_rating())
def game_results(season, save=False):
    """
    Currently built for non Neutral site games b/c Neutral site games have no "Home" team
    """

    season_gm_results = pd.DataFrame(
        columns=[
            "Game Date",
            "Location",
            "Divisional Game",
            "Playoff Game",
            "Matchup",
            "Home Team",
            "Home Elo",
            "Home Lg Rank",
            "Home Pts",
            "Away Team",
            "Away Elo",
            "Away Lg Rank",
            "Away Pts",
            "Home W",
            "Home Pt Diff",
            "Home Spread",
            "Home Spread W",
            "Home Moneyline",
            "Away Moneyline",
        ]
    )

    # all teams
    team_df = get_teamnm()

    # read in nfl_odds()
    nfl_odds_df = nfl_odds(season).reset_index(drop=True)

    # run through each team to get the results and compile to df
    for n, tm_url in enumerate(team_df["Gamelog Name"].unique().tolist()):

        logger.info(f'Running {n+1}/{len(team_df)}: {team_df["Tm Name"][n]}')

        # read from saved boxscore .csv
        gmlog = pd.read_csv(
            f"~/personal-github/nfl-win-probability/csv_files/{season}/season{season}_tm_gamelogs.csv",
        )
        tm_gmlog = gmlog[gmlog["Tm"] == team_df["Tm Abbrv"][n]]

        tm_gmlog = tm_gmlog[
            (
                ~(tm_gmlog["Opp"].isin(["", "Opponent", "Opp", pd.NA, np.nan]))
                & ~(tm_gmlog["W/L"].isin(["", np.nan, pd.NA]))
                & ~(tm_gmlog["Location"] == "N")
            )
        ].reset_index(drop=True)

        tm_gmlog = tm_gmlog.astype(
            {
                "Tm Pts": int,
                "Opp Pts": int,
            }
        )

        for game in tm_gmlog.index:

            try:
                tmp_game = tm_gmlog.iloc[game]

                # divisional game flag
                if tmp_game["Tm Div"] == tmp_game["Opp Div"]:
                    divisional = 1
                else:
                    divisional = 0

                # playoff game flag
                if tmp_game["Playoffs"] == 1:
                    playoff = 1
                else:
                    playoff = 0

                if tmp_game["Location"] == "@":
                    tm_list = [tmp_game["Tm"], tmp_game["Opp"]]

                    ratings = tm_elo_rating(
                        season,
                        tmp_game["Date"],
                    )
                    try:
                        tm_ratings = ratings[ratings["Tm"].isin(tm_list)].reset_index(
                            drop=True
                        )
                        aw_elo = tm_ratings[
                            (tm_ratings["Tm"] == tmp_game["Tm"])
                            & (tm_ratings["Opp"] == tmp_game["Opp"])
                        ].reset_index(drop=True)["Tm Elo"][0]
                        hm_elo = tm_ratings[
                            (tm_ratings["Tm"] == tmp_game["Tm"])
                            & (tm_ratings["Opp"] == tmp_game["Opp"])
                        ].reset_index(drop=True)["Opp Elo"][0]
                    except KeyError:
                        hm_elo = ratings["Tm Elo"].min()

                    if tm_ratings.empty:
                        hm_elo = pd.NA
                        aw_elo = pd.NA

                    # lg rankings
                    lg_ranking = tm_lg_ranking(
                        season,
                        tmp_game["Date"],
                    )[["Tm", "Opp", "Tm Rnk", "Opp Rnk"]]
                    aw_lgr = lg_ranking[
                        (lg_ranking["Tm"] == tmp_game["Tm"])
                        & (lg_ranking["Opp"] == tmp_game["Opp"])
                    ].reset_index(drop=True)["Tm Rnk"][0]
                    hm_lgr = lg_ranking[
                        (lg_ranking["Tm"] == tmp_game["Tm"])
                        & (lg_ranking["Opp"] == tmp_game["Opp"])
                    ].reset_index(drop=True)["Opp Rnk"][0]

                    # win check
                    if tmp_game["Tm Pts"] > tmp_game["Opp Pts"]:
                        hm_tm_w = 0
                    else:
                        hm_tm_w = 1

                    hm_pt_diff = tmp_game["Opp Pts"] - tmp_game["Tm Pts"]

                    try:
                        matchup = f"{team_df["Tm Abbrv"][n]} vs. {team_df[team_df['Tm Abbrv'] == tmp_game["Opp"]].reset_index(drop=True)['Tm Abbrv'][0]}"
                    except KeyError:
                        matchup = f"{team_df['Tm Abbrv'][n]} vs. {tmp_game['Opp']}"

                    home_team = matchup.split(" vs. ", 1)[1]

                    # Home Spread
                    hm_spread = nfl_odds_df.loc[
                        nfl_odds_df["matchup"] == matchup, "hm_spread"
                    ].reset_index(drop=True)[0]

                    # Moneylines
                    hm_ml = nfl_odds_df.loc[
                        nfl_odds_df["matchup"] == matchup, "home_moneyline"
                    ].reset_index(drop=True)[0]
                    aw_ml = nfl_odds_df.loc[
                        nfl_odds_df["matchup"] == matchup, "away_moneyline"
                    ].reset_index(drop=True)[0]

                    # Home Spread W
                    if hm_spread <= 0:
                        if hm_tm_w == 1:
                            if hm_pt_diff > abs(hm_spread):
                                hm_spread_w = 1
                            else:
                                hm_spread_w = 0
                        else:
                            hm_spread_w = 0
                    else:
                        if hm_tm_w == 1:
                            hm_spread_w = 1
                        else:
                            if abs(hm_pt_diff) < abs(hm_spread):
                                hm_spread_w = 1
                            else:
                                hm_spread_w = 0

                    tmp_df = pd.DataFrame(
                        data={
                            "Game Date": [tmp_game["Date"]],
                            "Location": [f"@ {tmp_game["Opp"]}"],
                            # "Neutral Game": [0],
                            "Divisional Game": [divisional],
                            "Playoff Game": [playoff],
                            "Matchup": [matchup],
                            "Home Team": [home_team],
                            "Home Elo": [hm_elo],
                            "Home Lg Rank": [hm_lgr],
                            "Home Pts": [tmp_game["Opp Pts"]],
                            "Away Team": [team_df["Tm Abbrv"][n]],
                            "Away Elo": [aw_elo],
                            "Away Lg Rank": [aw_lgr],
                            "Away Pts": [tmp_game["Tm Pts"]],
                            "Home W": [hm_tm_w],
                            "Home Pt Diff": [hm_pt_diff],
                            "Home Spread": [hm_spread],
                            "Home Spread W": [hm_spread_w],
                            "Home Moneyline": [hm_ml],
                            "Away Moneyline": [aw_ml],
                        }
                    )

                else:
                    tm_list = [tmp_game["Tm"], tmp_game["Opp"]]

                    ratings = tm_elo_rating(
                        season,
                        tmp_game["Date"],
                    )
                    try:
                        tm_ratings = ratings[ratings["Tm"].isin(tm_list)].reset_index(
                            drop=True
                        )
                        hm_elo = tm_ratings[
                            (tm_ratings["Tm"] == tmp_game["Tm"])
                            & (tm_ratings["Opp"] == tmp_game["Opp"])
                        ].reset_index(drop=True)["Tm Elo"][0]
                        aw_elo = tm_ratings[
                            (tm_ratings["Tm"] == tmp_game["Tm"])
                            & (tm_ratings["Opp"] == tmp_game["Opp"])
                        ].reset_index(drop=True)["Opp Elo"][0]
                    except KeyError:
                        aw_elo = ratings["Tm Elo"].min()

                    if tm_ratings.empty:
                        hm_elo = pd.NA
                        aw_elo = pd.NA

                    # lg rankings
                    lg_ranking = tm_lg_ranking(
                        season,
                        tmp_game["Date"],
                    )[["Tm", "Opp", "Tm Rnk", "Opp Rnk"]]
                    hm_lgr = lg_ranking[
                        (lg_ranking["Tm"] == tmp_game["Tm"])
                        & (lg_ranking["Opp"] == tmp_game["Opp"])
                    ].reset_index(drop=True)["Tm Rnk"][0]
                    aw_lgr = lg_ranking[
                        (lg_ranking["Tm"] == tmp_game["Tm"])
                        & (lg_ranking["Opp"] == tmp_game["Opp"])
                    ].reset_index(drop=True)["Opp Rnk"][0]

                    # win check
                    if tmp_game["Tm Pts"] > tmp_game["Opp Pts"]:
                        hm_tm_w = 1
                    else:
                        hm_tm_w = 0

                    hm_pt_diff = tmp_game["Tm Pts"] - tmp_game["Opp Pts"]

                    try:
                        matchup = f"{team_df[team_df['Tm Abbrv'] == tmp_game["Opp"]].reset_index(drop=True)['Tm Abbrv'][0]} vs. {team_df['Tm Abbrv'][n]}"
                    except KeyError:
                        matchup = f"{tmp_game['Opp']} vs. {team_df['Tm Abbrv'][n]}"

                    away_team = matchup.split(" vs. ", 1)[0]

                    # Home Spread
                    hm_spread = nfl_odds_df.loc[
                        nfl_odds_df["matchup"] == matchup, "hm_spread"
                    ].reset_index(drop=True)[0]

                    # Moneylines
                    hm_ml = nfl_odds_df.loc[
                        nfl_odds_df["matchup"] == matchup, "home_moneyline"
                    ].reset_index(drop=True)[0]
                    aw_ml = nfl_odds_df.loc[
                        nfl_odds_df["matchup"] == matchup, "away_moneyline"
                    ].reset_index(drop=True)[0]

                    # Home Spread W
                    if hm_spread <= 0:
                        if hm_tm_w == 1:
                            if hm_pt_diff > abs(hm_spread):
                                hm_spread_w = 1
                            else:
                                hm_spread_w = 0
                        else:
                            hm_spread_w = 0
                    else:
                        if hm_tm_w == 1:
                            hm_spread_w = 1
                        else:
                            if abs(hm_pt_diff) < abs(hm_spread):
                                hm_spread_w = 1
                            else:
                                hm_spread_w = 0

                    tmp_df = pd.DataFrame(
                        data={
                            "Game Date": [tmp_game["Date"]],
                            "Location": [f"@ {team_df["Tm Abbrv"][n]}"],
                            # "Neutral Game": [0],
                            "Divisional Game": [divisional],
                            "Playoff Game": [playoff],
                            "Matchup": [matchup],
                            "Home Team": [team_df["Tm Abbrv"][n]],
                            "Home Elo": [hm_elo],
                            "Home Lg Rank": [hm_lgr],
                            "Home Pts": [tmp_game["Tm Pts"]],
                            "Away Team": [away_team],
                            "Away Elo": [aw_elo],
                            "Away Lg Rank": [aw_lgr],
                            "Away Pts": [tmp_game["Opp Pts"]],
                            "Home W": [hm_tm_w],
                            "Home Pt Diff": [hm_pt_diff],
                            "Home Spread": [hm_spread],
                            "Home Spread W": [hm_spread_w],
                            "Home Moneyline": [hm_ml],
                            "Away Moneyline": [aw_ml],
                        }
                    )

                if season_gm_results.empty:
                    season_gm_results = tmp_df.copy()
                else:
                    season_gm_results = pd.concat(
                        [season_gm_results, tmp_df]
                    ).reset_index(drop=True)
            except KeyError:
                pass
            except ValueError:
                pass

    season_gm_results = season_gm_results.drop_duplicates()

    if save:
        season_gm_results.to_csv(
            f"~/personal-github/nfl-win-probability/csv_files/{season}/season{season}_results.csv",
            index=False,
        )

    return season_gm_results


def season_data(season):

    # read results df
    results_df = pd.read_csv(
        f"~/personal-github/nfl-win-probability/csv_files/{season}/season{season}_results.csv",
    )
    season_df = pd.DataFrame()

    # for each result, run rolling_gamedata()
    for n, matchup in enumerate(results_df["Matchup"].tolist()):
        logger.info(
            f"{n+1}/{len(results_df["Matchup"].tolist())}: {results_df["Matchup"].tolist()[n]}"
        )

        hm_tm = results_df["Home Team"][n]
        aw_tm = results_df["Away Team"][n]
        gm_date = results_df["Game Date"][n]

        try:
            gamelog_stats = rolling_gamedata(season, hm_tm, aw_tm, gm_date)

            try:
                gamelog_stats = gamelog_stats.merge(
                    results_df[
                        [
                            "Matchup",
                            "Home Team",
                            "Home Elo",
                            "Home Lg Rank",
                            "Away Team",
                            "Away Elo",
                            "Away Lg Rank",
                            "Home W",
                            "Home Pt Diff",
                            "Home Spread",
                            "Home Spread W",
                            "Home Moneyline",
                            "Away Moneyline",
                            # "Neutral Game",
                            "Divisional Game",
                            "Playoff Game",
                        ]
                    ],
                    how="inner",
                    left_on=["Matchup", "Aw_Tm", "Hm_Tm"],
                    right_on=["Matchup", "Away Team", "Home Team"],
                )

                season_df = (
                    pd.concat([season_df, gamelog_stats])
                    .drop_duplicates(subset=["Matchup", "Game Date"])
                    .reset_index(drop=True)
                )

            except ValueError:
                pass
        except ValueError:
            pass
        except KeyError:
            pass

    # save .csv file
    season_df.to_csv(
        f"~/personal-github/nfl-win-probability/csv_files/{season}/season{season}_matchup_results.csv",
        index=False,
    )

    return season_df
