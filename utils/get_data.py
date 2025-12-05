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
from logzero import logger
from math import sqrt
from tqdm import tqdm
import time

from utils.webscrape_utils import (
    read_gamelog,
)

from utils.team_dict import get_teamnm

"""
    Team Stats by Game
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
                f"~/personal-github/nfl-win-probability/csv_files/season{season_year}_tm_gamelogs.csv",
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
            f"~/personal-github/nfl-win-probability/csv_files/season{season_year}_tm_gamelogs.csv",
            index=False,
        )

        # sleep 10 seconds after each data pull
        time.sleep(10)

    return print(f"{season_year} gamelogs saved to .csv")


def tm_elo_rating(season_year, today):

    # read out gamelog
    team_gamelog = pd.read_csv(
        f"~/personal-github/nfl-win-probability/csv_files/season{season_year}_tm_gamelogs.csv"
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
        f"~/personal-github/nfl-win-probability/csv_files/season{season_year}_tm_gamelogs.csv"
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
        f"~/personal-github/nfl-win-probability/csv_files/nfl_game_results.csv",
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
    nfl_df = csv_df[
        (csv_df["season"].astype("int16") == season_year)
        # & (csv_df["gameday"] <= pd.to_datetime(today))
    ]

    return nfl_df


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
            f"~/personal-github/nfl-win-probability/csv_files/season{season}_tm_gamelogs.csv",
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
            f"~/personal-github/nfl-win-probability/csv_files/season{season}_results.csv",
            index=False,
        )

    return season_gm_results
