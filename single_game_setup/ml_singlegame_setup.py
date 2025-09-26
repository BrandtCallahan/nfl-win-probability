from datetime import timedelta
from matplotlib.pylab import norm
import pandas as pd
import numpy as np
import pylab as p
import random
import math
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import matplotlib.image as mpimg
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from utils.get_data import *
from utils.team_dict import *
from utils.beautiful_soup_helper import *

pd.set_option("future.no_silent_downcasting", True)


def gamelog_setup(season, tm_name, gm_date):

    # make sure 'today' is a date and not string
    gm_date = pd.to_datetime(gm_date)

    team_df = get_teamnm()
    tm_df = team_df[team_df["Tm Abbrv"] == tm_name].reset_index(drop=True)

    # pull data from csv_files folder
    gamelog = pd.read_csv(
        f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/nfl/csv_files/season{season}_tm_gamelogs.csv",
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

    # W Streak == +, L Streak == -
    team_gamelog.loc[team_gamelog["W/L"] == "W", "Streak +/-"] = team_gamelog[
        "Running Streak"
    ]
    team_gamelog.loc[team_gamelog["W/L"] == "L", "Streak +/-"] = (
        team_gamelog["Running Streak"] * -1
    )
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
    )
    team_gamelog["Opp Poss"] = (
        team_gamelog["Opp Tot TO"].astype(int)
        + team_gamelog["Opp Punt Att"].astype(int)
        + team_gamelog["Opp Pass TD"].astype(int)
        + team_gamelog["Opp Rush TD"].astype(int)
        + team_gamelog["Opp FGA"].astype(int)
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
    team_gamelog["Off Eff"] = (
        team_gamelog["Adj Pass Off Eff"] + team_gamelog["Adj Rush Off Eff"]
    )

    # defense ratings
    team_gamelog["Pass Def Eff"] = team_gamelog["Opp Pass Rate"].astype(float)
    team_gamelog["Rush Def Eff"] = team_gamelog["Opp Rush Y/A"].astype(float)
    # compare to college avg.
    team_gamelog["Adj Pass Def Eff"] = team_gamelog["Pass Def Eff"] - 100
    team_gamelog["Adj Rush Def Eff"] = (
        team_gamelog["Rush Def Eff"] - gamelog["Opp Rush Y/A"].median()
    )
    # defense efficency
    team_gamelog["Def Eff"] = (
        team_gamelog["Adj Pass Def Eff"] + team_gamelog["Adj Rush Def Eff"]
    )

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

    # group gamelog stats (season avg.)
    gm_df = (
        gm_log.groupby(["Tm"], observed=True)
        .agg(
            W=("W/L Flag", "sum"),
            Wpct=("Total W%", "last"),
            Poss=("Poss", "median"),
            OppPoss=("Opp Poss", "median"),
            PassOffEff=("Adj Pass Off Eff", "median"),
            RushOffEff=("Adj Rush Off Eff", "median"),
            PassDefEff=("Adj Pass Def Eff", "median"),
            RushDefEff=("Adj Rush Def Eff", "median"),
            TmLuckW=("Tm Luck", "sum"),
            Pts=("Tm Pts", "median"),
            OppPts=("Opp Pts", "median"),
            TmDiv=("Tm Div", "first"),
            # start boxscore stats
            PassCmppct=("Pass Cmp %", "median"),
            PassAdjYdsAtt=("Pass Adj Y/A", "median"),
            RushYdsAtt=("Rush Y/A", "median"),
            PassTDG=("Pass TD", "median"),
            RushTDG=("Rush TD", "median"),
            FGpct=("FG%", "median"),
            PenYdsG=("Pen Yds", "median"),
            TOVG=("Tot TO", "median"),
            Dwn3Conv=("3Dwn Conv %", "median"),
            # opponent boxscore stats
            OppPassCmppct=("Opp Pass Cmp %", "median"),
            OppPassAdjYdsAtt=("Opp Pass Adj Y/A", "median"),
            OppRushYdsAtt=("Opp Rush Y/A", "median"),
            OppPassTDG=("Opp Pass TD", "median"),
            OppRushTDG=("Opp Rush TD", "median"),
            OppFGpct=("Opp FG%", "median"),
            OppPenYdsG=("Opp Pen Yds", "median"),
            OppTOVG=("Opp Tot TO", "median"),
            OppDwn3Conv=("3Dwn Conv %", "median"),
        )
        .reset_index()
    )
    gm_df["TmLuck"] = gm_df["W"] - gm_df["TmLuckW"]
    gm_df["Game Date"] = gm_date
    gm_df["Season"] = season

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


def season_data(season):

    # read results df
    results_df = pd.read_csv(
        f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/nfl/csv_files/season{season}_results.csv",
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
                            "Away Team",
                            "Away Elo",
                            "Home W",
                            "Home Pt Diff",
                            # "Neutral Game",
                            "Divisional Game",
                            "Playoff Game",
                        ]
                    ],
                    how="inner",
                    left_on=["Matchup", "Aw_Tm", "Hm_Tm"],
                    right_on=["Matchup", "Away Team", "Home Team"],
                )

                """
                # Instead of relying on Home Pt Diff alone, build out normalized variable (condensed options)
                gamelog_stats.loc[
                    gamelog_stats["Home Pt Diff"] < -21.5, "Home Pt Margin"
                ] = -21
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] > -21.5)
                    & (gamelog_stats["Home Pt Diff"] < -14.5),
                    "Home Pt Margin",
                ] = -14
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] > -14.5)
                    & (gamelog_stats["Home Pt Diff"] < -10.5),
                    "Home Pt Margin",
                ] = -10
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] > -10.5)
                    & (gamelog_stats["Home Pt Diff"] < -7.5),
                    "Home Pt Margin",
                ] = -7
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] > -7.5)
                    & (gamelog_stats["Home Pt Diff"] < 3.5),
                    "Home Pt Margin",
                ] = -3
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] > -3.5)
                    & (gamelog_stats["Home Pt Diff"] < -1.5),
                    "Home Pt Margin",
                ] = -1
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] < 1.5)
                    & (gamelog_stats["Home Pt Diff"] > -1.5),
                    "Home Pt Margin",
                ] = 0
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] < 3.5)
                    & (gamelog_stats["Home Pt Diff"] > 1.5),
                    "Home Pt Margin",
                ] = 1
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] < 7.5)
                    & (gamelog_stats["Home Pt Diff"] > 3.5),
                    "Home Pt Margin",
                ] = 3
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] < 10.5)
                    & (gamelog_stats["Home Pt Diff"] > 7.5),
                    "Home Pt Margin",
                ] = 7
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] < 14.5)
                    & (gamelog_stats["Home Pt Diff"] > 10.5),
                    "Home Pt Margin",
                ] = 10
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] < 21.5)
                    & (gamelog_stats["Home Pt Diff"] > 14.5),
                    "Home Pt Margin",
                ] = 14
                gamelog_stats.loc[
                    (gamelog_stats["Home Pt Diff"] > 21.5),
                    "Home Pt Margin",
                ] = 21
                """

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
        f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/nfl/csv_files/season{season}_matchup_results.csv",
        index=False,
    )

    return season_df


def single_game_model(data_seasons, today, matchup):

    # load up matchup results for training
    matchup_df = pd.DataFrame()
    for season in data_seasons:
        tmp_df = pd.read_csv(
            f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/nfl/csv_files/season{season}_matchup_results.csv",
        )
        tmp_df = tmp_df.astype({"Game Date": "datetime64[ns]"})

        # concat all years of data into one df
        matchup_df = pd.concat([matchup_df, tmp_df]).reset_index(drop=True)

    """
    # add boolean flags for Home Pt Margin (i.e. <3, >3, >7, >10, >14, >21)
    matchup_df["Hm_PtMargin-21+"] = matchup_df["Home Pt Diff"] < -21
    matchup_df["Hm_PtMargin-21"] = matchup_df["Home Pt Diff"] >= -21
    matchup_df["Hm_PtMargin-14"] = matchup_df["Home Pt Diff"] >= -14
    matchup_df["Hm_PtMargin-10"] = matchup_df["Home Pt Diff"] >= -10
    matchup_df["Hm_PtMargin-7"] = matchup_df["Home Pt Diff"] >= -7
    matchup_df["Hm_PtMargin-3"] = matchup_df["Home Pt Diff"] >= -3
    matchup_df["Hm_PtMargin-1"] = matchup_df["Home Pt Diff"] >= -1
    matchup_df["Hm_PtMargin1"] = matchup_df["Home Pt Diff"] <= 1
    matchup_df["Hm_PtMargin3"] = matchup_df["Home Pt Diff"] <= 3
    matchup_df["Hm_PtMargin7"] = matchup_df["Home Pt Diff"] <= 7
    matchup_df["Hm_PtMargin10"] = matchup_df["Home Pt Diff"] <= 10
    matchup_df["Hm_PtMargin14"] = matchup_df["Home Pt Diff"] <= 14
    matchup_df["Hm_PtMargin21"] = matchup_df["Home Pt Diff"] <= 21
    matchup_df["Hm_PtMargin21+"] = matchup_df["Home Pt Diff"] > 21

    for col in [
        "Hm_PtMargin-21+",
        "Hm_PtMargin-21",
        "Hm_PtMargin-14",
        "Hm_PtMargin-10",
        "Hm_PtMargin-7",
        "Hm_PtMargin-3",
        "Hm_PtMargin-1",
        "Hm_PtMargin1",
        "Hm_PtMargin3",
        "Hm_PtMargin7",
        "Hm_PtMargin10",
        "Hm_PtMargin14",
        "Hm_PtMargin21",
        "Hm_PtMargin21+",
    ]:
        matchup_df[f"{col}"] = matchup_df[f"{col}"].replace({True: 1, False: 0})
    """

    # make sure only data from before "today"
    matchup_df = matchup_df[
        matchup_df["Game Date"] < pd.to_datetime(today)
    ].reset_index(drop=True)

    # dummy variable conferences
    hm_conf_dummy_df = pd.get_dummies(
        matchup_df["Hm_TmDiv"],
        prefix="Hm_TmDiv",
        dtype=int,
    )
    aw_conf_dummy_df = pd.get_dummies(
        matchup_df["Aw_TmDiv"],
        prefix="Aw_TmDiv",
        dtype=int,
    )
    # join back to matchup_df
    matchup_df = pd.concat([matchup_df, hm_conf_dummy_df, aw_conf_dummy_df], axis=1)

    # build point margin variables (+7, +3, -3, -7)
    matchup_df["Hm +7"] = matchup_df["Home Pt Diff"] > 7
    matchup_df["Hm +3"] = matchup_df["Home Pt Diff"] > 3
    matchup_df["Hm -3"] = matchup_df["Home Pt Diff"] < 3
    matchup_df["Hm -7"] = matchup_df["Home Pt Diff"] < 7

    tf_dict = {True: 1, False: 0}
    for col in ["Hm +7", "Hm +3", "Hm -3", "Hm -7"]:
        matchup_df[f"{col}"] = matchup_df[f"{col}"].map(tf_dict)

    # data for the matchup
    hm_tm = matchup.split(" vs. ")[1]
    aw_tm = matchup.split(" vs. ")[0]
    matchup_data = rolling_gamedata(
        data_seasons[-1],
        hm_tm,
        aw_tm,
        pd.to_datetime(today),
    )
    matchup_data.loc[
        matchup_data["Hm_TmDiv"] == matchup_data["Aw_TmDiv"], "Divisional Game"
    ] = 1
    matchup_data["Divisional Game"] = matchup_data["Divisional Game"].fillna(0)

    # dummy variable conference
    hm_conf_dummy = pd.get_dummies(
        matchup_data["Hm_TmDiv"],
        prefix="Hm_TmDiv",
        dtype=int,
    )
    aw_conf_dummy = pd.get_dummies(
        matchup_data["Aw_TmDiv"],
        prefix="Aw_TmDiv",
        dtype=int,
    )

    # join back to matchup_data
    matchup_data = pd.concat([matchup_data, hm_conf_dummy, aw_conf_dummy], axis=1)
    matchup_data["Aw_FGpct"] = matchup_data["Aw_FGpct"].fillna(0)
    matchup_data["Hm_FGpct"] = matchup_data["Hm_FGpct"].fillna(0)

    # TODO: need to add in Home and Away Elo
    # matchup_data['']
    season = matchup_data["Hm_Season"][0]
    matchup_teams = [
        matchup_data["Matchup"][0].split(" vs. ")[0],
        matchup_data["Matchup"][0].split(" vs. ")[1],
    ]

    elo_df = tm_elo_rating(season, pd.to_datetime(today) + timedelta(days=7))
    elo_df = elo_df[pd.to_datetime(elo_df["Date"]) < pd.to_datetime(today)].reset_index(
        drop=True
    )

    tm_elo = (
        elo_df[(elo_df["Tm"].isin(matchup_teams)) | (elo_df["Opp"].isin(matchup_teams))]
        .sort_values(by=["Date"])
        .reset_index(drop=True)
    )
    hm_results = tm_elo[
        (tm_elo["Tm"] == matchup_data["Matchup"][0].split(" vs. ")[1])
        | (tm_elo["Opp"] == matchup_data["Matchup"][0].split(" vs. ")[1])
    ]
    hm_elo_df = hm_results.iloc[-1, :]
    if (matchup_data["Matchup"][0].split(" vs. ")[1]) in hm_elo_df["Tm"]:
        hm_elo = hm_elo_df["Tm Elo"]
    else:
        hm_elo = hm_elo_df["Opp Elo"]

    aw_results = tm_elo[
        (tm_elo["Tm"] == matchup_data["Matchup"][0].split(" vs. ")[0])
        | (tm_elo["Opp"] == matchup_data["Matchup"][0].split(" vs. ")[0])
    ]
    aw_elo_df = aw_results.iloc[-1, :]
    if (matchup_data["Matchup"][0].split(" vs. ")[0]) in aw_elo_df["Tm"]:
        aw_elo = aw_elo_df["Tm Elo"]
    else:
        aw_elo = aw_elo_df["Opp Elo"]

    matchup_data["Home Elo"] = [hm_elo]
    matchup_data["Away Elo"] = [aw_elo]
    matchup_data["Elo_diff"] = matchup_data["Home Elo"] - matchup_data["Away Elo"]

    # need to add all columns from matchup_df to matchup_data (the conference dummies)
    for col in matchup_df.columns.tolist():
        if col in matchup_data.columns.tolist():
            pass
        else:
            matchup_data[f"{col}"] = 0

    """
    # get team ratings
    tm_ratings = tm_rating(data_seasons[-1], today)
    aw_tm_rating = (
        tm_ratings[tm_ratings["Tm"] == aw_tm]
        .reset_index(drop=True)
        .rename(
            columns={
                "Tm": "Aw_Tm",
                "Tm Rank": "Aw_Rank",
            }
        )
    )
    hm_tm_rating = (
        tm_ratings[tm_ratings["Tm"] == hm_tm]
        .reset_index(drop=True)
        .rename(
            columns={
                "Tm": "Hm_Tm",
                "Tm Rank": "Hm_Rank",
            }
        )
    )
    

    # join to matchup_data
    matchup_data = matchup_data.merge(
        aw_tm_rating[["Aw_Tm", "Aw_Rank"]], how="inner", on=["Aw_Tm"]
    ).merge(hm_tm_rating[["Hm_Tm", "Hm_Rank"]], how="inner", on=["Hm_Tm"])
    matchup_data = matchup_data.rename(
        columns={"Hm_Rank": "Home Rk", "Aw_Rank": "Away Rk", "Hm_Season": "Season"}
    ).drop(columns=["Aw_Season"])
    
    # if NCAA Tourney game get seed
    if game_details["NCAA Tourney Game"]:
        seeding_df = pd.read_csv(
            f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/csv_files/season_bracketology.csv",
        )
        season_seeds = seeding_df[seeding_df["season"] == data_seasons[-1]]

        # away seed
        aw_seed = (
            season_seeds[season_seeds["tm"] == aw_tm]
            .reset_index(drop=True)
            .rename(
                columns={
                    "tm": "Aw_Tm",
                    "seed": "Aw_Seed",
                }
            )
        )
        # home seed
        hm_seed = (
            season_seeds[season_seeds["tm"] == hm_tm]
            .reset_index(drop=True)
            .rename(
                columns={
                    "tm": "Hm_Tm",
                    "seed": "Hm_Seed",
                }
            )
        )
        # join to matchup_data
        matchup_data = matchup_data.merge(
            aw_seed[["Aw_Tm", "Aw_Seed"]], how="inner", on=["Aw_Tm"]
        ).merge(hm_seed[["Hm_Tm", "Hm_Seed"]], how="inner", on=["Hm_Tm"])
    else:
        matchup_data["Aw_Seed"] = pd.NA
        matchup_data["Hm_Seed"] = pd.NA
    
    # additional matchup details
    matchup_data["Neutral Game"] = [game_details["Neutral Game"]]
    matchup_data["Conference Game"] = [game_details["Conference Game"]]
    matchup_data["Postseason Game"] = [game_details["Postseason Game"]]
    matchup_data["NCAA Tourney Game"] = [game_details["NCAA Tourney Game"]]

    matchup_data = matchup_data.fillna(np.nan)
    """

    """
        Build Out Model
    """
    # Exclude Playoff Games for now
    matchup_df = matchup_df[matchup_df["Playoff Game"] == 0]

    model_df = (
        matchup_df.sort_values(by=["Game Date", "Matchup"])
        .reset_index(drop=True)[
            [
                "Matchup",
                "Game Date",
                "Aw_TmDiv",
                "Hm_TmDiv",
                "Home W",
                "Home Pt Diff",
                "Hm +7",
                "Hm +3",
                "Hm -3",
                "Hm -7",
                "Divisional Game",
                "Away Team",
                "Away Elo",
                "Aw_W",
                "Aw_Wpct",
                "Aw_Poss",
                "Aw_OppPoss",
                "Aw_PassOffEff",
                "Aw_RushOffEff",
                "Aw_PassDefEff",
                "Aw_RushDefEff",
                "Aw_TmLuckW",
                "Aw_Pts",
                "Aw_OppPts",
                "Aw_TmDiv",
                "Aw_PassCmppct",
                "Aw_PassAdjYdsAtt",
                "Aw_RushYdsAtt",
                "Aw_PassTDG",
                "Aw_RushTDG",
                "Aw_FGpct",
                "Aw_PenYdsG",
                "Aw_TOVG",
                "Aw_Dwn3Conv",
                "Aw_OppPassCmppct",
                "Aw_OppPassAdjYdsAtt",
                "Aw_OppRushYdsAtt",
                "Aw_OppPassTDG",
                "Aw_OppRushTDG",
                "Aw_OppFGpct",
                "Aw_OppPenYdsG",
                "Aw_OppTOVG",
                "Aw_OppDwn3Conv",
                "Aw_TmLuck",
                "Aw_TmDiv_AFC East",
                "Aw_TmDiv_AFC North",
                "Aw_TmDiv_AFC South",
                "Aw_TmDiv_AFC West",
                "Aw_TmDiv_NFC East",
                "Aw_TmDiv_NFC North",
                "Aw_TmDiv_NFC South",
                "Aw_TmDiv_NFC West",
                "Home Team",
                "Home Elo",
                "Hm_W",
                "Hm_Wpct",
                "Hm_Poss",
                "Hm_OppPoss",
                "Hm_PassOffEff",
                "Hm_RushOffEff",
                "Hm_PassDefEff",
                "Hm_RushDefEff",
                "Hm_TmLuckW",
                "Hm_Pts",
                "Hm_OppPts",
                "Hm_TmDiv",
                "Hm_PassCmppct",
                "Hm_PassAdjYdsAtt",
                "Hm_RushYdsAtt",
                "Hm_PassTDG",
                "Hm_RushTDG",
                "Hm_FGpct",
                "Hm_PenYdsG",
                "Hm_TOVG",
                "Hm_Dwn3Conv",
                "Hm_OppPassCmppct",
                "Hm_OppPassAdjYdsAtt",
                "Hm_OppRushYdsAtt",
                "Hm_OppPassTDG",
                "Hm_OppRushTDG",
                "Hm_OppFGpct",
                "Hm_OppPenYdsG",
                "Hm_OppTOVG",
                "Hm_OppDwn3Conv",
                "Hm_TmLuck",
                "Hm_TmDiv_AFC East",
                "Hm_TmDiv_AFC North",
                "Hm_TmDiv_AFC South",
                "Hm_TmDiv_AFC West",
                "Hm_TmDiv_NFC East",
                "Hm_TmDiv_NFC North",
                "Hm_TmDiv_NFC South",
                "Hm_TmDiv_NFC West",
            ]
        ]
        .sort_values(by=["Game Date", "Matchup"])
        .reset_index(drop=True)
    )

    # fill FG% with median
    model_df["Hm_FGpct"] = model_df["Hm_FGpct"].fillna(model_df["Hm_FGpct"].median())
    model_df["Hm_OppFGpct"] = model_df["Hm_OppFGpct"].fillna(
        model_df["Hm_OppFGpct"].median()
    )
    model_df["Aw_FGpct"] = model_df["Aw_FGpct"].fillna(model_df["Aw_FGpct"].median())
    model_df["Aw_OppFGpct"] = model_df["Aw_OppFGpct"].fillna(
        model_df["Aw_OppFGpct"].median()
    )

    # Elo rating difference
    model_df["Elo_diff"] = model_df["Home Elo"] - model_df["Away Elo"]

    # feature selection
    model_df = model_df[
        [
            "Home W",
            "Home Team",
            "Away Team",
            "Matchup",
            "Game Date",
            "Home Pt Diff",
            "Hm +7",
            "Hm +3",
            "Hm -3",
            "Hm -7",
            "Home Elo",
            "Away Elo",
            "Aw_PassOffEff",
            "Aw_Pts",
            "Aw_PassCmppct",
            "Aw_PassAdjYdsAtt",
            "Aw_PassTDG",
            "Aw_Dwn3Conv",
            "Aw_OppRushYdsAtt",
            "Aw_OppTOVG",
            "Aw_OppDwn3Conv",
            "Aw_TmLuck",
            "Aw_TmDiv_NFC West",
            "Hm_W",
            "Hm_Poss",
            "Hm_PassOffEff",
            "Hm_RushOffEff",
            "Hm_TmLuckW",
            "Hm_Pts",
            "Hm_PassCmppct",
            "Hm_PassAdjYdsAtt",
            "Hm_RushYdsAtt",
            "Hm_RushTDG",
            "Hm_Dwn3Conv",
            "Hm_OppPassAdjYdsAtt",
            "Hm_OppTOVG",
            "Hm_OppDwn3Conv",
            "Hm_TmLuck",
            "Hm_TmDiv_AFC North",
            "Hm_TmDiv_NFC South",
            "Elo_diff",
        ]
    ]

    """
        Set Target Variables/DFs
    """
    final_pred_df = pd.DataFrame()
    final_model_coef = pd.DataFrame()
    final_model_stats = pd.DataFrame()
    for target_variable in [
        "Home W",
        "Home Pt Diff",
        "Hm_Pts",
        "Aw_Pts",
    ]:
        logger.info(f"data transformed: setting target variable - {target_variable}")
        # target variable
        y = model_df[
            [
                f"Matchup",
                f"{target_variable}",
            ]
        ]
        model_y = y.drop(columns=["Matchup"])

        if target_variable in ['Home W', 'Home Pt Diff']:
            X = model_df.drop(
                columns=[
                    "Home W",
                    "Home Team",
                    "Away Team",
                    "Game Date",
                    "Home Pt Diff",
                    "Hm +7",
                    "Hm +3",
                    "Hm -3",
                    "Hm -7",
                    "Home Elo",
                    "Away Elo",
                    # # excluded and put in as dummy variable
                    # "Aw_TmDiv",
                    # "Hm_TmDiv",
                    f"{target_variable}",
                ]
            )
            model_X = X.drop(columns=["Matchup"])
        else:
            X = model_df.drop(
                columns=[
                    "Home W",
                    "Home Team",
                    "Away Team",
                    "Game Date",
                    "Home Pt Diff",
                    "Hm_Pts",
                    "Aw_Pts",
                    "Hm +7",
                    "Hm +3",
                    "Hm -3",
                    "Hm -7",
                    "Home Elo",
                    "Away Elo",
                    # # excluded and put in as dummy variable
                    # "Aw_TmDiv",
                    # "Hm_TmDiv",
                    f"{target_variable}",
                ]
            )
            model_X = X.drop(columns=["Matchup"])

        """
            FINDING best features
            features = model_X.columns.tolist()
            f_statistic, p_values = f_classif(model_X, model_y)

            feat_df = pd.DataFrame(data={
                'features': features,
                'f_stat': f_statistic,
                'p_values': p_values,
            })
        """

        """
            Run Model
        """
        logger.info(f"commence model run... NOW")
        # in order to predict probability of attendance use "model.predict_proba()"

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            model_X.iloc[1:],
            model_y.iloc[1:],
            test_size=0.2,
            random_state=14,
        )

        # call model with parameters
        if target_variable in ["Home W"]:
            model = KNeighborsClassifier(
                n_neighbors=100,
                weights="distance",
                metric="cityblock",
                p=1,
            )
            model.fit(X_train, np.ravel(y_train))

        elif target_variable in ["Home Pt Margin"]:
            model = KNeighborsClassifier(
                n_neighbors=25,
                weights="uniform",
                metric="cityblock",
            )
            model.fit(X_train, np.ravel(y_train))

        else:
            model = KNeighborsRegressor(
                n_neighbors=55,
                weights="distance",
                metric="cityblock",
            )
            model.fit(X_train, np.ravel(y_train))

        # suppress scientific notation
        # logger.info(f"predicting test set")
        np.set_printoptions(suppress=True)
        predictions = model.predict(X_test)
        if target_variable in ["Home W"]:
            predict_prob = model.predict_proba(X_test)

        # roll through prediction probabilities and give me the probability of attending the game
        #   i.e. the second number of the array for each index
        w_prob = []
        w_pred = []
        for i in range(len(predict_prob)):
            tmp_pred = predictions[i]
            w_pred += [tmp_pred]

            if target_variable in ["Home W"]:
                tmp_prob = predict_prob[i]
                w_prob += [tmp_prob[1]]

        # view results in df (with error)
        tmp_df = pd.DataFrame(
            columns=[
                "Matchup",
                "Game Date",
                "Home Team",
                "Away Team",
                "Home W",
                "Predict",
                "Predict Probability",
            ]
        )

        # dynamically get test data info for final df
        indexes = y_test.index.tolist()
        mylist = []
        for x in indexes:
            mylist += [x]

        tmp_df["Matchup"] = model_df.iloc[mylist]["Matchup"]
        tmp_df["Game Date"] = model_df.iloc[mylist]["Game Date"]
        tmp_df["Home Team"] = model_df.iloc[mylist]["Home Team"]
        tmp_df["Away Team"] = model_df.iloc[mylist]["Away Team"]
        tmp_df["Home W"] = model_df.iloc[mylist]["Home W"]

        tmp_df["Predict"] = w_pred
        if target_variable in ["Home W"]:
            tmp_df["Predict Probability"] = w_prob

        # model statistics
        if target_variable in ["Home W"]:
            model_stats_df = pd.DataFrame(
                columns=[
                    "Target Variable",
                    "R^2",
                    "RMSE",
                    "Recall Score",
                    "Precision Score",
                    "ROC AUC",
                    "F1 Score",
                ]
            )
            model_stats_df["Target Variable"] = [target_variable]
            model_stats_df["R^2"] = [model.score(X_test, y_test)]
            model_stats_df["RMSE"] = [
                np.sqrt(mean_squared_error(y_test, predictions))
            ]
            model_stats_df["Recall Score"] = [
                recall_score(y_test, predictions)
            ]  # tp / (tp + fn)
            model_stats_df["Precision Score"] = [
                precision_score(y_test, predictions)
            ]  # tp / (tp + fp)
            model_stats_df["ROC AUC"] = [roc_auc_score(y_test, w_prob)]
            model_stats_df["F1 Score"] = [f1_score(y_test, predictions)]
        else:
            model_stats_df = pd.DataFrame(
                columns=[
                    "Target Variable",
                    "R^2",
                ]
            )
            model_stats_df["Target Variable"] = [target_variable]
            model_stats_df["R^2"] = [model.score(X_test, y_test)]
            model_stats_df["RMSE"] = [np.sqrt(mean_squared_error(y_test, predictions))]

        """
            Confusion Matrix for Train/Test

        matrix = confusion_matrix(y_test, predictions)
        matrix_viz = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
        matrix_viz.plot()
        plt.show()
        """

        """
            Matchup Prediction (probability of home team winning)
        """
        logger.info(f"predicting {target_variable}: {matchup_data["Matchup"][0]}")

        if target_variable in ['Home W', 'Home Pt Diff']:
            prediction_df = matchup_data[
            [
                "Aw_PassOffEff",
                "Aw_Pts",
                "Aw_PassCmppct",
                "Aw_PassAdjYdsAtt",
                "Aw_PassTDG",
                "Aw_Dwn3Conv",
                "Aw_OppRushYdsAtt",
                "Aw_OppTOVG",
                "Aw_OppDwn3Conv",
                "Aw_TmLuck",
                "Aw_TmDiv_NFC West",
                "Hm_W",
                "Hm_Poss",
                "Hm_PassOffEff",
                "Hm_RushOffEff",
                "Hm_TmLuckW",
                "Hm_Pts",
                "Hm_PassCmppct",
                "Hm_PassAdjYdsAtt",
                "Hm_RushYdsAtt",
                "Hm_RushTDG",
                "Hm_Dwn3Conv",
                "Hm_OppPassAdjYdsAtt",
                "Hm_OppTOVG",
                "Hm_OppDwn3Conv",
                "Hm_TmLuck",
                "Hm_TmDiv_AFC North",
                "Hm_TmDiv_NFC South",
                "Elo_diff",
            ]
        ]
        else:
            prediction_df = matchup_data[
                [
                    "Aw_PassOffEff",
                    "Aw_PassCmppct",
                    "Aw_PassAdjYdsAtt",
                    "Aw_PassTDG",
                    "Aw_Dwn3Conv",
                    "Aw_OppRushYdsAtt",
                    "Aw_OppTOVG",
                    "Aw_OppDwn3Conv",
                    "Aw_TmLuck",
                    "Aw_TmDiv_NFC West",
                    "Hm_W",
                    "Hm_Poss",
                    "Hm_PassOffEff",
                    "Hm_RushOffEff",
                    "Hm_TmLuckW",
                    "Hm_PassCmppct",
                    "Hm_PassAdjYdsAtt",
                    "Hm_RushYdsAtt",
                    "Hm_RushTDG",
                    "Hm_Dwn3Conv",
                    "Hm_OppPassAdjYdsAtt",
                    "Hm_OppTOVG",
                    "Hm_OppDwn3Conv",
                    "Hm_TmLuck",
                    "Hm_TmDiv_AFC North",
                    "Hm_TmDiv_NFC South",
                    "Elo_diff",
                ]
            ]
        # prediction_df = prediction_df[limit_X_train.feature.tolist()]

        if target_variable in ["Home W"]:

            prediction_prob = model.predict_proba(prediction_df)
        prediction = model.predict(prediction_df)

        # roll through prediction probabilities and give me the probability of attending the game
        #   i.e. the second number of the array for each index
        w_prob = []
        w_pred = []
        for i in range(len(prediction_prob)):
            tmp_pred = prediction[i]
            w_pred += [tmp_pred]

            if target_variable in ["Home W"]:
                tmp_prob = prediction_prob[i]
                w_prob += [tmp_prob[1]]

        # view results in df
        pred_df = pd.DataFrame(
            columns=[
                "Matchup",
                "Game Date",
                "Home Team",
                "Away Team",
                f"Predict",
                f"Predict Probability",
            ]
        )

        # dynamically get test data info for final df
        indexes = prediction_df.index.tolist()
        mylist = []
        for x in indexes:
            mylist += [x]

        pred_df["Matchup"] = matchup_data.iloc[mylist]["Matchup"]
        pred_df["Game Date"] = matchup_data.iloc[mylist]["Game Date"]
        pred_df["Home Team"] = matchup_data.iloc[mylist]["Hm_Tm"]
        pred_df["Away Team"] = matchup_data.iloc[mylist]["Aw_Tm"]

        pred_df[f"Predict"] = w_pred

        if target_variable in ["Home W"]:
            pred_df[f"Predict Probability"] = w_prob

            # round the probability variable
            pred_df["Predict Probability"] = np.round(pred_df["Predict Probability"], 4)

        if final_pred_df.empty:
            final_pred_df = pred_df.rename(
                columns={
                    "Predict": f"{target_variable}",
                    "Predict Probability": f"{target_variable} Probability",
                }
            )
        elif target_variable in ["Home W"]:
            final_pred_df = final_pred_df.merge(
                pred_df[["Matchup", "Predict", "Predict Probability"]],
                how="inner",
                on="Matchup",
            ).rename(
                columns={
                    "Predict": f"{target_variable}",
                    "Predict Probability": f"{target_variable} Probability",
                }
            )
        else:
            final_pred_df = final_pred_df.merge(
                pred_df[["Matchup", "Predict"]], how="inner", on="Matchup"
            ).rename(columns={"Predict": f"{target_variable}"})

        # concat model_stats_df
        final_model_stats = pd.concat([final_model_stats, model_stats_df]).reset_index(
            drop=True
        )

    # format a df to fit the donut chart
    sg_win = pd.DataFrame(
        data={
            "Tm": [
                final_pred_df["Away Team"][0],
                final_pred_df["Home Team"][0],
            ],
            "Win Prob.": [
                1 - final_pred_df["Home W Probability"][0],
                final_pred_df["Home W Probability"][0],
            ],
            "Point Diff": [
                final_pred_df["Home Pt Diff"][0] * -1,
                final_pred_df["Home Pt Diff"][0],
            ],
            "Pred. Pts": [
                round(final_pred_df['Aw_Pts'][0], 0),
                round(final_pred_df['Hm_Pts'][0], 0),
            ]
        }
    )

    return [final_pred_df, final_model_stats, final_model_coef, sg_win]


def sim_donut_graph(season, away_tm, home_tm, sim_results_df, hm_tm_prim, aw_tm_prim):

    team_df = get_teamnm()

    away_abbr = team_df[(team_df["Tm Abbrv"] == away_tm)].reset_index(drop=True)[
        "Tm Name"
    ][0]
    home_abbr = team_df[(team_df["Tm Abbrv"] == home_tm)].reset_index(drop=True)[
        "Tm Name"
    ][0]

    # away_tm_color = team_df[
    #     (team_df["Tm Name"] == away_tm)
    # ].reset_index(drop=True)["Tm Primary Color"][0]
    # home_tm_color = team_df[
    #     (team_df["Tm Name"] == home_tm)
    # ].reset_index(drop=True)["Tm Primary Color"][0]

    stdev = 10

    if sim_results_df["Win Prob."][0] > sim_results_df["Win Prob."][1]:
        gm_winner = sim_results_df["Tm"][0]
        pt_spread = abs(sim_results_df["Point Diff"][0])
        away_win_prob = sim_results_df["Win Prob."][0]
        home_win_prob = sim_results_df["Win Prob."][1]
    else:
        gm_winner = sim_results_df["Tm"][1]
        pt_spread = abs(sim_results_df["Point Diff"][1])
        away_win_prob = sim_results_df["Win Prob."][0]
        home_win_prob = sim_results_df["Win Prob."][1]
    
    away_score = sim_results_df['Pred. Pts'][0]
    home_score = sim_results_df['Pred. Pts'][1]

    sim_results = [gm_winner, pt_spread, away_win_prob, home_win_prob]
    win_prob = [away_win_prob, home_win_prob]

    # relabel margin of victory for image
    # if sim_results[1] == 21:
    #     mov = "21+"
    # elif sim_results[1] == 14:
    #     mov = "14-20"
    # elif sim_results[1] == 10:
    #     mov = "10-13"
    # elif sim_results[1] == 7:
    #     mov = "7-9"
    # elif sim_results[1] == 3:
    #     mov = "3-6"
    # elif sim_results[1] == 0:
    #     mov = "1-2"
    mov = sim_results[1]

    home_tm_color_prim = get_teamcolor_prim(home_tm)
    home_tm_color_sec = get_teamcolor_sec(home_tm)
    away_tm_color_prim = get_teamcolor_prim(away_tm)
    away_tm_color_sec = get_teamcolor_sec(away_tm)

    if hm_tm_prim:
        home_tm_color = home_tm_color_prim
    else:
        home_tm_color = home_tm_color_sec
    if aw_tm_prim:
        away_tm_color = away_tm_color_prim
    else:
        away_tm_color = away_tm_color_sec
    game_colors = [away_tm_color, home_tm_color]

    # explosions
    explode = [0.05, 0.05]

    # Add team logos
    try:
        hm_img = Image.open(
            f"C:/Users/bcallahan/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/nfl/logos/helmets/{home_tm} L.png"
        )
        hm_img_array = np.array(hm_img)
        aw_img = Image.open(
            f"C:/Users/bcallahan/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/nfl/logos/helmets/{away_tm} R.png"
        )
        aw_img_array = np.array(aw_img)
    except:
        pass

    # Pie Chart
    wedges, texts, autotexts = plt.pie(
        win_prob,
        # labels=[away_tm, home_tm],
        colors=game_colors,
        textprops={
            "color": "white",
            "fontsize": 11,
        },
        autopct="%1.1f%%",
        pctdistance=0.82,
        explode=explode,
        wedgeprops={
            "linewidth": 1,
            "edgecolor": "000000",
            "width": 0.8,
        },
        startangle=90,
    )
    # set different colors for the label text
    for i, text in enumerate(texts):
        text.set_color(game_colors[i])  # Set label color to match the wedge color

    # set color for the percentage text (autopct)
    for autotext in autotexts:
        autotext.set_color("white")  # Example: set percentage text to white

    # draw circle
    center_circle = plt.Circle((0, 0), 0.7, fc="white")
    fig = plt.gcf()

    # adding circle in Pie Chart
    fig.gca().add_artist(center_circle)

    # add game information to center of chart
    plt.text(
        0,
        0,
        f"Location: @ {home_tm}\n Score Prediction:\n\n {away_tm}: {int(away_score)}\n {home_tm}: {int(home_score)}",
        # f"Location: @ {home_abbr}",
        ha="center",
        va="center",
        fontsize=11,
    )

    # add Legends
    plt.legend([away_abbr, home_abbr], loc="upper right")

    # add team logos
    ## extent = [left x, right x, lower y, upper y]
    plt.imshow(aw_img, extent=[-1.65, -0.75, -1.25, -0.5])
    plt.imshow(hm_img, extent=[0.75, 1.65, -1.25, -0.5])

    # ensuring circle proportion
    plt.axis("equal")

    # title
    plt.title(f"{away_abbr} @ {home_abbr}\n", fontsize=14)
    plt.suptitle("Win Probability", x=0.5, y=0.92, fontsize=10)

    return plt.show()
