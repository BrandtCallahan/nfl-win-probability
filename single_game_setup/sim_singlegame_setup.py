from matplotlib.pylab import norm
import pandas as pd
import numpy as np
import pylab as p
import random
import math
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

from utils.get_data import *
from utils.team_dict import *
from utils.beautiful_soup_helper import *

pd.set_option("future.no_silent_downcasting", True)


def pregame(season, away_tm, home_tm, today):

    team_df = get_teamnm()
    hm_df = team_df[team_df["Tm Abbrv"] == home_tm].reset_index(drop=True)
    aw_df = team_df[team_df["Tm Abbrv"] == away_tm].reset_index(drop=True)

    # pull in team data
    hm_gmlog = gamelog_setup(season, home_tm, today)
    aw_gmlog = gamelog_setup(season, away_tm, today)

    # filter out for select teams
    tm_gamelog = pd.concat([hm_gmlog, aw_gmlog])
    tm_gamelog = tm_gamelog.fillna(
        {
            "FG%": 0,
            "Opp FG%": 0,
        }
    )

    # calculate scoring plays (Pass TD, Rush TD, FGM)
    tm_gamelog["ScoringPlays"] = (
        tm_gamelog["Pass TD"] + tm_gamelog["Rush TD"] + tm_gamelog["FGM"]
    )
    tm_gamelog["OppScoringPlays"] = (
        tm_gamelog["Opp Pass TD"] + tm_gamelog["Opp Rush TD"] + tm_gamelog["Opp FGM"]
    )

    # calculate losses
    tm_gamelog["L"] = tm_gamelog["G"] - tm_gamelog["Total W"]

    # season group by
    tm_season = (
        tm_gamelog.groupby(["Tm"], observed=True)
        .agg(
            G=("G", "max"),
            TmPts=("Tm Pts", "sum"),
            AvgTmPts=("Tm Pts", "mean"),
            TotScorePlays=("ScoringPlays", "sum"),
            OppPts=("Opp Pts", "sum"),
            AvgOppPts=("Opp Pts", "sum"),
            TotOppScorePlays=("OppScoringPlays", "sum"),
            # Efficiencies
            AvgPoss=("Poss", "mean"),
            AvgOffEff=("Off Eff", "mean"),
            AvgDefEff=("Def Eff", "mean"),
            # Box Score Stats
            FG=("FGM", "sum"),
            FGA=("FGA", "sum"),
            XP=("XPM", "sum"),
            XPA=("XPA", "sum"),
            RushYA=("Rush Y/A", "mean"),
            PassYA=("Pass Adj Y/A", "mean"),
            PassRate=("Pass Rate", "mean"),
            Sacks=("Sacks", "mean"),
            PenYds=("Pen Yds", "mean"),
            Dwn3Conv=("3Dwn Conv %", "mean"),
            TOV=("Tot TO", "sum"),
            Dwn4Att=("4th Dwn Att", "sum"),
            Dwn4Conv=("4th Dwn Conv", "sum"),
            PuntAtt=("Punt Att", "sum"),
            PassTD=("Pass TD", "sum"),
            RushTD=("Rush TD", "sum"),
            OppFG=("Opp FGM", "sum"),
            OppFGA=("Opp FGA", "sum"),
            OppXP=("Opp XPM", "sum"),
            OppXPA=("Opp XPA", "sum"),
            OppRushYA=("Opp Rush Y/A", "mean"),
            OppPassYA=("Opp Pass Adj Y/A", "mean"),
            OppPassRate=("Opp Pass Rate", "mean"),
            OppSacks=("Opp Sacks", "mean"),
            OppPenYds=("Opp Pen Yds", "mean"),
            OppDwn3Conv=("Opp 3Dwn Conv %", "mean"),
            OppTOV=("Opp Tot TO", "sum"),
            OppDwn4Att=("Opp 4th Dwn Att", "sum"),
            OppDwn4Conv=("Opp 4th Dwn Conv", "sum"),
            OppPuntAtt=("Opp Punt Att", "sum"),
            OppPassTD=("Opp Pass TD", "sum"),
            OppRushTD=("Opp Rush TD", "sum"),
            # Final Stats
            TOVpm=("TO +/-", "sum"),
            WStreak=("Streak +/-", "last"),
            OppSOS=("Opp SOS", "mean"),
            W=("Total W", "max"),
            L=("L", "max"),
            Luck=("Tm Luck", "sum"),
        )
        .reset_index()
    )

    # team records
    tm_season["Records"] = (
        tm_season["W"].astype(int).astype(str)
        + "-"
        + tm_season["L"].astype(int).astype(str)
    )

    tm_season["FG%"] = tm_season["FG"] / tm_season["FGA"]
    tm_season["XP%"] = tm_season["XP"] / tm_season["XPA"]

    # % of Total Pts as TD or FG (7 vs. 3 Pts)
    tm_season["7pt%"] = (tm_season["PassTD"] + tm_season["RushTD"]) / (
        tm_season["PassTD"] + tm_season["RushTD"] + tm_season["FG"]
    )
    tm_season["3pt%"] = 1 - tm_season["7pt%"]

    tm_season["LuckFactor"] = tm_season["W"] - tm_season["Luck"]

    tm_season["Poss"] = (
        tm_season["TOV"].astype(int)
        + tm_season["PuntAtt"].astype(int)
        + tm_season["PassTD"].astype(int)
        + tm_season["RushTD"].astype(int)
        + tm_season["FGA"].astype(int)
        + (tm_season["Dwn4Att"].astype(int) - tm_season["Dwn4Conv"].astype(int))
    )
    tm_season["OppPoss"] = (
        tm_season["OppTOV"].astype(int)
        + tm_season["OppPuntAtt"].astype(int)
        + tm_season["OppPassTD"].astype(int)
        + tm_season["OppRushTD"].astype(int)
        + tm_season["OppFGA"].astype(int)
        + (tm_season["OppDwn4Att"].astype(int) - tm_season["OppDwn4Conv"].astype(int))
    )

    tm_season["OffEff"] = ((tm_season["TotScorePlays"] / tm_season["Poss"])) * (
        1 + (tm_season["LuckFactor"] / tm_season["G"])
    )
    tm_season["DefEff"] = ((tm_season["TotOppScorePlays"] / tm_season["OppPoss"])) * (
        1 + ((1 - tm_season["LuckFactor"]) / tm_season["G"])
    )

    # recent group by (4 games)
    tm_recent = pd.DataFrame()
    for tm in [away_tm, home_tm]:
        recent = (
            tm_gamelog[tm_gamelog["Tm"] == tm]
            .sort_values(by="G")
            .reset_index(drop=True)
        )
        recent = recent.iloc[-4:]

        tm_recent = pd.concat([tm_recent, recent]).reset_index(drop=True)

    tm_recency = (
        tm_recent.groupby(["Tm"], observed=True)
        .agg(
            G=("G", "count"),
            TmPts=("Tm Pts", "sum"),
            AvgTmPts=("Tm Pts", "mean"),
            TotScorePlays=("ScoringPlays", "sum"),
            OppPts=("Opp Pts", "sum"),
            AvgOppPts=("Opp Pts", "mean"),
            TotOppScorePlays=("OppScoringPlays", "sum"),
            # Efficiencies
            AvgPoss=("Poss", "mean"),
            AvgOffEff=("Off Eff", "mean"),
            AvgDefEff=("Def Eff", "mean"),
            # Box Score Stats
            FG=("FGM", "sum"),
            FGA=("FGA", "sum"),
            XP=("XPM", "sum"),
            XPA=("XPA", "sum"),
            RushYA=("Rush Y/A", "mean"),
            PassYA=("Pass Adj Y/A", "mean"),
            PassRate=("Pass Rate", "mean"),
            Sacks=("Sacks", "mean"),
            PenYds=("Pen Yds", "mean"),
            Dwn3Conv=("3Dwn Conv %", "mean"),
            TOV=("Tot TO", "sum"),
            Dwn4Att=("4th Dwn Att", "sum"),
            Dwn4Conv=("4th Dwn Conv", "sum"),
            PuntAtt=("Punt Att", "sum"),
            PassTD=("Pass TD", "sum"),
            RushTD=("Rush TD", "sum"),
            OppFG=("Opp FGM", "sum"),
            OppFGA=("Opp FGA", "sum"),
            OppXP=("Opp XPM", "sum"),
            OppXPA=("Opp XPA", "sum"),
            OppRushYA=("Opp Rush Y/A", "mean"),
            OppPassYA=("Opp Pass Adj Y/A", "mean"),
            OppPassRate=("Opp Pass Rate", "mean"),
            OppSacks=("Opp Sacks", "mean"),
            OppPenYds=("Opp Pen Yds", "mean"),
            OppDwn3Conv=("Opp 3Dwn Conv %", "mean"),
            OppTOV=("Opp Tot TO", "sum"),
            OppDwn4Att=("Opp 4th Dwn Att", "sum"),
            OppDwn4Conv=("Opp 4th Dwn Conv", "sum"),
            OppPuntAtt=("Opp Punt Att", "sum"),
            OppPassTD=("Opp Pass TD", "sum"),
            OppRushTD=("Opp Rush TD", "sum"),
            # Final Stats
            TOVpm=("TO +/-", "sum"),
            WStreak=("Streak +/-", "last"),
            OppSOS=("Opp SOS", "mean"),
            W=("Total W", "max"),
            Luck=("Tm Luck", "sum"),
        )
        .reset_index()
    )

    tm_recency["FG%"] = tm_recency["FG"] / tm_recency["FGA"]
    tm_recency["XP%"] = tm_recency["XP"] / tm_recency["XPA"]

    # % of Total Pts as TD or FG (7 vs. 3 Pts)
    tm_recency["7pt%"] = (tm_recency["PassTD"] + tm_recency["RushTD"]) / (
        tm_recency["PassTD"] + tm_recency["RushTD"] + tm_recency["FG"]
    )
    tm_recency["3pt%"] = 1 - tm_recency["7pt%"]

    tm_recency["LuckFactor"] = tm_recency["W"] - tm_recency["Luck"]

    tm_recency["Poss"] = (
        tm_recency["TOV"].astype(int)
        + tm_recency["PuntAtt"].astype(int)
        + tm_recency["PassTD"].astype(int)
        + tm_recency["RushTD"].astype(int)
        + tm_recency["FGA"].astype(int)
        + (tm_recency["Dwn4Att"].astype(int) - tm_recency["Dwn4Conv"].astype(int))
    )
    tm_recency["OppPoss"] = (
        tm_recency["OppTOV"].astype(int)
        + tm_recency["OppPuntAtt"].astype(int)
        + tm_recency["OppPassTD"].astype(int)
        + tm_recency["OppRushTD"].astype(int)
        + tm_recency["OppFGA"].astype(int)
        + (tm_recency["OppDwn4Att"].astype(int) - tm_recency["OppDwn4Conv"].astype(int))
    )

    tm_recency["OffEff"] = ((tm_recency["TotScorePlays"] / tm_recency["Poss"])) * (
        1 + (tm_recency["LuckFactor"] / tm_recency["G"])
    )
    tm_recency["DefEff"] = (
        (tm_recency["TotOppScorePlays"] / tm_recency["OppPoss"])
    ) * (1 + ((1 - tm_recency["LuckFactor"]) / tm_recency["G"]))

    # combine season with recency (weighting recency slightly)
    season_wght = 10
    recency_wght = 0
    tm_df = pd.DataFrame(
        data={
            "Tm": tm_season["Tm"],
            "G": tm_season["G"],
            "Record": tm_season["Records"],
            "Poss": (
                ((tm_season["Poss"] / tm_season["G"]) * season_wght)
                + ((tm_recency["Poss"] / tm_recency["G"]) * recency_wght)
            )
            / 10,
            "OppPoss": (
                ((tm_season["OppPoss"] / tm_season["G"]) * season_wght)
                + ((tm_recency["OppPoss"] / tm_recency["G"]) * recency_wght)
            )
            / 10,
            "OffEff": (
                (tm_season["OffEff"] * season_wght)
                + (tm_recency["OffEff"] * recency_wght)
            )
            / 10,
            "DefEff": (
                (tm_season["DefEff"] * season_wght)
                + (tm_recency["DefEff"] * recency_wght)
            )
            / 10,
            "OppSOS": (
                (tm_season["OppSOS"] * season_wght)
                + (tm_recency["OppSOS"] * recency_wght)
            )
            / 10,
            "PassRate": (
                (tm_season["PassRate"] * season_wght)
                + (tm_recency["PassRate"] * recency_wght)
            )
            / 10,
            "OppPassRate": (
                (tm_season["OppPassRate"] * season_wght)
                + (tm_recency["OppPassRate"] * recency_wght)
            )
            / 10,
            "FG%": (
                (tm_season["FG%"] * season_wght) + (tm_recency["FG%"] * recency_wght)
            )
            / 10,
            "7pt%": (
                (tm_season["7pt%"] * season_wght) + (tm_recency["7pt%"] * recency_wght)
            )
            / 10,
            "3pt%": (
                (tm_season["3pt%"] * season_wght) + (tm_recency["3pt%"] * recency_wght)
            )
            / 10,
        },
    )

    # need to sort df with away team in [0] and home team in [1]
    tm_dict = {
        f"{away_tm}": 0,
        f"{home_tm}": 1,
    }
    tm_df["Sort Rnk"] = tm_df["Tm"].map(tm_dict)
    tm_df = (
        tm_df.sort_values(by="Sort Rnk")
        .drop(columns=["Sort Rnk"])
        .reset_index(drop=True)
    )

    # gambling lines for current matchup
    odds_df = nfl_odds(season).reset_index(drop=True)
    odds_df = odds_df[
        (odds_df["matchup"] == f"{tm_df['Tm'].values[0]} vs. {tm_df['Tm'].values[1]}")
        & (
            (pd.to_datetime(odds_df["gameday"].astype(object)).astype(str))
            == str(pd.to_datetime(today).strftime("%Y-%m-%d"))
        )
    ].reset_index(drop=True)

    tm_df = pd.concat(
        [
            tm_df,
            pd.DataFrame(
                data={
                    "M/L": [
                        odds_df["away_moneyline"].values[0],
                        odds_df["home_moneyline"].values[0],
                    ],
                    "Spread": [
                        odds_df["hm_spread"].values[0] * -1,
                        odds_df["hm_spread"].values[0],
                    ],
                    "Spread Odds": [
                        odds_df["away_spread_odds"].values[0],
                        odds_df["home_spread_odds"].values[0],
                    ],
                }
            ),
        ],
        axis=1,
    )

    tm_df = tm_df.astype(
        {
            "M/L": "int16",
            "Spread Odds": "int16",
        }
    )

    return tm_df


def game(tm_df, neutral):
    score_df = pd.DataFrame()
    home_tm = tm_df["Tm"][1]

    # simulate a game
    for tm in tm_df["Tm"]:
        tm_possessions = math.ceil(
            (tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["Poss"][0])
        )
        opp_possessions = math.ceil(
            (tm_df[~(tm_df["Tm"] == tm)].reset_index(drop=True)["OppPoss"][0])
        )

        possessions = math.ceil(((tm_possessions) + (opp_possessions)) / 2)
        eff = 1

        score = 0
        while possessions > 0:
            # drive
            drive = random.random()

            # points scored
            if drive <= (tm_df[tm_df["Tm"] == tm]["OffEff"].values[0]):

                if (
                    1 == 1
                ):  ##  drive <= (tm_df[~(tm_df["Tm"] == tm)]["DefEff"].values[0]):
                    points = random.random()

                    # 7 Pt
                    if points <= tm_df[tm_df["Tm"] == tm]["7pt%"].values[0]:
                        score += 7
                    # 3 Pt
                    elif points > tm_df[tm_df["Tm"] == tm]["7pt%"].values[0]:
                        # possible missed field goal (if above FG%)
                        fg = random.random()
                        if fg <= tm_df[tm_df["Tm"] == tm]["FG%"].values[0]:
                            score += 3
                else:
                    score += 0

            # drive resulted in no points
            else:
                score += 0

            possessions += -1

        if neutral == False:
            if tm == home_tm:
                score = (score * eff) + 2.5
        else:
            score = score * eff

        # input into final df
        score_df = pd.concat(
            [score_df, pd.DataFrame(data={"Tm": [tm], "Score": [math.ceil(score)]})]
        ).reset_index(drop=True)

    return score_df


def game_sim(season, away_tm, home_tm, today, neutral, n):

    pregame_df = pregame(season, away_tm, home_tm, today)

    results_df = pd.DataFrame()
    for i in range(n):

        # play the game
        game_df = game(pregame_df, neutral)

        # tally results
        if game_df["Score"][0] >= game_df["Score"][1]:
            winner = game_df["Tm"][0]  # home team wins
            pt_spread = game_df["Score"][0] - game_df["Score"][1]
        else:
            winner = game_df["Tm"][1]  # away team wins
            pt_spread = game_df["Score"][1] - game_df["Score"][0]

        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    data={
                        "Game": [i + 1],
                        "Winner": [winner],
                        "Point Spread": [pt_spread],
                        "Away Pts": [game_df["Score"][0]],
                        "Home Pts": [game_df["Score"][1]],
                    }
                ),
            ]
        ).reset_index(drop=True)

    # find winner from monte carlo
    home_tm_w = results_df["Winner"].str.count(f"{home_tm.split("(")[0]}").sum()
    away_tm_w = results_df["Winner"].str.count(f"{away_tm.split("(")[0]}").sum()

    if home_tm_w > away_tm_w:
        game_winner = home_tm
        win_pct = home_tm_w / n
    else:
        game_winner = away_tm
        win_pct = away_tm_w / n

    # find average pts scored per team
    hm_pts_avg = results_df["Home Pts"].mean()
    aw_pts_avg = results_df["Away Pts"].mean()

    # find average score diff
    point_diff = results_df[results_df["Winner"] == game_winner]["Point Spread"].mean()
    if results_df[~(results_df["Winner"] == game_winner)]["Point Spread"].empty:
        opp_point_diff = 0
    else:
        opp_point_diff = results_df[~(results_df["Winner"] == game_winner)][
            "Point Spread"
        ].mean()

    tot_point_diff = point_diff - opp_point_diff

    if game_winner == away_tm:
        aw_point_diff = tot_point_diff * -1
        hm_point_diff = tot_point_diff
    else:
        aw_point_diff = tot_point_diff
        hm_point_diff = tot_point_diff * -1

    gm_results = pd.DataFrame(
        data={
            "Tm": [away_tm, home_tm],
            "Records": [pregame_df["Record"].values[0], pregame_df["Record"].values[1]],
            "Win Prob.": [away_tm_w / n, home_tm_w / n],
            "Point Diff": [aw_point_diff, hm_point_diff],
            "Pts Scored": [aw_pts_avg, hm_pts_avg],
            "M/L": [
                pregame_df["M/L"].values[0],
                pregame_df["M/L"].values[1],
            ],
            "Spread": [pregame_df["Spread"].values[0], pregame_df["Spread"].values[1]],
        }
    )

    return gm_results


def sim_donut_graph(season, away_tm, home_tm, sim_results_df, hm_tm_prim, aw_tm_prim):

    team_df = get_teamnm()

    away_abbr = team_df[(team_df["Tm Abbrv"] == away_tm)].reset_index(drop=True)[
        "Tm Name"
    ][0]
    home_abbr = team_df[(team_df["Tm Abbrv"] == home_tm)].reset_index(drop=True)[
        "Tm Name"
    ][0]

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

    away_score = sim_results_df["Pts Scored"][0]
    home_score = sim_results_df["Pts Scored"][1]

    sim_results = [gm_winner, pt_spread, away_win_prob, home_win_prob]
    win_prob = [away_win_prob, home_win_prob]

    mov = sim_results[1]

    # team records
    hm_record = sim_results_df["Records"][1]
    aw_record = sim_results_df["Records"][0]

    # gambling lines
    spread = sim_results_df["Spread"][1]
    if spread < 0:
        spread = f"{spread}"
    elif spread > 0:
        spread = f"+{spread}"
    else:
        spread = "Even"

    aw_moneyline = int(sim_results_df["M/L"][0])
    if aw_moneyline > 0:
        aw_moneyline = f"+{aw_moneyline}"
    else:
        aw_moneyline = f"{aw_moneyline}"

    hm_moneyline = int(sim_results_df["M/L"][1])
    if hm_moneyline > 0:
        hm_moneyline = f"+{hm_moneyline}"
    else:
        hm_moneyline = f"{hm_moneyline}"

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
            f"C:/Users/{os.getlogin()}/personal-github/nfl-win-probability/logos/helmets/{home_tm} L.png"
        )
        hm_img_array = np.array(hm_img)
        aw_img = Image.open(
            f"C:/Users/{os.getlogin()}/personal-github/nfl-win-probability/logos/helmets/{away_tm} R.png"
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
        f"Location: @ {home_tm} ({spread})\n\n Total Pts: {int(round(away_score, 0)) + int(round(home_score, 0))}\n Margin of Victory: {math.ceil(pt_spread)}",
        ha="center",
        va="center",
        fontsize=11,
    )

    # add moneylines below team helmets
    ## home helmet
    plt.text(
        1.45,
        -1.25,
        f"{hm_moneyline}",
        ha="center",
        va="center",
        fontsize=11,
    )
    ## away helmet
    plt.text(
        -1.45,
        -1.25,
        f"{aw_moneyline}",
        ha="center",
        va="center",
        fontsize=11,
    )

    # add Legends
    plt.legend(
        [f"{away_abbr} ({aw_record})", f"{home_abbr} ({hm_record})"], loc="upper right"
    )

    # add team helmets/logos
    ## extent = [left x, right x, lower y, upper y]
    try:
        plt.imshow(aw_img, extent=[-1.65, -0.75, -1.25, -0.5])
        plt.imshow(hm_img, extent=[0.75, 1.65, -1.25, -0.5])
    except:
        pass

    # ensuring circle proportion
    plt.axis("equal")

    # title
    plt.title(f"{away_abbr} @ {home_abbr}\n", fontsize=14)
    plt.suptitle("Win Probability", x=0.5, y=0.92, fontsize=10)

    return plt.show()
