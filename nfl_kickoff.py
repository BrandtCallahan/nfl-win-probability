import pandas as pd
from datetime import datetime, timedelta
from single_game_setup.ml_singlegame_setup import *
from nfl_model_results.nfl_weekly_results_utils import *


"""
    Helper functions
"""


def refresh_data(season: int):

    team_df = get_teamnm()
    team_name_list = team_df["Tm Abbrv"].unique().tolist()

    today = datetime.now().strftime("%Y-%m-%d")

    # pull and save nfl odds data from NFL FastR
    save_nfl_odds()

    # pull data from football reference
    save_team_stats(season, team_name_list, today)

    # run game_results()
    game_results(season, True)

    # run season_data()
    season_data(season)

    return


def run_model(
    season: int,
    week: int,
    team_focus: str,
    num_seasons: int,
    visualize: bool,
):

    nfl_df = pd.DataFrame()
    nfl_model_stats = pd.DataFrame()

    # data season
    season_years = [season]
    for i in range(num_seasons):
        season_years += [season - (i + 1)]
    season_years.sort()

    # using nfl fast r data pull to automate the matchups
    nfl_fastr = nfl_odds(season)

    # get the week for all games
    nfl_fastr["week"] = nfl_fastr["game_id"].str.split("_", expand=True)[1]
    nfl_fastr = nfl_fastr.astype({"week": "int16"})

    # min date for each week
    nfl_fastr = nfl_fastr.merge(
        nfl_fastr.groupby(["week"]).agg(min_gamedate=("gameday", "min")).reset_index(),
        how="inner",
        on="week",
    )

    week_df = nfl_fastr[nfl_fastr["week"] == week].reset_index(drop=True)

    wk_matchup = []
    for i in range(len(week_df.index)):
        wk_matchup += [
            [
                week_df.iloc[i]["away_team"],
                week_df.iloc[i]["home_team"],
                week_df.iloc[i]["gameday"],
            ]
        ]

    if team_focus.upper() == "ALL":
        matchup = wk_matchup
    else:
        for gm in wk_matchup:
            if team_focus in gm:
                matchup = [gm]

    for n, gm in enumerate(matchup):
        # matchup
        away_tm = gm[0]
        home_tm = gm[1]
        today = gm[2]

        logger.info(f"Running {n+1}/{len(matchup)}: {away_tm} vs. {home_tm}")

        # model type
        model = single_game_model(
            data_seasons=season_years,
            today=today,
            matchup=f"{away_tm} vs. {home_tm}",
        )
        sg_win = model[3]

        # reformat df
        if sg_win["Win Prob."][0] > sg_win["Win Prob."][1]:
            # Away Tm win
            tmp_df = pd.DataFrame(
                data={
                    "Matchup": [f"{away_tm} vs. {home_tm}"],
                    "Away Pts": [sg_win["Pred. Pts"][0]],
                    "Home Pts": [sg_win["Pred. Pts"][1]],
                    "Total Pts": [sg_win["Pred. Pts"][0] + sg_win["Pred. Pts"][1]],
                    "Pt Diff": [abs(sg_win["Point Diff"][0])],
                    "Pred. W": [f"{away_tm}"],
                    "Win Prob.": [sg_win["Win Prob."][0]],
                    "Away Moneyline": [sg_win["M/L"][0]],
                    "Home Moneyline": [sg_win["M/L"][1]],
                    "Home Spread": [sg_win["Spread"][1]],
                }
            )
            if sg_win["Spread W"][0] > sg_win["Spread W"][1]:
                tmp_df["Pred. Spread W"] = f"{away_tm}"
                tmp_df["Spread W Prob."] = f"{round(sg_win['Spread W'][0], 3)}"
            else:
                tmp_df["Pred. Spread W"] = f"{home_tm}"
                tmp_df["Spread W Prob."] = f"{round(sg_win['Spread W'][1], 3)}"
        else:
            # Home Tm win
            tmp_df = pd.DataFrame(
                data={
                    "Matchup": [f"{away_tm} vs. {home_tm}"],
                    "Away Pts": [sg_win["Pred. Pts"][0]],
                    "Home Pts": [sg_win["Pred. Pts"][1]],
                    "Total Pts": [sg_win["Pred. Pts"][0] + sg_win["Pred. Pts"][1]],
                    "Pt Diff": [abs(sg_win["Point Diff"][1])],
                    "Pred. W": [f"{home_tm}"],
                    "Win Prob.": [sg_win["Win Prob."][1]],
                    "Away Moneyline": [sg_win["M/L"][0]],
                    "Home Moneyline": [sg_win["M/L"][1]],
                    "Home Spread": [sg_win["Spread"][1]],
                }
            )
            if sg_win["Spread W"][0] > sg_win["Spread W"][1]:
                tmp_df["Pred. Spread W"] = f"{away_tm}"
                tmp_df["Spread W Prob."] = f"{round(sg_win['Spread W'][0], 3)}"
            else:
                tmp_df["Pred. Spread W"] = f"{home_tm}"
                tmp_df["Spread W Prob."] = f"{round(sg_win['Spread W'][1], 3)}"

        model_stats = model[1]

        # reformat model stats
        tmp_stats = pd.DataFrame(
            data={
                "Matchup": [f"{away_tm} vs. {home_tm}"],
                "W Exp. Accuracy": [round(model_stats["F1 Score"][0], 3)],
                "Spread Exp. Accuracy": [round(model_stats["F1 Score"][1], 3)],
                "Pt Diff Accuracy": [round(model_stats["R^2"][2], 3)],
                "Home Pts Accuracy": [round(model_stats["R^2"][3], 3)],
                "Home Pts +/-": [round(model_stats["RMSE"][3], 2)],
            }
        )

        if visualize:
            # donut chart for single game
            sim_donut_graph(
                season=season,
                away_tm=away_tm,
                home_tm=home_tm,
                sim_results_df=sg_win,
                hm_tm_prim=True,
                aw_tm_prim=True,
            )

        if nfl_df.empty:
            nfl_df = tmp_df.copy()
        else:
            nfl_df = pd.concat([nfl_df, tmp_df]).reset_index(drop=True)

        if nfl_model_stats.empty:
            nfl_model_stats = tmp_stats.copy()
        else:
            nfl_model_stats = pd.concat([nfl_model_stats, tmp_stats]).reset_index(
                drop=True
            )

    return [nfl_df, nfl_model_stats]


#############################################

"""
    To refresh the data for any year run this
"""
refresh_data(2025)


"""
    Run model for desired SEASON and WEEK
"""
nfl_df = run_model(
    season=2025,
    week=16,
    team_focus="TEN",
    num_seasons=10,
    visualize=True,
)

"""
    Save weekly results
"""
save_nfl_wk_model(
    season=2025,
    week=7,
    num_seasons=10,
)
