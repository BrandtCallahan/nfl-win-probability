import pandas as pd
from datetime import datetime, timedelta
from single_game_setup.ml_singlegame_setup import *
from nfl_model_results.nfl_weekly_results_utils import *
from single_game_setup.sim_singlegame_setup import *


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
    today = week_df["gameday"].min()
    for i in range(len(week_df.index)):
        wk_matchup += [
            [
                week_df.iloc[i]["away_team"],
                week_df.iloc[i]["home_team"],
            ]
        ]

    if team_focus.upper() != "ALL":
        for gm in wk_matchup:
            if team_focus in gm:
                wk_matchup = [gm]

    # run the model
    model = single_game_model(
        data_seasons=season_years,
        today=today,
        week=week,
        matchups=wk_matchup,
    )

    if (visualize) & (team_focus != "ALL"):
        # ensure Pt Diff reflects Hm value properly
        if model[0]["Home W"][0] == 0:
            if model[0]["Home Pt Diff"][0] > 0:
                model[0]["Home Pt Diff"][0] = model[0]["Home Pt Diff"][0] * -1
        else:
            if model[0]["Home Pt Diff"][0] < 0:
                model[0]["Home Pt Diff"][0] = model[0]["Home Pt Diff"][0] * -1

        # Tm Scores
        if model[0]["Home Pt Diff"][0] < 0:
            pts = (model[0]['Total Pts'][0]) + (model[0]['Home Pt Diff'][0])

            aw_points = pts / 2
            hm_points = (pts / 2) + (model[0]['Home Pt Diff'][0])
        else:
            pts = (model[0]["Total Pts"][0]) - (model[0]["Home Pt Diff"][0])

            aw_points = pts / 2
            hm_points = (pts / 2) + (model[0]["Home Pt Diff"][0])

        # format a df to fit the donut chart
        sg_win = pd.DataFrame(
            data={
                "Tm": [
                    model[0]["Away Team"][0],
                    model[0]["Home Team"][0],
                ],
                "Records": [
                    model[0]["Away Record"][0],
                    model[0]["Home Record"][0],
                ],
                "Win Prob.": [
                    1 - model[0]["Home W Probability"][0],
                    model[0]["Home W Probability"][0],
                ],
                "Point Diff": [
                    model[0]["Home Pt Diff"][0] * -1,
                    model[0]["Home Pt Diff"][0],
                ],
                "Pred. Pts": [
                    round(aw_points, 0),
                    round(hm_points, 0),
                ],
                "Spread W": [
                    1 - model[0]["Home Spread W Probability"][0],
                    model[0]["Home Spread W Probability"][0],
                ],
                "M/L": [
                    model[0]["Away Moneyline"][0],
                    model[0]["Home Moneyline"][0],
                ],
                "Spread": [
                    model[0]["Home Spread"][0] * -1,
                    model[0]["Home Spread"][0],
                ],
            }
        )

        # donut chart for single game
        sim_donut_graph(
            season=season,
            away_tm=sg_win["Tm"][0],
            home_tm=sg_win["Tm"][1],
            sim_results_df=sg_win,
            hm_tm_prim=True,
            aw_tm_prim=True,
        )

    return [model[0], model[1]]


def run_simulation(
    season: int,
    week: int,
    team_focus: str,
    num_seasons: int,
    visualize: bool,
):

    nfl_df = pd.DataFrame()

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
        sg_win = game_sim(
            data_seasons=season_years,
            away_tm=away_tm,
            home_tm=home_tm,
            neutral=False,
            today=today,
            n=5001,
        )

        # reformat df
        if sg_win["Win Prob."][0] > sg_win["Win Prob."][1]:
            # Away Tm win
            tmp_df = pd.DataFrame(
                data={
                    "Matchup": [f"{away_tm} vs. {home_tm}"],
                    "Away Pts": [math.ceil(sg_win["Pts Scored"][0])],
                    "Home Pts": [math.ceil(sg_win["Pts Scored"][1])],
                    "Total Pts": [
                        math.ceil(sg_win["Pts Scored"][0])
                        + math.ceil(sg_win["Pts Scored"][1])
                    ],
                    "Pt Diff": [abs(sg_win["Point Diff"][0])],
                    "Pred. W": [f"{away_tm}"],
                    "Win Prob.": [sg_win["Win Prob."][0]],
                    "Away Moneyline": [sg_win["M/L"][0]],
                    "Home Moneyline": [sg_win["M/L"][1]],
                    "Home Spread": [sg_win["Spread"][1]],
                }
            )
            if sg_win["Spread"][1] < 0:
                # Home Favorite
                tmp_df["Pred. Spread W"] = f"{away_tm}"
            else:
                # Away Favorite
                if sg_win["Point Diff"][1] > sg_win["Spread"][1]:
                    tmp_df["Pred. Spread W"] = f"{home_tm}"
                else:
                    tmp_df["Pred. Spread W"] = f"{away_tm}"
        else:
            # Home Tm win
            tmp_df = pd.DataFrame(
                data={
                    "Matchup": [f"{away_tm} vs. {home_tm}"],
                    "Away Pts": [math.ceil(sg_win["Pts Scored"][0])],
                    "Home Pts": [math.ceil(sg_win["Pts Scored"][1])],
                    "Total Pts": [
                        math.ceil(sg_win["Pts Scored"][0])
                        + math.ceil(sg_win["Pts Scored"][1])
                    ],
                    "Pt Diff": [abs(sg_win["Point Diff"][1])],
                    "Pred. W": [f"{home_tm}"],
                    "Win Prob.": [sg_win["Win Prob."][1]],
                    "Away Moneyline": [sg_win["M/L"][0]],
                    "Home Moneyline": [sg_win["M/L"][1]],
                    "Home Spread": [sg_win["Spread"][1]],
                }
            )
            if sg_win["Spread"][1] < 0:
                # Home Favorite
                if sg_win["Point Diff"][1] < sg_win["Spread"][1]:
                    tmp_df["Pred. Spread W"] = f"{home_tm}"
                else:
                    tmp_df["Pred. Spread W"] = f"{away_tm}"
            else:
                # Away Favorite
                tmp_df["Pred. Spread W"] = f"{home_tm}"

        if visualize:
            # donut chart for single game
            sim_donut_graph(
                season=season,
                away_tm=away_tm,
                home_tm=home_tm,
                sim_results_df=sg_win,
                hm_tm_prim=True,
                aw_tm_prim=False,
            )

        if nfl_df.empty:
            nfl_df = tmp_df.copy()
        else:
            nfl_df = pd.concat([nfl_df, tmp_df]).reset_index(drop=True)

    return nfl_df


#############################################

"""
    To refresh the data for any year run this
"""
refresh_data(2025)


"""
    Run predictive model for desired SEASON and WEEK
        - team_focus == 'ALL' for all games that week
        - team_focus == '{insert team abbrv}' for only that team's game
"""
nfl_df = run_model(
    season=2025,
    week=18,
    team_focus="TEN",
    num_seasons=10,
    visualize=True,
)

"""
    Run simulation for desired SEASON and WEEK
        - team_focus == 'ALL' for all games that week
        - team_focus == '{insert team abbrv}' for only that team's game
"""
nfl_df = run_simulation(
    season=2025,
    week=16,
    team_focus="TEN",
    num_seasons=10,
    visualize=True,
)

"""
    Save weekly results
    TODO: update nfl_model_results folder
"""
save_nfl_wk_model(
    season=2025,
    week=7,
    num_seasons=10,
)
