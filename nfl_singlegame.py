import pandas as pd
from datetime import datetime, timedelta
from single_game_setup.ml_singlegame_setup import *


"""
    Helper functions
"""


def refresh_data(season: int):

    team_df = get_teamnm()
    team_name_list = team_df["Tm Abbrv"].unique().tolist()

    today = datetime.now().strftime("%Y-%m-%d")

    # pull data from football reference
    save_team_stats(season, team_name_list, today)

    # run game_results()
    game_results(season, True)

    # run season_data()
    season_data(season)

    return


def run_model(
    season: int,
    today: datetime,
    num_seasons: int,
    matchup: list,
    visualize: bool,
):

    nfl_df = pd.DataFrame()
    nfl_model_stats = pd.DataFrame()

    # data season
    season_years = [season]
    for i in range(num_seasons):
        season_years += [season - (i + 1)]
    season_years.sort()

    for gm in matchup:
        # matchup
        away_tm = gm[0]
        home_tm = gm[1]

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
                aw_tm_prim=False,
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
    Set up starting with Week 5 matchups of 2025
"""
# Week 5
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-10-02").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["SFO", "LAR"],
        ["MIN", "CLE"],
        ["LVR", "IND"],
        ["NYG", "NOR"],
        ["DAL", "NYJ"],
        ["DEN", "PHI"],
        ["MIA", "CAR"],
        ["HOU", "BAL"],
        ["TEN", "ARI"],
        ["TAM", "SEA"],
        ["DET", "CIN"],
        ["WAS", "LAC"],
        ["NWE", "BUF"],
        ["KAN", "JAX"],
    ],
    visualize=False,
)

# Week 6
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-10-09").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["PHI", "NYG"],
        ["DEN", "NYJ"],
        ["ARI", "IND"],
        ["LAC", "MIA"],
        ["NWE", "NOR"],
        ["CLE", "PIT"],
        ["DAL", "CAR"],
        ["SEA", "JAX"],
        ["LAR", "BAL"],
        ["TEN", "LVR"],
        ["GNB", "CIN"],
        ["SFO", "TAM"],
        ["DET", "KAN"],
        ["BUF", "ATL"],
        ["CHI", "WAS"],
    ],
    visualize=False,
)

# Week 7
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-10-15").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["PIT", "CIN"],
        ["LAR", "JAX"],
        ["NWE", "TEN"],
        ["NOR", "CHI"],
        ["MIA", "CLE"],
        ["LVR", "KAN"],
        ["PHI", "MIN"],
        ["NYG", "DEN"],
        ["IND", "LAC"],
        ["WAS", "DAL"],
        ["GNB", "ARI"],
        ["ATL", "SFO"],
        ["TAM", "DET"],
        ["HOU", "SEA"],
    ],
    visualize=False,
)

# Week 8
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-10-22").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["MIN", "LAC"],
        ["MIA", "ATL"],
        ["NYJ", "CIN"],
        ["CLE", "NWE"],
        ["NYG", "PHI"],
        ["BUF", "CAR"],
        ["CHI", "BAL"],
        ["SFO", "HOU"],
        ["TAM", "NOR"],
        ["TEN", "IND"],
        ["DAL", "DEN"],
        ["GNB", "PIT"],
        ["WAS", "KAN"],
    ],
    visualize=False,
)

# Week 9
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-10-29").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["BAL", "MIA"],
        ["LAC", "TEN"],
        ["CHI", "CIN"],
        ["MIN", "DET"],
        ["CAR", "GNB"],
        ["ATL", "NWE"],
        ["SFO", "NYG"],
        ["IND", "PIT"],
        ["DEN", "HOU"],
        ["JAX", "LVR"],
        ["NOR", "LAR"],
        ["KAN", "BUF"],
        ["SEA", "WAS"],
        ["ARI", "DAL"],
    ],
    visualize=False,
)

# Week 10
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-11-05").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["LVR", "DEN"],
        ["ATL", "IND"],
        ["NYG", "CHI"],
        ["BUF", "MIA"],
        ["BAL", "MIN"],
        ["CLE", "NYJ"],
        ["NWE", "TAM"],
        ["NOR", "CAR"],
        ["JAX", "HOU"],
        ["ARI", "SEA"],
        ["LAR", "SFO"],
        ["DET", "WAS"],
        ["PIT", "LAC"],
        ["PHI", "GNB"],
    ],
    visualize=False,
)

# Week 11
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-11-12").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["NYJ", "NWE"],
        ["WAS", "MIA"],
        ["HOU", "TEN"],
        ["CAR", "ATL"],
        ["TAM", "BUF"],
        ["CHI", "MIN"],
        ["GNB", "NYG"],
        ["CIN", "PIT"],
        ["LAC", "JAX"],
        ["SEA", "LAR"],
        ["SFO", "ARI"],
        ["BAL", "CLE"],
        ["KAN", "DEN"],
        ["DET", "PHI"],
        ["DAL", "LVR"],
    ],
    visualize=False,
)

# Week 12
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-11-19").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["BUF", "HOU"],
        ["SEA", "TEN"],
        ["IND", "KAN"],
        ["PIT", "CHI"],
        ["NWE", "CIN"],
        ["NYJ", "BAL"],
        ["MIN", "GNB"],
        ["NYG", "DET"],
        ["JAX", "ARI"],
        ["CLE", "LVR"],
        ["ATL", "NOR"],
        ["PHI", "DAL"],
        ["TAM", "LAR"],
        ["CAR", "SFO"],
    ],
    visualize=False,
)

# Week 13
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-11-26").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["GNB", "DET"],
        ["KAN", "DAL"],
        ["CIN", "BAL"],
        ["CHI", "PHI"],
        ["JAX", "TEN"],
        ["SFO", "CLE"],
        ["HOU", "IND"],
        ["NOR", "MIA"],
        ["ATL", "NYJ"],
        ["LAR", "CIN"],
        ["ARI", "TAM"],
        ["MIN", "SEA"],
        ["LVR", "LAC"],
        ["BUF", "PIT"],
        ["DEN", "WAS"],
        ["NYG", "NWE"],
    ],
    visualize=False,
)

# Week 14
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime("2025-12-04").strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[
        ["DAL", "DET"],
        ["TEN", "CLE"],
        ["MIA", "NYJ"],
        ["PIT", "BAL"],
        ["NOR", "TAM"],
        ["IND", "JAX"],
        ["WAS", "MIN"],
        ["SEA", "ATL"],
        ["CIN", "BUF"],
        ["DEN", "LVR"],
        ["CHI", "GNB"],
        ["LAR", "ARI"],
        ["HOU", "KAN"],
        ["PHI", "LAC"],
    ],
    visualize=False,
)
