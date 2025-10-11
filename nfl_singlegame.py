import pandas as pd
from single_game_setup.ml_singlegame_setup import *

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
        if sg_win['Win Prob.'][0] > sg_win['Win Prob.'][1]:
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

        model_stats = model[1]

        # reformat model stats
        tmp_stats = pd.DataFrame(
            data={
                "Matchup": [f"{away_tm} vs. {home_tm}"],
                "W Exp. Accuracy": [round(model_stats["F1 Score"][0], 3)],
                "Pt Diff Accuracy": [round(model_stats["R^2"][1], 3)],
                "Away Pts Accuracy": [round(model_stats["R^2"][3], 3)],
                "Away Pts +/-": [round(model_stats["RMSE"][3], 2)],
                "Home Pts Accuracy": [round(model_stats["R^2"][2], 3)],
                "Home Pts +/-": [round(model_stats["RMSE"][2], 2)],
            }
        )

        if visualize:
            # donut chart for single game
            sim_donut_graph(
                season, away_tm, home_tm, sg_win, hm_tm_prim=True, aw_tm_prim=True
            )

        if nfl_df.empty:
            nfl_df = tmp_df.copy()
        else:
            nfl_df = pd.concat([nfl_df, tmp_df]).reset_index(drop=True)

        if nfl_model_stats.empty:
            nfl_model_stats = tmp_stats.copy()
        else:
            nfl_model_stats = pd.concat([nfl_model_stats, tmp_stats]).reset_index(drop=True)

    return [nfl_df, nfl_model_stats]


# Week 5
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime('2025-10-02').strftime("%Y-%m-%d"),
    num_seasons=10,
    matchup=[["SFO", "LAR"],
             ['MIN', 'CLE'],
             ['LVR', 'IND'],
             ['NYG', 'NOR'],
             ['DAL', 'NYJ'],
             ['DEN', 'PHI'],
             ['MIA', 'CAR'],
             ['HOU', 'BAL'],
             ['TEN', 'ARI'],
             ['TAM', 'SEA'],
             ['DET', 'CIN'],
             ['WAS', 'LAC'],
             ['NWE', 'BUF'],
             ['KAN', 'JAX']],
    visualize=False,
)

# Week 6
## added Tm Off Eff and Tm Eff
nfl_df = run_model(
    season=2025,
    today=pd.to_datetime('2025-10-09').strftime("%Y-%m-%d"),
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
        ["SFO", "TAM"],
        ["DET", "KAN"],
        ["BUF", "ATL"],
        ["CHI", "WAS"],
    ],
    visualize=False,
)
