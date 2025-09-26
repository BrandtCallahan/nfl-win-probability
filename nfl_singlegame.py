import pandas as pd
from nfl.single_game_setup.ml_singlegame_setup import *


# input season year
season = 2025

# # manual input for date
# today = pd.to_datetime("2025-09-20").strftime("%Y-%m-%d")
# automatic date (today)
today = datetime.now().strftime("%Y-%m-%d")

# matchup
num_seasons = 10
season_years = [season]
for i in range(num_seasons):
    season_years += [season - (i+1)]
away_tm = "TEN"
home_tm = "HOU"

# model type
model_type = "Model"

"""
POSSESSION SIMULATION DOESN'T EXIST FOR NFL YET

if model_type == "Simulation":
    # number of simulation iterations
    n = 301
    # neutral site game
    neutral_gm = False  # NCAA Tournament

    # single game results (win probability and average point spread)
    sg_win = game_sim(season, away_tm, home_tm, today, neutral_gm, n)

el
"""
if model_type == "Model":
    model = single_game_model(
        data_seasons=season_years,
        today=today,
        matchup=f"{away_tm} vs. {home_tm}",
    )
    sg_win = model[3]
    model_stats = model[1]

# donut chart for single game
sim_donut_graph(season, away_tm, home_tm, sg_win, hm_tm_prim=True, aw_tm_prim=True)

