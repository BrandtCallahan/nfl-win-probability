import bidict
import pandas as pd
from utils.beautiful_soup_helper import *


def get_teamnm():
    team_df = pd.read_csv(
        f"~/nfl-win-probability/csv_files/team_df.csv"
    )

    return team_df


def get_teamcolor_prim(tm):
    team_df = get_teamnm()

    team_color = (
        team_df[(team_df["Tm Abbrv"] == tm)]
        .reset_index(drop=True)["Tm Primary Color"]
        .to_list()
        .pop()
    )

    return team_color


def get_teamcolor_sec(tm):
    team_df = get_teamnm()

    team_color = (
        team_df[(team_df["Tm Abbrv"] == tm)]
        .reset_index(drop=True)["Tm Secondary Color"]
        .to_list()
        .pop()
    )

    return team_color

