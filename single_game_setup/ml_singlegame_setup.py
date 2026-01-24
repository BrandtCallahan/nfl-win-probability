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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, RidgeClassifier, Ridge
from sklearn.feature_selection import f_classif

# import from other folders
from utils.get_data import *
from utils.team_dict import *
from utils.beautiful_soup_helper import *

pd.set_option("future.no_silent_downcasting", True)


def single_game_model(data_seasons, today, week, matchups):

    # make sure data_seasons is sorted
    data_seasons.sort()

    # load up matchup results for training
    matchup_df = pd.DataFrame()
    for season in data_seasons:
        tmp_df = pd.read_csv(
            f"C:/Users/{os.getlogin()}/personal-github/nfl-win-probability/csv_files/{season}/season{season}_matchup_results.csv",
        )
        tmp_df = tmp_df.astype({"Game Date": "datetime64[ns]"})
        # concat all years of data into one df
        if matchup_df.empty:
            matchup_df = tmp_df.copy()
        else:
            matchup_df = pd.concat([matchup_df, tmp_df]).reset_index(drop=True)

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

    # dictionary available to map True/False values to 1/0 (respectively)
    tf_dict = {True: 1, False: 0}

    # data for the matchup
    matchup_data = pd.DataFrame()
    for matchup in matchups:
        hm_tm = matchup[1]
        aw_tm = matchup[0]
        tmp_data = rolling_gamedata(
            data_seasons[-1],
            hm_tm,
            aw_tm,
            pd.to_datetime(today),
        )
        tmp_data.loc[
            tmp_data["Hm_TmDiv"] == tmp_data["Aw_TmDiv"], "Divisional Game"
        ] = 1
        tmp_data["Divisional Game"] = tmp_data["Divisional Game"].fillna(0)

        if matchup_data.empty:
            matchup_data = tmp_data.copy()
        else:
            matchup_data = pd.concat([matchup_data, tmp_data]).reset_index(drop=True)

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

    season = matchup_data["Hm_Season"][0]

    for n, matchup in enumerate(matchups):
        matchup_teams = [
            matchup[0],
            matchup[1],
        ]

        elo_df = tm_elo_rating(season, pd.to_datetime(today) + timedelta(days=7))
        elo_df = elo_df[
            pd.to_datetime(elo_df["Date"]) < pd.to_datetime(today)
        ].reset_index(drop=True)

        tm_elo = (
            elo_df[
                (elo_df["Tm"].isin(matchup_teams)) | (elo_df["Opp"].isin(matchup_teams))
            ]
            .sort_values(by=["Date"])
            .reset_index(drop=True)
        )
        hm_results = tm_elo[
            (tm_elo["Tm"] == matchup[1]) | (tm_elo["Opp"] == matchup[1])
        ]
        hm_elo_df = hm_results.iloc[-1, :]
        if (matchup[1]) in hm_elo_df["Tm"]:
            hm_elo = hm_elo_df["Tm Elo"]
        else:
            hm_elo = hm_elo_df["Opp Elo"]

        aw_results = tm_elo[
            (tm_elo["Tm"] == matchup[0]) | (tm_elo["Opp"] == matchup[0])
        ]
        aw_elo_df = aw_results.iloc[-1, :]
        if (matchup[0]) in aw_elo_df["Tm"]:
            aw_elo = aw_elo_df["Tm Elo"]
        else:
            aw_elo = aw_elo_df["Opp Elo"]

        matchup_data.loc[n, "Home Elo"] = hm_elo
        matchup_data.loc[n, "Away Elo"] = aw_elo
        matchup_data.loc[n, "Elo_diff"] = (
            matchup_data["Home Elo"][n] - matchup_data["Away Elo"][n]
        )

        # gambling lines for current matchup
        odds_df = nfl_odds(max(data_seasons)).reset_index(drop=True)

        # quick week calc
        if week < 10:
            week = "0"+str(week)

        odds_df = odds_df[
            (odds_df["matchup"] == f"{matchup[0]} vs. {matchup[1]}")
            & (odds_df["game_id"].str[:7] == f"{max(data_seasons)}_{week}")
        ]

        matchup_data.loc[n, "Home Spread"] = odds_df["hm_spread"].values[0]
        matchup_data.loc[n, "Home Moneyline"] = odds_df["home_moneyline"].values[0]
        matchup_data.loc[n, "Away Moneyline"] = odds_df["away_moneyline"].values[0]

    # need to add all columns from matchup_df to matchup_data (the conference dummies)
    col_list = []
    for col in matchup_df.columns.tolist():
        if col in matchup_data.columns.tolist():
            pass
        else:
            # matchup_data[f"{col}"] = 0
            col_list += [col]

    # ensure FG % is not null
    for col in ["Aw_FGpct", "Aw_OppFGpct", "Hm_FGpct", "Hm_OppFGpct"]:
        matchup_df[f"{col}"] = matchup_df[f"{col}"].fillna(0)
        matchup_data[f"{col}"] = matchup_data[f"{col}"].fillna(0)

    # favorite/underdog boolean instead of moneyline odds
    ## will show Home Favorite based on the spread (negative designates favorite)
    matchup_df["Hm_Favorite"] = matchup_df["Home Spread"] <= 0
    matchup_data["Hm_Favorite"] = matchup_data["Home Spread"] <= 0

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
                "Home Pts",
                "Away Pts",
                "Home Pt Diff",
                "Home Spread",
                "Home Spread W",
                "Away Moneyline",
                "Home Moneyline",
                "Hm_Favorite",
                "Divisional Game",
                "Away Team",
                "Away Elo",
                # "Away Lg Rank",
                "Aw_TmOffEff",
                "Aw_TmDefEff",
                "Aw_TmEff",
                "Aw_OffEffAdj",
                "Aw_DefEffAdj",
                "Aw_W",
                "Aw_G",
                "Aw_Wpct",
                "Aw_WStreak",
                "Aw_OppSOS",
                "Aw_TotPtDiff",
                "Aw_AvgMoV",
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
                # "Home Lg Rank",
                "Hm_TmOffEff",
                "Hm_TmDefEff",
                "Hm_TmEff",
                "Hm_OffEffAdj",
                "Hm_DefEffAdj",
                "Hm_W",
                "Hm_G",
                "Hm_Wpct",
                "Hm_WStreak",
                "Hm_OppSOS",
                "Hm_TotPtDiff",
                "Hm_AvgMoV",
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
        .fillna({"Home Moneyline": -110, "Away Moneyline": -110})
    )

    # map Hm_Favorite to numeric value
    model_df["Hm_Favorite"] = model_df["Hm_Favorite"].map(tf_dict)

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

    # Total Points
    model_df['Total Pts'] = model_df['Home Pts'] + model_df['Away Pts']

    # feature selection
    model_df = model_df[
        [
            "Home W",
            "Home Team",
            "Home Pts",
            "Away Team",
            "Away Pts",
            "Matchup",
            "Game Date",
            'Total Pts',
            "Home Pt Diff",
            "Home Spread",
            "Home Spread W",
            "Away Moneyline",
            "Home Moneyline",
            "Hm_Favorite",
            "Aw_TmOffEff",
            "Aw_TmEff",
            "Aw_OffEffAdj",
            "Aw_DefEffAdj",
            "Aw_W",
            "Aw_G",
            "Aw_WStreak",
            "Aw_OppSOS",
            "Aw_TotPtDiff",
            "Aw_AvgMoV",
            "Aw_PassOffEff",
            "Aw_Pts",
            "Aw_PassCmppct",
            "Aw_PassAdjYdsAtt",
            "Aw_PassTDG",
            "Aw_Dwn3Conv",
            "Aw_OppTOVG",
            "Aw_OppDwn3Conv",
            "Aw_TmLuck",
            "Hm_TmOffEff",
            "Hm_TmEff",
            "Hm_OffEffAdj",
            "Hm_DefEffAdj",
            "Hm_W",
            "Hm_G",
            "Hm_WStreak",
            "Hm_OppSOS",
            "Hm_TotPtDiff",
            "Hm_AvgMoV",
            "Hm_Poss",
            "Hm_RushOffEff",
            "Hm_TmLuckW",
            "Hm_Pts",
            "Hm_OppPts",
            "Hm_PassCmppct",
            "Hm_PassAdjYdsAtt",
            "Hm_RushYdsAtt",
            "Hm_RushTDG",
            "Hm_Dwn3Conv",
            "Hm_OppPassAdjYdsAtt",
            "Hm_OppTOVG",
            "Hm_OppDwn3Conv",
            "Hm_TmLuck",
            "Elo_diff",
            "Home Elo",
            "Away Elo",
            # "Home Lg Rank",
            # "Away Lg Rank",
        ]
    ].astype(
        {
            "Aw_WStreak": "float64",
            "Hm_WStreak": "float64",
            "Home Pt Diff": "float64",
            "Home W": "float64",
        }
    )

    # adding record columns (format: W-L)
    ## loss calculated as G minus W
    model_df["Hm_L"] = model_df["Hm_G"] - model_df["Hm_W"]
    model_df["Hm_Record"] = (
        (model_df["Hm_W"].astype(int).astype(str))
        + "-"
        + (model_df["Hm_L"].astype(int).astype(str))
    )
    model_df["Aw_L"] = model_df["Aw_G"] - model_df["Aw_W"]
    model_df["Aw_Record"] = (
        (model_df["Aw_W"].astype(int).astype(str))
        + "-"
        + (model_df["Aw_L"].astype(int).astype(str))
    )

    # adding to matchup_data
    ## loss calculated as G minus W
    matchup_data["Hm_L"] = matchup_data["Hm_G"] - matchup_data["Hm_W"]
    matchup_data["Hm_Record"] = (
        (matchup_data["Hm_W"].astype(int).astype(str))
        + "-"
        + (matchup_data["Hm_L"].astype(int).astype(str))
    )
    matchup_data["Aw_L"] = matchup_data["Aw_G"] - matchup_data["Aw_W"]
    matchup_data["Aw_Record"] = (
        (matchup_data["Aw_W"].astype(int).astype(str))
        + "-"
        + (matchup_data["Aw_L"].astype(int).astype(str))
    )

    """
        Set Target Variables/DFs
    """
    final_pred_df = pd.DataFrame()
    final_model_stats = pd.DataFrame()
    for target_variable in [
        "Home W",
        "Home Spread W",
        "Home Pt Diff",
        "Total Pts",
    ]:
        # logger.info(f"data transformed: setting target variable - {target_variable}")
        # target variable
        y = model_df[
            [
                f"Matchup",
                f"{target_variable}",
            ]
        ]
        model_y = y.drop(columns=["Matchup"])

        if target_variable in ["Home W", "Home Pt Diff", "Home Spread W"]:
            X = model_df.drop(
                columns=[
                    "Home W",
                    "Home Team",
                    "Home Pts",
                    "Away Team",
                    "Away Pts",
                    "Total Pts",
                    "Game Date",
                    "Home Pt Diff",
                    "Home Spread W",
                    "Home Spread",
                    "Away Moneyline",
                    "Home Moneyline",
                    "Hm_Favorite",
                    "Aw_Record",
                    "Hm_Record",
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
                    "Home Spread W",
                    "Home Pts",
                    "Away Pts",
                    "Total Pts",
                    "Away Moneyline",
                    "Home Moneyline",
                    "Hm_Favorite",
                    "Home Spread",
                    "Aw_Record",
                    "Hm_Record",
                    f"{target_variable}",
                ]
            )
            model_X = X.drop(columns=["Matchup"])

        """
            Run Model
        """
        # logger.info(f"commence model run... NOW")
        # in order to predict probability of attendance use "model.predict_proba()"

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            model_X.iloc[1:],
            model_y.iloc[1:],
            test_size=0.25,
            random_state=14,
        )

        # call model with parameters
        if target_variable in ["Home W", "Home Spread W"]:

            model = RandomForestClassifier(
                class_weight="balanced",
                n_estimators=500,
                criterion='gini',
                random_state=14,
            )
            model.fit(X_train, np.ravel(y_train))

        else:
            model = LinearRegression()

            model.fit(X_train, np.ravel(y_train))

        """
                FINDING best features
        """
        features = model_X.columns.tolist()
        f_statistic, p_values = f_classif(model_X, np.ravel(model_y))

        feat_df = pd.DataFrame(
            data={
                "features": features,
                "f_stat": f_statistic,
                "p_values": p_values,
            }
        ).reset_index(drop=True)

        # limit to feature significance of <= 0.05
        feature_list = feat_df[feat_df["p_values"] <= 0.05].features.tolist()

        # limit to Top n features
        # feature_list = feat_df.iloc[:5].features.tolist()

        # USE IF ALL FEATURES WANTED
        # feature_list = feat_df.features.tolist()

        # re-run model with "important" features
        model.fit(X_train[feature_list], np.ravel(y_train))

        if target_variable in ("Home Pt Diff", "Total Pts"):
            # adding coefficients values for Linear Regression models
            feat_df = (
                pd.concat(
                    [
                        feat_df[feat_df["p_values"] <= 0.05].reset_index(drop=True),
                        pd.DataFrame(data={"linear_coef": model.coef_}),
                    ],
                    axis=1,
                )
                .sort_values(by=["p_values"], ascending=True)
                .reset_index(drop=True)
            )

        # suppress scientific notation
        np.set_printoptions(suppress=True)
        predictions = model.predict(X_test[feature_list])
        if target_variable in ["Home W", "Home Spread W"]:
            predict_prob = model.predict_proba(X_test[feature_list])

        # roll through prediction probabilities and give me the probability of attending the game
        #   i.e. the second number of the array for each index
        w_prob = []
        w_pred = []
        for i in range(len(predictions)):
            tmp_pred = predictions[i]
            w_pred += [tmp_pred]

            if target_variable in ["Home W", "Home Spread W"]:
                tmp_prob = predict_prob[i]
                w_prob += [tmp_prob[1]]

        # view results in df (with error)
        tmp_df = pd.DataFrame(
            columns=[
                "Matchup",
                "Game Date",
                "Home Team",
                "Home Record",
                "Away Team",
                "Away Record",
                "Home W",
                "Home Spread W",
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
        tmp_df["Home Record"] = model_df.iloc[mylist]["Hm_Record"]
        tmp_df["Away Team"] = model_df.iloc[mylist]["Away Team"]
        tmp_df["Away Record"] = model_df.iloc[mylist]["Aw_Record"]
        tmp_df["Home W"] = model_df.iloc[mylist]["Home W"]
        tmp_df["Home Spread W"] = model_df.iloc[mylist]["Home Spread W"]

        tmp_df["Predict"] = w_pred
        if target_variable in ["Home W", "Home Spread W"]:
            tmp_df["Predict Probability"] = w_prob

        # model statistics
        if target_variable in ["Home W", "Home Spread W"]:
            model_stats_df = pd.DataFrame(
                columns=[
                    "Target Variable",
                    "R^2",
                    "RMSE",
                    "MAE",
                    "Recall Score",
                    "Precision Score",
                    "ROC AUC",
                    "F1 Score",
                ]
            )
            model_stats_df["Target Variable"] = [target_variable]
            model_stats_df["R^2"] = [model.score(X_test[feature_list], y_test)]
            model_stats_df["RMSE"] = [np.sqrt(mean_squared_error(y_test, predictions))]
            model_stats_df["MAE"] = [mean_absolute_error(y_test, predictions)]
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
                    "RMSE",
                    "MAE",
                ]
            )
            model_stats_df["Target Variable"] = [target_variable]
            model_stats_df["R^2"] = [model.score(X_test[feature_list], y_test)]
            model_stats_df["RMSE"] = [np.sqrt(mean_squared_error(y_test, predictions))]
            model_stats_df["MAE"] = [mean_absolute_error(y_test, predictions)]

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
        # logger.info(f"predicting {target_variable}: {matchup_data["Matchup"][0]}")

        prediction_df = matchup_data[feature_list]

        if target_variable in ["Home W", "Home Spread W"]:

            prediction_prob = model.predict_proba(prediction_df)
        prediction = model.predict(prediction_df)

        # roll through prediction probabilities and give me the probability of attending the game
        #   i.e. the second number of the array for each index
        w_prob = []
        w_pred = []
        for i in range(len(prediction)):
            tmp_pred = prediction[i]
            w_pred += [tmp_pred]

            if target_variable in ["Home W", "Home Spread W"]:
                tmp_prob = prediction_prob[i]
                w_prob += [tmp_prob[1]]

        # view results in df
        pred_df = pd.DataFrame(
            columns=[
                "Matchup",
                "Game Date",
                "Home Team",
                "Home Record",
                "Away Team",
                "Away Record",
                f"Predict",
                f"Predict Probability",
                "Away Moneyline",
                "Home Moneyline",
                "Home Spread",
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
        pred_df["Home Record"] = matchup_data.iloc[mylist]["Hm_Record"]
        pred_df["Away Team"] = matchup_data.iloc[mylist]["Aw_Tm"]
        pred_df["Away Record"] = matchup_data.iloc[mylist]["Aw_Record"]
        pred_df["Away Moneyline"] = matchup_data.iloc[mylist]["Away Moneyline"]
        pred_df["Home Moneyline"] = matchup_data.iloc[mylist]["Home Moneyline"]
        pred_df["Home Spread"] = matchup_data.iloc[mylist]["Home Spread"]

        pred_df[f"Predict"] = w_pred

        if target_variable in ["Home W", "Home Spread W"]:
            pred_df[f"Predict Probability"] = w_prob

            # round the probability variable
            pred_df["Predict Probability"] = np.round(pred_df["Predict Probability"], 4)

        if (final_pred_df.empty) & (target_variable in ["Home W", "Home Spread W"]):
            final_pred_df = pred_df.rename(
                columns={
                    "Predict": f"{target_variable}",
                    "Predict Probability": f"{target_variable} Probability",
                }
            )
        elif (final_pred_df.empty) & (
            target_variable not in ["Home W", "Home Spred W"]
        ):
            final_pred_df = pred_df.rename(
                columns={
                    "Predict": f"{target_variable}",
                }
            )
        elif target_variable in ["Home W", "Home Spread W"]:
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

    return [final_pred_df, final_model_stats]


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

    away_score = sim_results_df["Pred. Pts"][0]
    home_score = sim_results_df["Pred. Pts"][1]

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
        f"@ {home_tm} ({spread})\n\n Total Pts: {int(round(away_score, 0)) + int(round(home_score, 0))}\n Margin of Victory: {abs(int(round(away_score, 0)) - int(round(home_score, 0)))}\n\n {away_tm}: {int(round(away_score, 0))}\n {home_tm}: {int(round(home_score, 0))}",
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
    # TODO: records are looking funky with a possible negative number
    # plt.legend(
    #     [f"{away_abbr} ({aw_record})", f"{home_abbr} ({hm_record})"], loc="upper right"
    # )
    plt.legend(
        [f"{away_abbr}", f"{home_abbr}"], loc="upper right"
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
