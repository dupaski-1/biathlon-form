# pipeline.py
# The data pipeline of the project. The steps so far:
#
#   python pipeline.py features    build data/features.csv
#   python pipeline.py split       print the season split and the validation folds
#   python pipeline.py baselines   score the 3 baseline models, save the tables in outputs/
#   python pipeline.py models      ridge and random forest, choose the final model
#   python pipeline.py robustness  step 6, part A: the step 5 candidates on a sliding window
#   python pipeline.py test        step 6: THE FINAL TEST. Opens the test seasons. Walk
#                                  forward: each test season is predicted by the final
#                                  model refitted on every season before it.
#
# About "features":
# It reads data/sprints_raw.csv and data/races_all.csv, cleans them, builds
# the target and the 13 features, and writes data/features.csv with
# 1 row per finisher per sprint.
#
# Golden rule: a feature of a race only uses races that started BEFORE it.

import os
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SPRINT_FILE = "data/sprints_raw.csv"    # sprints with split times (from download_sprints.py)
RACES_FILE = "data/races_all.csv"       # all formats, results only (from download_races.py)
FEATURES_FILE = "data/features.csv"     # what this step creates
OUTPUT_DIR = "outputs"                  # the score tables of step 4 go here

RELAY_CODES = ["RL", "SR"]              # relay and single mixed relay

TARGET = "finish_pct"

# The window of features 6, 8 and 9: the sprints of the current season
# before the race, plus the whole previous season.
# Dispersion index (feature 9), shrunk toward pure luck (index 1), the same
# idea as the adjusted beta of Vasicek in finance: the raw value of an athlete
# is mixed with the value 1, and the fewer sprints, the more weight goes to 1.
# shrunk = (n x D + DISPERSION_PRIOR_WEIGHT x 1) / (n + DISPERSION_PRIOR_WEIGHT).
# 5 means: the value 1 counts as much as 5 sprints of the athlete.
DISPERSION_PRIOR_WEIGHT = 5
# The sample variance (ddof=1) needs at least 2 sprints. With fewer: the index is 1.
MIN_SPRINTS_DISPERSION = 2
# A sprint has 2 shootings of 5 targets: 10 shots.
SHOTS_PER_SPRINT = 10

# The season split. A season is written like "1314" for the winter 2013/14.
# History only: the first season of the data. Its rows have almost no past
# (nothing exists before 2012), so we never train on them and never score them.
# They were only useful to build the features of the later seasons.
HISTORY_SEASONS = ["1213"]
# Training: the seasons the models may learn from.
TRAIN_SEASONS = ["1314", "1415", "1516", "1617", "1718",
                 "1819", "1920", "2021", "2122", "2223"]
# Test: the 3 last seasons. They stay closed until step 6 (the final score),
# so that no choice we make can be influenced by them.
TEST_SEASONS = ["2324", "2425", "2526"]
# Validation: each of these seasons is predicted once by a model trained on
# all training seasons before it. Every model uses these same 5 folds.
VALIDATION_SEASONS = ["1819", "1920", "2021", "2122", "2223"]

TOP_N = 10      # the hit rate metric looks at the top 10 of each race

# Range time rule: in a sprint, a range time above 10 minutes cannot be true
# (2 shootings take about 1 to 2 minutes together). Such a value is a mistake
# in the raw data, so we treat it as missing. 10 minutes = 600 seconds.
MAX_RANGE_SECONDS = 600

# The 16 races that were cancelled (seasons 2012/13 to 2022/23). They have no
# rows in the data files. The list was printed by download_races.py at the
# end of the download and is copied here once.
CANCELLED_RACES = [
    "BT1314SWRLCP01SWPU", "BT1314SWRLCP01SMPU", "BT1314SWRLCP06SWRL",
    "BT1516SWRLCP09SMMS", "BT1516SWRLCP09SWMS", "BT1819SWRLCP07SMSP",
    "BT1819SWRLCP07SWSP", "BT1920SWRLCP08MXSR", "BT1920SWRLCP08MXRL",
    "BT1920SWRLCP09SWSP", "BT1920SWRLCP09SMSP", "BT1920SWRLCP09SWPU",
    "BT1920SWRLCP09SMPU", "BT1920SWRLCP09SWMS", "BT1920SWRLCP09SMMS",
    "BT2223SWRLCP09SWPU",
]

# Step 4, the linear models. Alpha says how strongly the weights are pulled
# towards 0. Step 4 scores every alpha of the grids and chooses nothing.
ALPHA_GRID = [0.01, 0.1, 1, 10, 100, 1000]                # ridge
LASSO_ALPHAS = [0.00001, 0.0001, 0.001, 0.01, 0.1]       # lasso
LASSO_MAX_ITER = 10000      # how many rounds the lasso may use to find its weights

# The value the simple models predict when they know nothing: the middle.
MIDDLE = 0.5

# The 5 scores, in the order of the tables. list_mae is the primary metric.
SCORES = ["list_mae", "list_rmse", "list_r2", "spearman", "top10"]

# The 13 features. Changed on 17 Nov: feature 4 and 6 have a new rule,
# 8 and 9 are new (they replace "days since the last sprint" and "days since
# the last race day", which were dropped with their 365 day caps).
# "Window" = the sprints of the current season before the race + the whole previous season.
FEATURES = [
    "last3_finish_pct",    # 1  mean finish percentile, last 3 sprints
    "last3_ski_pct",       # 2  mean ski percentile, last 3 sprints
    "last3_range_pct",     # 3  mean range percentile, last 3 sprints
    "last3_miss_pct",      # 4  mean miss percentile, last 3 sprints
    "last_finish_pct",     # 5  finish percentile of the last sprint
    "n_sprints_window",    # 6  number of sprints in the window
    "last_place_any",      # 7  finish place in the last race of any format, no relays
    "window_mean_misses",  # 8  mean misses per sprint in the window
    "window_miss_dispersion",  # 9  dispersion index of the misses in the window
    "race_days_21d",       # 10 number of race days in the last 21 days, relays included
    "missed_prev_event",   # 11 flag: did not start at the previous event of the season
    "first_start_major",   # 12 flag: first start at this World Championships / Olympics
    "no_history",          # 13 flag: no previous sprint
]


# Step 5: 4 model families on 2 input sets. All rules were written down
# BEFORE looking at any step 5 result.
# The 3 load features: how much and how recently the athlete raced.
LOAD_FEATURES = ["race_days_21d", "missed_prev_event", "first_start_major"]
# FULL = all 13 inputs ("with load"). NO_LOAD = without the 3 load features (10 inputs).
INPUT_SETS = {
    "FULL": FEATURES,
    "NO_LOAD": [name for name in FEATURES if name not in LOAD_FEATURES],
}

# The random forest: 300 trees, each split looks at a third of the inputs,
# a leaf may hold 1 single row (the default), and the seed makes every run the same.
FOREST_TREES = 300
FOREST_MAX_FEATURES = 1 / 3
FOREST_MIN_LEAF = 1
FOREST_SEED = 42
# The depths we try. NO_LIMIT stands for "max_depth=None": the trees grow
# until the end. It is written as infinity, so that it sorts as the deepest.
NO_LIMIT = np.inf
DEPTH_GRID = [2, 4, 6, 8, 12, 16, NO_LIMIT]
# Refinement round, once: around the best depth, the 2 depths halfway to its
# neighbours in the grid are added.
EXTRA_DEPTHS = {2: [1, 3], 4: [3, 5], 6: [5, 7], 8: [7, 10],
                12: [10, 14], 16: [14, 20], NO_LIMIT: [20, 24]}

# Gradient boosting (HistGradientBoostingRegressor): at most 2000 trees with
# early stopping, the seed 42, and the grid of the course (L12).
BOOST_MAX_ITER = 2000
BOOST_SEED = 42
BOOST_RATES = [0.02, 0.05, 0.1, 0.2]
BOOST_DEPTHS = [3, 4, 6]

# The selection rule (one standard error rule), step 4 of the rule: among the
# eligible candidates the simplest wins. Family order: linear before trees,
# ridge before lasso, forest before boosting.
FAMILY_ORDER = {"ridge": 1, "lasso": 2, "forest": 3, "boosting": 4}
# Grid edge: if the choice is at the edge of the ridge or lasso grid, that
# grid is extended by 2 steps of 10 beyond the edge, once.
EDGE_STEPS = 2

# Step 6, part A: the robustness run. Each validation season is predicted
# from the WINDOW_SEASONS seasons just before it only (a sliding window),
# and not from all earlier seasons as in step 5.
WINDOW_SEASONS = 5

# Step 6: the final test. The final model chosen in step 5 (18 Nov),
# written here by hand after "python pipeline.py models". Frozen: nothing
# here may change after the test numbers have been seen. In the walk forward
# test the same frozen model is fitted again for each test season, on every
# season before it: the rows change, the model and its settings do not.
FINAL_FAMILY = "lasso"           # chosen by the one standard error rule
FINAL_INPUTS = "NO_LOAD"         # 10 inputs, without the 3 load features
FINAL_VALUE = 0.01               # alpha of the lasso
# load_test_rows refuses to read the test seasons while this is False.
# Set to True by hand on 18 Nov, after the go of the project owner.
TEST_UNLOCKED = True
# What we expect to load. If a count is different, the command stops
# before any test score is computed.
EXPECTED_TRAIN_ROWS = 18183
EXPECTED_TEST_ROWS = 4778
EXPECTED_TEST_RACES = 48
# Per athlete predictions go to data/ (not in git). outputs/ holds no athlete rows.
TEST_PREDICTIONS_FILE = "data/test_predictions.csv"
STORY_FIGURE = "test_story.png"
# Permutation importance: how often each column is shuffled, and the seed.
PERM_REPEATS = 20
PERM_SEED = 42
# The groups of inputs that are shuffled together. Together they hold each
# of the 10 NO_LOAD inputs exactly once.
INPUT_GROUPS = {
    "results": ["last3_finish_pct", "last_finish_pct", "last_place_any"],
    "skiing": ["last3_ski_pct"],
    "shooting": ["last3_range_pct", "last3_miss_pct", "window_mean_misses", "window_miss_dispersion"],
    "experience": ["n_sprints_window", "no_history"],
}
# The limits of the slices.
PCT_FRONT = 0.10        # real finish percentile up to here = the front of the race
PCT_BACK = 0.90         # from here on = the back of the race
SPRINTS_FEW = 3         # n_sprints_window: 0 to 3, 4 to 9, 10 or more
SPRINTS_MANY = 10
# The story race: number 24 of the 48 races sorted by list MAE. 48 is even,
# so there are 2 middle races (24 and 25); we take the lower one.
STORY_NUMBER = 24
# The 3 models that are scored on the test seasons, and their column of predictions.
TEST_MODELS = {"final": "pred_final", "constant": "pred_constant", "naive": "pred_naive"}
# The slices look at these 2 models only.
SLICE_MODELS = ["final", "naive"]

# The learning curve: the shares of the training RACES, and the seed that
# picks the random subset of whole races.
CURVE_SHARES = [0.2, 0.4, 0.6, 0.8, 1.0]
CURVE_SEED = 42


# ------------------------------------------------------------------ cleaning

def to_seconds(t):
    """Turn a time written as text into seconds. '25:10.4' -> 1510.4"""
    if pd.isna(t):
        return np.nan                    # empty stays empty
    t = str(t).replace("+", "")          # 'Behind' is written like '+18.1'
    try:
        secs = 0.0
        for part in t.split(":"):        # hours : minutes : seconds
            secs = secs * 60 + float(part)
        return secs
    except ValueError:
        return np.nan                    # a word like 'Lapped' is not a time


def clean_irm(col):
    """In IRM, an empty cell and the word 'None' both mean 'normal finisher'."""
    return col.replace("None", np.nan)


def load_sprints():
    """Load the sprint file and clean it."""
    df = pd.read_csv(SPRINT_FILE, dtype={"season": str, "Bib": str})
    df["IRM"] = clean_irm(df["IRM"])
    # Every time column gets a twin in seconds, with "_s" at the end of the name.
    for col in ["TotalTime", "Behind", "CRST", "RNGT", "STTM",
                "CRS1", "CRS2", "CRS3", "RNG1", "RNG2"]:
        df[col + "_s"] = df[col].apply(to_seconds)
    df["start"] = pd.to_datetime(df["start"])      # text -> real date and time
    df["date"] = df["start"].dt.normalize()        # the day only, without the hour
    return df


def load_races():
    """Load the file with all formats and clean it. Everything in it is text."""
    df = pd.read_csv(RACES_FILE, dtype=str)
    df["IRM"] = clean_irm(df["IRM"])
    df["Rank"] = pd.to_numeric(df["Rank"])                    # text -> number
    df["is_major_event"] = df["is_major_event"] == "True"     # text -> True / False
    for col in ["TotalTime", "Behind"]:
        df[col + "_s"] = df[col].apply(to_seconds)
    df["start"] = pd.to_datetime(df["start"])
    df["date"] = df["start"].dt.normalize()
    return df


# ------------------------------------------------------------------ target

def percentile(values, race_id):
    """Rank the values inside each race and put the rank on a 0 to 1 scale.
    0 = best of the race, 1 = last of the race."""
    rank = values.groupby(race_id).rank(method="min")     # ties share the better rank
    n = values.groupby(race_id).transform("count")        # how many have a value
    return (rank - 1) / (n - 1)


def build_target(sprints):
    """Keep the finishers and compute what they did in each sprint."""
    fin = sprints[sprints["IRM"].isna()].copy()            # finishers only

    # Finish percentile = (rank - 1) / (number of finishers - 1).
    n_finishers = fin.groupby("race_id")["Rank"].transform("count")
    fin[TARGET] = (fin["Rank"] - 1) / (n_finishers - 1)

    # Range time rule: a range time above the limit counts as missing.
    # This is done here, BEFORE the range percentiles are computed, so the
    # wrong value gets no percentile and does not push the others.
    fin.loc[fin["RNGT_s"] > MAX_RANGE_SECONDS, "RNGT_s"] = np.nan

    # Same idea with the skiing time (CRST) and the time on the range (RNGT).
    fin["ski_pct"] = percentile(fin["CRST_s"], fin["race_id"])
    fin["range_pct"] = percentile(fin["RNGT_s"], fin["race_id"])

    fin["misses"] = fin["ShootingTotal"]                   # number of missed targets

    # Miss percentile: rank the finishers of each sprint by misses, fewest
    # first. Athletes with the same number of misses share their ranks: they
    # all get the average of these ranks (method="average"). Then the 0 to 1 scale.
    miss_rank = fin.groupby("race_id")["misses"].rank(method="average")
    n_finishers = fin.groupby("race_id")["misses"].transform("count")
    fin["miss_pct"] = (miss_rank - 1) / (n_finishers - 1)
    return fin


# ------------------------------------------------------------------ features

def add_sprint_history(fin):
    """Features 1 to 5 and 13: they only need the earlier sprints where the
    athlete had a valid finish. 'fin' holds exactly those sprints."""
    # Put the sprints of each athlete in time order.
    fin = fin.sort_values(["IBUId", "start"]).reset_index(drop=True)
    by_athlete = fin.groupby("IBUId")

    # shift(1) moves every value 1 line down inside the athlete, so that a
    # line only sees the sprints before it and never its own result.
    # rolling(3, min_periods=1) takes the mean of up to 3 of those values;
    # with 1 or 2 it uses what is there, with none the result stays empty.
    def mean_last3(col):
        return col.shift(1).rolling(3, min_periods=1).mean()

    fin["last3_finish_pct"] = by_athlete[TARGET].transform(mean_last3)
    fin["last3_ski_pct"] = by_athlete["ski_pct"].transform(mean_last3)
    fin["last3_range_pct"] = by_athlete["range_pct"].transform(mean_last3)
    fin["last3_miss_pct"] = by_athlete["miss_pct"].transform(mean_last3)

    # Feature 5: the result of the sprint just before.
    fin["last_finish_pct"] = by_athlete[TARGET].shift(1)

    # cumcount() numbers the lines of an athlete 0, 1, 2, ... which is the
    # number of valid sprints before this one, over the whole history.
    fin["n_before_all"] = by_athlete.cumcount()

    # Feature 13: 1 when there is no earlier sprint.
    fin["no_history"] = (fin["n_before_all"] == 0).astype(int)
    return fin


def previous_season(season):
    """The season before, like "1314" -> "1213". Empty for the first season."""
    seasons = HISTORY_SEASONS + TRAIN_SEASONS + TEST_SEASONS      # in time order
    position = seasons.index(season)
    if position == 0:
        return None
    return seasons[position - 1]


def add_window_history(fin):
    """Features 6, 8 and 9: the sprints with a valid finish in the window =
    the current season before this race + the whole previous season."""
    key = ["race_id", "IBUId"]
    fin["prev_season"] = fin["season"].map(previous_season)

    # Put next to every sprint row ALL valid sprints of the same athlete.
    past = fin[["IBUId", "season", "start", "misses"]].rename(
        columns={"season": "h_season", "start": "h_start", "misses": "h_misses"})
    pairs = fin[key + ["season", "prev_season", "start"]].merge(past, on="IBUId")
    # Keep the window. The golden rule: only sprints that started BEFORE this one.
    earlier = pairs["h_start"] < pairs["start"]
    in_window = (pairs["h_season"] == pairs["season"]) | (pairs["h_season"] == pairs["prev_season"])
    window = pairs[earlier & in_window]

    # 1 line per sprint row: how many sprints, their mean misses and the
    # sample variance of their misses (ddof=1, divided by count - 1).
    stats = window.groupby(key)["h_misses"].agg(n_sprints_window="count",
                                                window_mean_misses="mean",
                                                window_miss_var=lambda m: m.var(ddof=1))
    fin = fin.merge(stats.reset_index(), on=key, how="left")

    # Feature 6: no sprint in the window means 0 (not "unknown"). No cap.
    fin["n_sprints_window"] = fin["n_sprints_window"].fillna(0).astype(int)
    # Feature 8 stays empty when the window has no sprint.

    # Feature 9, step 1: the raw dispersion index D = sample variance /
    # (10 x p x (1 - p)), with p = mean misses / 10, the share of missed shots.
    # 10 x p x (1 - p) is the variance we would see if every shot were a
    # coin toss with the same chance p ("pure luck"): D = 1 = as expected,
    # above 1 = the athlete has good and bad days, below 1 = very regular.
    n = fin["n_sprints_window"]
    p = fin["window_mean_misses"] / SHOTS_PER_SPRINT
    expected = SHOTS_PER_SPRINT * p * (1 - p)
    raw_index = fin["window_miss_var"] / expected
    # Step 2: shrink D toward 1. With n sprints, D gets the weight n and the
    # value 1 gets the weight DISPERSION_PRIOR_WEIGHT.
    shrunk = (n * raw_index + DISPERSION_PRIOR_WEIGHT * 1) / (n + DISPERSION_PRIOR_WEIGHT)
    # Step 3: where D cannot be computed (fewer than 2 sprints, or p is 0 or 1,
    # which would divide by 0), the feature is 1: we know nothing, so we
    # assume pure luck. The feature is never empty.
    can_compute = (n >= MIN_SPRINTS_DISPERSION) & (p > 0) & (p < 1)
    fin["window_miss_dispersion"] = np.where(can_compute, shrunk, 1.0)
    return fin


def add_race_history(fin, races):
    """Features 7, 10, 11 and 12: they need the races of all formats."""
    # A sprint row is identified by the race and the athlete.
    key = ["race_id", "IBUId"]

    # Take the event and the major event flag of each sprint from the big file.
    race_info = races.drop_duplicates("race_id")[["race_id", "event_id", "is_major_event"]]
    fin = fin.merge(race_info, on="race_id", how="left")

    # A race day = the athlete was in the race and it is not a DNS.
    # DNF, DSQ, LAP and relays (also when lapped before the athlete's leg) count.
    days = races[races["IRM"] != "DNS"]
    days = days[["IBUId", "start", "date", "event_id", "discipline", "IRM", "Rank"]]
    days = days.rename(columns={"start": "h_start", "date": "h_date",
                                "event_id": "h_event", "Rank": "h_rank"})

    # Put next to every sprint row ALL race days of the same athlete,
    # then keep only those that started before this sprint.
    # This line is the golden rule: nothing from the same race or later.
    pairs = fin[key + ["start", "date", "event_id"]].merge(days, on="IBUId")
    earlier = pairs[pairs["h_start"] < pairs["start"]]

    # Feature 7: last race with a valid finish place, relays excluded.
    # (DNF, DSQ and LAP have no rank, so they drop out here.)
    solo = earlier[~earlier["discipline"].isin(RELAY_CODES) & earlier["h_rank"].notna()]
    solo = solo.sort_values("h_start")
    last_place = solo.groupby(key)["h_rank"].last().rename("last_place_any")

    # A helper column (no feature): the date of the most recent race day.
    last_day = earlier.groupby(key)["h_date"].max().rename("last_race_date")

    # Feature 10: race days in the 21 days before. nunique() counts different
    # dates, so 2 rows on the same day (single mixed relay) count once.
    recent = earlier[earlier["h_date"] >= earlier["date"] - pd.Timedelta(days=21)]
    n_recent = recent.groupby(key)["h_date"].nunique().rename("race_days_21d")

    # Feature 12 needs: number of earlier race days at the same event.
    same_event = earlier[earlier["h_event"] == earlier["event_id"]]
    n_same = same_event.groupby(key).size().rename("n_before_in_event")

    # Bring the 4 results back to the sprint rows (1 value per race and athlete).
    for result in [last_place, last_day, n_recent, n_same]:
        fin = fin.merge(result.reset_index(), on=key, how="left")

    # No race day found in the window means 0 race days, not "unknown".
    fin["race_days_21d"] = fin["race_days_21d"].fillna(0).astype(int)
    # Flag 12: major event and no earlier race day at this event. 0 at World Cups.
    fin["first_start_major"] = (fin["is_major_event"]
                                & fin["n_before_in_event"].isna()).astype(int)

    # Feature 11: did the athlete miss the previous event of the same season?
    # Step a: list the events in time order and find, for each, the one before.
    events = races.groupby("event_id").agg(season=("season", "first"),
                                           first_start=("start", "min")).reset_index()
    events = events.sort_values("first_start")
    events["prev_event"] = events.groupby("season")["event_id"].shift(1)
    fin = fin.merge(events[["event_id", "prev_event"]], on="event_id", how="left")
    # Step b: list who had a race day at which event.
    was_there = days[["IBUId", "h_event"]].drop_duplicates()
    was_there = was_there.rename(columns={"h_event": "prev_event"})
    was_there["at_prev_event"] = True
    fin = fin.merge(was_there, on=["IBUId", "prev_event"], how="left")
    # Flag = 1 when there is a previous event and the athlete was not there.
    # At the first event of a season there is no previous event: flag = 0.
    fin["missed_prev_event"] = (fin["prev_event"].notna()
                                & fin["at_prev_event"].isna()).astype(int)
    return fin


# ------------------------------------------------------------------ checks

def print_checks(out, fin, races):
    """Print the checks of step 2. The file holds all seasons, but the checks
    only look at the open seasons: the test seasons stay closed until step 6."""
    # The only thing shown about all seasons together: the 2 counts.
    print("rows:", len(out), "(expected 24957)   races:", out["race_id"].nunique(), "(expected 252)")

    # From here on the 3 tables hold the open seasons only (2012/13 to 2022/23).
    # Everything printed below is computed on these smaller tables.
    out = open_seasons(out)
    fin = open_seasons(fin)
    races = open_seasons(races)
    print("open seasons only: rows:", len(out), "  races:", out["race_id"].nunique())

    print("\n--- share of missing values, min, mean, max (open seasons only) ---")
    table = pd.DataFrame({"missing": out[[TARGET] + FEATURES].isna().mean(),
                          "min": out[[TARGET] + FEATURES].min(),
                          "mean": out[[TARGET] + FEATURES].mean(),
                          "max": out[[TARGET] + FEATURES].max()})
    print(table.round(3).to_string())

    # Leak test on 1 athlete: raw results next to the features.
    # His last sprints in the open seasons are from 2022/23, a training season.
    # He also has 1 sprint with a DNS in that period, which must not be used.
    name = "LAEGREID Sturla Holm"
    one = fin[fin["Name"] == name].sort_values("start")
    one = one.assign(day=one["date"].dt.date)       # the date in a short form for printing
    print("\n--- leak test:", name, "---")
    print("\nHis last 9 valid sprints, raw results (the features of the last 6 are below):")
    raw_cols = ["day", "place", "Rank", TARGET, "ski_pct", "range_pct", "misses", "miss_pct"]
    print(one[raw_cols].tail(9).round(3).to_string(index=False))

    print("\nFeatures of his last 6 sprints:")
    last6 = one.tail(6)
    print(last6[["day", "place"] + FEATURES[:6]].round(3).to_string(index=False))
    print(last6[["day", "place"] + FEATURES[6:]].round(3).to_string(index=False))

    # All his races of any format around these sprints, to check 7, 9, 10, 11, 12.
    first_day = last6["date"].min() - pd.Timedelta(days=30)
    log = races[(races["IBUId"] == one["IBUId"].iloc[0]) & (races["date"] >= first_day)]
    log = log.sort_values("start").assign(day=log["date"].dt.date)
    print("\nAll his races from 30 days before the first of these 6 sprints:")
    print(log[["day", "event_id", "place", "discipline", "leg", "Rank", "IRM"]].to_string(index=False))


# ------------------------------------------------------------------ split

def open_seasons(df):
    """Remove the test seasons from a table. They stay closed until step 6."""
    return df[~df["season"].isin(TEST_SEASONS)].copy()


def load_fin():
    """Rebuild the working table of step 2 without the test seasons.
    1 row per finisher per sprint. It holds the raw result, the target, the
    13 features and the helper columns. Used by the lab book."""
    sprints = open_seasons(load_sprints())
    races = open_seasons(load_races())
    fin = build_target(sprints)
    fin = add_sprint_history(fin)
    fin = add_window_history(fin)
    fin = add_race_history(fin, races)
    fin["day"] = fin["date"].dt.date        # the date in a short form for printing
    return fin


def load_features():
    """Load data/features.csv WITHOUT the test seasons.
    Every later step gets its data from here, so no step can see the test rows."""
    df = pd.read_csv(FEATURES_FILE, dtype={"season": str})
    return df[~df["season"].isin(TEST_SEASONS)].reset_index(drop=True)


def count_test():
    """Count the rows and races of the test seasons, and nothing more.
    usecols reads only these 2 columns: no feature and no result is loaded."""
    df = pd.read_csv(FEATURES_FILE, dtype={"season": str}, usecols=["season", "race_id"])
    return df[df["season"].isin(TEST_SEASONS)]


def make_window_folds(df):
    """The folds of the robustness run (step 6, part A): the same 5 validation
    seasons, but each is predicted from the 5 seasons just before it only."""
    folds = []
    for val_season in VALIDATION_SEASONS:
        position = TRAIN_SEASONS.index(val_season)            # where the season is in the list
        seasons_before = TRAIN_SEASONS[position - WINDOW_SEASONS:position]    # the 5 before it
        train_part = df[df["season"].isin(seasons_before)]
        val_part = df[df["season"] == val_season]
        folds.append((val_season, train_part, val_part))
    return folds


def make_folds(df):
    """Build the walk forward folds. Returns a list with, for each fold:
    the validation season, the training part and the validation part."""
    folds = []
    for val_season in VALIDATION_SEASONS:
        # The training seasons that come before the validation season.
        # Seasons written as text sort in time order ("1314" < "1415" < ...).
        seasons_before = [s for s in TRAIN_SEASONS if s < val_season]
        train_part = df[df["season"].isin(seasons_before)]
        val_part = df[df["season"] == val_season]
        folds.append((val_season, train_part, val_part))
    return folds


# ------------------------------------------------------------------ metrics
# Each metric gets the rows of some WHOLE races (with race_id and the true
# finish percentile) and 1 prediction per row, in the same order.
#
# The primary metric is the LIST MAE. The model is used to make a predicted
# result list, so we score the list and not the predicted values:
# 1. in each race, sort the predictions from low to high and give each
#    athlete a predicted place 1, 2, ..., n (n = finishers in the race),
# 2. turn the place into a predicted percentile with the formula of the
#    target: (place - 1) / (n - 1),
# 3. compare this predicted percentile with the real finish percentile.
# RMSE and R2 are computed on the same predicted percentile.
# The raw MAE, RMSE and R2 on the predicted values were removed on 18 Nov.

def list_percentile(rows, pred):
    """The predicted result list, written as a percentile: 0 for the athlete
    predicted first in the race, 1 for the one predicted last.
    Equal predictions: the rows of the race are first mixed in a random order
    with the seed of the race (the same as in top_n_hit_rate), then sorted.
    The sort is "stable": equal predictions keep the mixed order."""
    table = pd.DataFrame({"race_id": rows["race_id"].to_numpy(),
                          "pred": np.asarray(pred, dtype=float)})
    pct = np.zeros(len(table))
    for race_id, race in table.groupby("race_id"):
        seed = zlib.crc32(race_id.encode())             # the seed of the race
        race = race.sample(frac=1, random_state=seed)   # mix the rows
        race = race.sort_values("pred", kind="stable")  # lowest prediction first
        place = np.arange(1, len(race) + 1)             # predicted place 1 to n
        pct[race.index] = (place - 1) / (len(race) - 1)
    return pct


def list_mae(rows, pred):
    """List MAE, the primary metric: the mean distance between the predicted
    percentile from the list and the real finish percentile.
    0 = the list is the real result. A random list gives about 1/3."""
    return np.mean(np.abs(list_percentile(rows, pred) - rows[TARGET].to_numpy()))


def list_rmse(rows, pred):
    """Like the list MAE, but big errors count more: every error is squared
    before the mean is taken, and the square root is taken at the end."""
    errors = list_percentile(rows, pred) - rows[TARGET].to_numpy()
    return np.sqrt(np.mean(errors ** 2))


def list_r2(rows, pred):
    """R2 of the predicted percentile from the list. 1 = perfect.
    0 = as good as predicting the mean of the truth for all. A random list
    gives about -1: its squared errors are about twice the spread of the truth."""
    true = rows[TARGET].to_numpy()
    left_over = np.sum((true - list_percentile(rows, pred)) ** 2)   # squared errors of the list
    spread = np.sum((true - true.mean()) ** 2)                      # squared errors of "always the mean"
    return 1 - left_over / spread


def mean_spearman(rows, pred):
    """Rank correlation between prediction and truth in each race, then the
    mean over the races. 1 = same order, 0 = no link."""
    table = pd.DataFrame({"race_id": rows["race_id"].to_numpy(),
                          "true": rows[TARGET].to_numpy(),
                          "pred": np.asarray(pred)})
    values = []
    for race_id, race in table.groupby("race_id"):
        # If a model predicts the same value for everybody in a race, there
        # is no order to compare: the correlation does not exist, skip the race.
        if race["pred"].nunique() < 2:
            continue
        # Spearman = the usual correlation, computed on the ranks.
        values.append(race["pred"].rank().corr(race["true"].rank()))
    if len(values) == 0:
        return np.nan                # no race could be scored
    return np.mean(values)


def top_n_hit_rate(rows, pred):
    """In each race: take the TOP_N athletes with the lowest predicted
    percentile and count the share that really finished in the top TOP_N.
    Then the mean over the races."""
    table = pd.DataFrame({"race_id": rows["race_id"].to_numpy(),
                          "true": rows[TARGET].to_numpy(),
                          "pred": np.asarray(pred)})
    values = []
    for race_id, race in table.groupby("race_id"):
        # Mix the rows of the race in a random order. The file is sorted by
        # real rank, so without this, equal predictions would be picked in
        # the real order and a model that predicts the same value for all
        # would look perfect.
        # Each race gets its own order. The seed is a number made from the
        # race ID: crc32 turns a text into a number, and the same text gives
        # the same number on every run and on every machine.
        seed = zlib.crc32(race_id.encode())
        race = race.sample(frac=1, random_state=seed)
        # The real place in the race, found again from the true percentile.
        place = race["true"].rank(method="min")
        picked = race["pred"].nsmallest(TOP_N).index      # our predicted top 10
        values.append((place[picked] <= TOP_N).mean())    # share of real top 10
    return np.mean(values)


def all_metrics(rows, pred):
    """The 5 metrics together, as a small dictionary. list_mae is the primary one."""
    return {"list_mae": list_mae(rows, pred),
            "list_rmse": list_rmse(rows, pred),
            "list_r2": list_r2(rows, pred),
            "spearman": mean_spearman(rows, pred),
            "top10": top_n_hit_rate(rows, pred)}


def show_size(label, df):
    """Print the number of rows and races of a table."""
    print(f"{label:<28} rows: {len(df):>6}   races: {df['race_id'].nunique():>4}")


def show_split():
    """Step 3: print the season split and the validation folds."""
    df = load_features()            # the test seasons are already removed here

    print("--- season split ---")
    show_size("history " + HISTORY_SEASONS[0], df[df["season"].isin(HISTORY_SEASONS)])
    show_size("training " + TRAIN_SEASONS[0] + "-" + TRAIN_SEASONS[-1],
              df[df["season"].isin(TRAIN_SEASONS)])
    show_size("test " + TEST_SEASONS[0] + "-" + TEST_SEASONS[-1] + " (counts only)",
              count_test())

    print("\n--- validation folds (walk forward by season) ---")
    for val_season, train_part, val_part in make_folds(df):
        seasons = train_part["season"].unique()
        show_size("train " + seasons.min() + "-" + seasons.max(), train_part)
        show_size("  validate " + val_season, val_part)


# ------------------------------------------------------------------ baselines
# Step 4: the ladder of simple models (constant, naive rule, ridge, lasso),
# scored with the list metrics on the 5 walk forward folds. Every number a
# model learns comes from the training part of the fold. No alpha is chosen.

def predict_constant(rows, value=MIDDLE):
    """Model 1: the same value for everybody. In step 4 the value is the mean
    of the target in the training part of the fold. All predictions tie, so
    the list comes from the seed of each race: a random order."""
    return np.full(len(rows), value)


def predict_naive(rows, fill=MIDDLE):
    """Model 2: sort by the mean finish percentile of the last 3 sprints.
    Where it is empty (no previous sprint), the value 'fill' is used. In
    step 4 'fill' is the median of this column in the training part of the
    fold. Nothing else is learned."""
    return rows["last3_finish_pct"].fillna(fill).to_numpy()


def make_ridge(alpha):
    """Model 3: 3 steps in a row, in 1 scikit-learn Pipeline.
    fill:  put the median of the column into the empty cells,
    scale: bring every column to mean 0 and spread 1,
    ridge: a linear model whose weights are pulled towards 0 by alpha.
    When we call fit, each step learns its numbers (medians, means and
    spreads, weights) from the rows given to fit, and from no other row."""
    return Pipeline([("fill", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("ridge", Ridge(alpha=alpha))])


def make_lasso(alpha):
    """Model 4: the same 3 steps, with a lasso at the end. A lasso also pulls
    the weights towards 0, but it can set a weight exactly to 0: the input
    is then dropped from the model."""
    return Pipeline([("fill", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("lasso", Lasso(alpha=alpha, max_iter=LASSO_MAX_ITER))])


def history_list_mae(rows, pred):
    """The list MAE of the rows without and with a previous sprint. The list
    is made on the WHOLE race first, then the rows are split."""
    errors = np.abs(list_percentile(rows, pred) - rows[TARGET].to_numpy())
    new = (rows["no_history"] == 1).to_numpy()
    return {"n_no_history": int(new.sum()),
            "list_mae_no_history": errors[new].mean() if new.any() else np.nan,
            "list_mae_with_history": errors[~new].mean()}


def score_line(model, alpha, fold, val_season, part, rows, pred):
    """1 line of the score table: who was scored, on which rows, and the scores."""
    line = {"model": model, "alpha": alpha, "fold": fold,
            "val_season": val_season, "part": part, "n_rows": len(rows)}
    line.update(all_metrics(rows, pred))
    line.update(history_list_mae(rows, pred))
    return line


def score_baselines():
    """Fit and score every candidate in every fold. Returns 3 tables:
    the scores, the weights of the linear models, and 1 line per linear
    model and fold with its number of nonzero weights and its warnings."""
    df = load_features()            # the test seasons are already removed here
    scores = []
    weights = []
    linear = []
    fold = 0
    for val_season, train_part, val_part in make_folds(df):
        fold = fold + 1
        parts = [("train", train_part), ("validation", val_part)]

        # 1. Constant: the mean of the target in the training part.
        mean = train_part[TARGET].mean()
        for part, rows in parts:
            scores.append(score_line("constant", np.nan, fold, val_season, part, rows,
                                     predict_constant(rows, mean)))

        # 2. Naive rule: empty values get the median of the training part.
        fill = train_part["last3_finish_pct"].median()
        for part, rows in parts:
            scores.append(score_line("naive", np.nan, fold, val_season, part, rows,
                                     predict_naive(rows, fill)))

        # 3. and 4. Ridge and lasso at every alpha of their grid.
        for family, make, alphas in [("ridge", make_ridge, ALPHA_GRID),
                                     ("lasso", make_lasso, LASSO_ALPHAS)]:
            for alpha in alphas:
                model = make(alpha)
                # Catch the warnings of fit, so that we can count them. A
                # ConvergenceWarning means the lasso stopped after max_iter
                # rounds before its weights had settled.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(train_part[FEATURES], train_part[TARGET])   # training part only
                coef = model.steps[-1][1].coef_          # 1 weight per input, on the scaled inputs
                for name, weight in zip(FEATURES, coef):
                    weights.append({"model": family, "alpha": alpha, "fold": fold,
                                    "val_season": val_season, "feature": name, "weight": weight})
                linear.append({"model": family, "alpha": alpha, "fold": fold, "val_season": val_season,
                               "nonzero": int((coef != 0).sum()),
                               "dropped": ", ".join(n for n, w in zip(FEATURES, coef) if w == 0),
                               "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
                               "other_warnings": sum(not issubclass(w.category, ConvergenceWarning) for w in caught)})
                for part, rows in parts:
                    scores.append(score_line(family, alpha, fold, val_season, part, rows,
                                             model.predict(rows[FEATURES])))
        print("  fold", fold, "of", len(VALIDATION_SEASONS), "done")
    return pd.DataFrame(scores), pd.DataFrame(weights), pd.DataFrame(linear)


def summary_table(scores):
    """Mean and standard deviation over the 5 folds, for every candidate and
    part. The standard deviation is the sample one (ddof = 1)."""
    numbers = SCORES + ["list_mae_no_history", "list_mae_with_history"]
    # dropna=False keeps the constant and the naive rule, which have no alpha.
    table = scores.groupby(["model", "alpha", "part"], dropna=False)[numbers].agg(["mean", "std"])
    table.columns = [score + "_" + what for score, what in table.columns]
    table = table.reset_index()
    order = {"constant": 1, "naive": 2, "ridge": 3, "lasso": 4}
    table["order"] = table["model"].map(order)
    table = table.sort_values(["order", "alpha", "part"]).drop(columns="order")
    return table.reset_index(drop=True)


def ladder_table(summary):
    """The ladder: every candidate, the 5 scores, mean and std, train and validation."""
    columns = ["model", "alpha", "part"] + [s + "_" + w for s in SCORES for w in ["mean", "std"]]
    return summary[columns]


def alpha_table(summary):
    """Mean list MAE of ridge and lasso for each alpha, on train and on validation."""
    linear = summary[summary["model"].isin(["ridge", "lasso"])]
    table = linear.pivot_table(index=["model", "alpha"], columns="part",
                               values="list_mae_mean").reset_index()
    table = table.rename(columns={"train": "train_list_mae", "validation": "val_list_mae"})
    table["gap_val_minus_train"] = table["val_list_mae"] - table["train_list_mae"]
    return table


def history_table(summary):
    """Validation list MAE of every candidate for the rows without and with history."""
    val = summary[summary["part"] == "validation"]
    return val[["model", "alpha", "list_mae_mean", "list_mae_std",
                "list_mae_no_history_mean", "list_mae_no_history_std",
                "list_mae_with_history_mean", "list_mae_with_history_std"]].reset_index(drop=True)


def load_output(name):
    """Load 1 of the tables that the commands saved in outputs/."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, name + ".csv"), dtype={"val_season": str})


def run_baselines():
    """The whole step 4: score, save the tables, print. Nothing is chosen."""
    scores, weights, linear = score_baselines()
    summary = summary_table(scores)
    ladder = ladder_table(summary)
    alphas = alpha_table(summary)
    history = history_table(summary)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tables = {"baselines_scores_per_fold": scores, "baselines_summary": summary,
              "baselines_ladder": ladder, "baselines_alpha": alphas,
              "baselines_history": history, "baselines_weights": weights,
              "baselines_linear": linear}
    for name, table in tables.items():
        table.to_csv(os.path.join(OUTPUT_DIR, name + ".csv"), index=False)

    val = ladder[ladder["part"] == "validation"]
    print("\n--- validation, mean and std over the 5 folds ---")
    print(val[["model", "alpha"] + [s + "_" + w for s in SCORES for w in ["mean", "std"]]]
          .round(4).to_string(index=False))
    print("\n--- ridge and lasso: mean list MAE by alpha ---")
    print(alphas.round(5).to_string(index=False))
    print("\n--- validation list MAE, rows without and with history ---")
    print(history.round(4).to_string(index=False))
    print("\n--- lasso: nonzero weights per fold, warnings ---")
    lasso = linear[linear["model"] == "lasso"]
    print(lasso.pivot(index="alpha", columns="fold", values="nonzero").to_string())
    print("convergence warnings, all fits:", int(linear["convergence_warnings"].sum()),
          "  other warnings:", int(linear["other_warnings"].sum()))
    print("\nsaved in " + OUTPUT_DIR + "/:", ", ".join(name + ".csv" for name in tables))


# ------------------------------------------------------------------ models
# Step 5: 4 model families (ridge, lasso, random forest, gradient boosting)
# on 2 input sets, scored with the list metrics on the 5 walk forward folds,
# and the final choice. A "candidate" is 1 family + 1 input set + 1 setting:
# alpha (ridge, lasso), depth (forest), learning rate and depth (boosting).
# The selection rule was written down before any step 5 result was seen.

def make_forest(depth):
    """The random forest pipeline: median fill, then the forest. No scaler:
    a tree only asks "is the value above or below a limit", so the scale of
    an input does not matter. n_jobs=-1 only uses all processor cores."""
    if depth == NO_LIMIT:
        max_depth = None                 # how scikit-learn writes "no limit"
    else:
        max_depth = int(depth)
    return Pipeline([("fill", SimpleImputer(strategy="median")),
                     ("forest", RandomForestRegressor(n_estimators=FOREST_TREES,
                                                      max_features=FOREST_MAX_FEATURES,
                                                      min_samples_leaf=FOREST_MIN_LEAF,
                                                      max_depth=max_depth,
                                                      random_state=FOREST_SEED,
                                                      n_jobs=-1))])


def make_boosting(learning_rate, depth):
    """Gradient boosting: trees are added one after the other, each one
    corrects the errors of the trees before it, scaled down by the learning
    rate. No fill and no scaler: this model handles empty values itself.
    Early stopping: 10 % of the training rows are kept apart, and adding
    trees stops when they no longer help on these rows (at most 2000)."""
    return HistGradientBoostingRegressor(max_iter=BOOST_MAX_ITER, early_stopping=True,
                                         learning_rate=learning_rate, max_depth=int(depth),
                                         random_state=BOOST_SEED)


def make_model(family, value, learning_rate=np.nan):
    """Build the model of a candidate."""
    if family == "ridge":
        return make_ridge(value)
    if family == "lasso":
        return make_lasso(value)
    if family == "forest":
        return make_forest(value)
    return make_boosting(learning_rate, value)


def candidate_name(family, inputs, value, learning_rate=np.nan):
    """A short name, like 'ridge NO_LOAD alpha=10' or 'boosting FULL lr=0.05 depth=4'."""
    if family in ["ridge", "lasso"]:
        return family + " " + inputs + " alpha=" + format(value, "g")
    if family == "forest":
        return "forest " + inputs + " depth=" + ("None" if value == NO_LIMIT else str(int(value)))
    return "boosting " + inputs + " lr=" + format(learning_rate, "g") + " depth=" + str(int(value))


def score_candidates(candidates, folds_function=make_folds):
    """Fit and score a list of candidates (family, inputs, value, learning_rate)
    on the training part and the validation part of each fold.
    1 line per candidate, fold and part."""
    df = load_features()            # the test seasons are already removed here
    lines = []
    fold = 0
    for val_season, train_part, val_part in folds_function(df):
        fold = fold + 1
        for family, inputs, value, learning_rate in candidates:
            columns = INPUT_SETS[inputs]
            model = make_model(family, value, learning_rate)
            # fit sees the training part of this fold only. The warnings are
            # caught, so that we can count them.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train_part[columns], train_part[TARGET])
            info = {"family": family, "inputs": inputs, "value": value,
                    "learning_rate": learning_rate,
                    "name": candidate_name(family, inputs, value, learning_rate),
                    "n_inputs": len(columns), "fold": fold, "val_season": val_season,
                    "nonzero": np.nan, "trees_used": np.nan,
                    "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
                    "other_warnings": sum(not issubclass(w.category, ConvergenceWarning) for w in caught)}
            if family == "lasso":
                info["nonzero"] = int((model.named_steps["lasso"].coef_ != 0).sum())
            if family == "forest":
                info["trees_used"] = FOREST_TREES
            if family == "boosting":
                info["trees_used"] = model.n_iter_           # trees kept by early stopping
            for part, rows in [("train", train_part), ("validation", val_part)]:
                pred = model.predict(rows[columns])
                line = dict(info, part=part, n_rows=len(rows))
                line.update(all_metrics(rows, pred))
                line.update(history_list_mae(rows, pred))
                lines.append(line)
        print("  fold", fold, "of", len(VALIDATION_SEASONS), "done")
    return pd.DataFrame(lines)


def candidate_table(scores):
    """1 line per candidate: mean and standard deviation over the 5 folds of
    every score, on train and on validation (sample std, ddof = 1).
    SE = standard deviation of the 5 validation list MAEs / sqrt(5)."""
    keys = ["family", "inputs", "value", "learning_rate", "name", "n_inputs"]
    numbers = SCORES + ["list_mae_no_history", "list_mae_with_history"]
    parts = []
    for part, short in [("validation", "val"), ("train", "train")]:
        rows = scores[scores["part"] == part]
        table = rows.groupby(keys, dropna=False)[numbers].agg(["mean", "std"])
        table.columns = [short + "_" + score + "_" + what for score, what in table.columns]
        parts.append(table)
    table = pd.concat(parts, axis=1).reset_index()
    val = scores[scores["part"] == "validation"]
    extra = val.groupby("name").agg(max_nonzero=("nonzero", "max"), trees_used_mean=("trees_used", "mean"),
                                    convergence_warnings=("convergence_warnings", "sum"),
                                    other_warnings=("other_warnings", "sum"))
    table = table.merge(extra.reset_index(), on="name")
    n_folds = scores["fold"].nunique()
    table["val_list_mae_se"] = table["val_list_mae_std"] / np.sqrt(n_folds)
    table["gap_val_minus_train"] = table["val_list_mae_mean"] - table["train_list_mae_mean"]
    # The number of inputs of the simplicity rule: for lasso the largest
    # number of nonzero weights over the 5 folds, for the others the input set.
    table["simplicity_inputs"] = np.where(table["family"] == "lasso", table["max_nonzero"], table["n_inputs"])
    table = table.sort_values("val_list_mae_mean").reset_index(drop=True)
    table["raw_rank"] = table.index + 1                       # 1 = lowest mean validation list MAE
    return table


def simplest_first(table):
    """Sort candidates from the simplest to the most complex:
    a) fewer inputs (lasso: the largest number of nonzero weights over the folds),
    b) ridge, lasso, forest, boosting,
    c) ridge and lasso: larger alpha first; forest: smaller depth first;
       boosting: smaller depth first, then fewer trees used (mean over folds)."""
    table = table.copy()
    table["key_family"] = table["family"].map(FAMILY_ORDER)
    # For ridge and lasso a LARGER alpha is simpler, so we turn the sign around.
    table["key_value"] = np.where(table["family"].isin(["ridge", "lasso"]), -table["value"], table["value"])
    table["key_trees"] = np.where(table["family"] == "boosting", table["trees_used_mean"], 0)
    table = table.sort_values(["simplicity_inputs", "key_family", "key_value", "key_trees"])
    return table.drop(columns=["key_family", "key_value", "key_trees"])


def one_se_rule(table):
    """The one standard error rule on a table of candidates."""
    best = table.loc[table["val_list_mae_mean"].idxmin()]          # lowest mean validation list MAE
    threshold = best["val_list_mae_mean"] + best["val_list_mae_se"]  # best mean + its SE
    eligible = table[table["val_list_mae_mean"] <= threshold]
    chosen = simplest_first(eligible).iloc[0]
    return {"best": best["name"], "best_mean": best["val_list_mae_mean"],
            "se": best["val_list_mae_se"], "threshold": threshold,
            "eligible": simplest_first(eligible)["name"].tolist(), "chosen": chosen["name"],
            "chosen_mean": chosen["val_list_mae_mean"], "chosen_raw_rank": int(chosen["raw_rank"])}


def grid_edge(line):
    """Does a candidate sit at the edge of its grid? Returns the extra values
    for ridge and lasso (2 steps of 10 beyond the edge), None when it is not
    at an edge, and the word "undefined" for forest and boosting, where the
    brief does not say how to go beyond the edge."""
    family, value = line["family"], line["value"]
    if family in ["ridge", "lasso"]:
        grid = ALPHA_GRID if family == "ridge" else LASSO_ALPHAS
        if value == max(grid):
            return [value * 10 ** step for step in range(1, EDGE_STEPS + 1)]
        if value == min(grid):
            return [value / 10 ** step for step in range(1, EDGE_STEPS + 1)]
        return None
    if family == "forest":
        return "undefined" if value in [min(DEPTH_GRID), NO_LIMIT] else None
    edge_rate = line["learning_rate"] in [min(BOOST_RATES), max(BOOST_RATES)]
    edge_depth = value in [min(BOOST_DEPTHS), max(BOOST_DEPTHS)]
    return "undefined" if edge_rate or edge_depth else None


def fold_values(scores, name, column="list_mae"):
    """The 5 validation values of 1 candidate, 1 per fold, in fold order."""
    rows = scores[(scores["name"] == name) & (scores["part"] == "validation")]
    return rows.sort_values("fold")[column].to_numpy()


def compare(scores, table, name_a, name_b, what):
    """2 candidates side by side: the 2 mean validation list MAEs, the
    difference (a minus b) in each fold, its mean and its SE."""
    diff = fold_values(scores, name_a) - fold_values(scores, name_b)
    by_name = table.set_index("name")
    line = {"comparison": what, "a": name_a, "b": name_b,
            "a_mean": by_name.loc[name_a, "val_list_mae_mean"],
            "b_mean": by_name.loc[name_b, "val_list_mae_mean"],
            "diff_mean": diff.mean(),
            # SE of the difference: sample std of the 5 fold differences / sqrt(5).
            "diff_se": diff.std(ddof=1) / np.sqrt(len(diff))}
    for number, value in enumerate(diff):
        line["diff_fold_" + str(number + 1)] = value
    return line


def best_name(table, family=None, inputs=None):
    """The candidate with the lowest mean validation list MAE, inside 1 family
    and / or 1 input set if they are given."""
    if family is not None:
        table = table[table["family"] == family]
    if inputs is not None:
        table = table[table["inputs"] == inputs]
    return table.loc[table["val_list_mae_mean"].idxmin(), "name"]


def learning_curve(table, names):
    """Train the given candidates on 20, 40, 60, 80 and 100 % of the training
    RACES of each fold (whole races, because a result list needs whole races)
    and score them on the same validation part."""
    df = load_features()
    lines = []
    fold = 0
    for val_season, train_part, val_part in make_folds(df):
        fold = fold + 1
        races = pd.Series(train_part["race_id"].unique())
        for share in CURVE_SHARES:
            # A random subset of whole races, always the same because of the seed.
            picked = races.sample(frac=share, random_state=CURVE_SEED)
            subset = train_part[train_part["race_id"].isin(picked)]
            for name in names:
                cand = table.set_index("name").loc[name]
                columns = INPUT_SETS[cand["inputs"]]
                model = make_model(cand["family"], cand["value"], cand["learning_rate"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(subset[columns], subset[TARGET])
                lines.append({"name": name, "fold": fold, "val_season": val_season, "share": share,
                              "n_train_races": subset["race_id"].nunique(), "n_train_rows": len(subset),
                              "train_list_mae": list_mae(subset, model.predict(subset[columns])),
                              "val_list_mae": list_mae(val_part, model.predict(val_part[columns]))})
    return pd.DataFrame(lines)


def importance_table(name, table):
    """FIRST LOOK ONLY. Fit the given forest on all training seasons and list
    its feature_importances_ (they add up to 1), sorted."""
    cand = table.set_index("name").loc[name]
    columns = INPUT_SETS[cand["inputs"]]
    df = load_features()
    train = df[df["season"].isin(TRAIN_SEASONS)]
    model = make_forest(cand["value"])
    model.fit(train[columns], train[TARGET])
    out = pd.DataFrame({"feature": columns,
                        "importance": model.named_steps["forest"].feature_importances_})
    out["forest"] = name
    out["label"] = "first look only"
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def run_models():
    """The whole step 5. Every rule is applied in the order it was written down."""
    # 1. The 4 families on the 2 input sets, with their grids.
    candidates = []
    for inputs in INPUT_SETS:
        candidates += [("ridge", inputs, alpha, np.nan) for alpha in ALPHA_GRID]
        candidates += [("lasso", inputs, alpha, np.nan) for alpha in LASSO_ALPHAS]
        candidates += [("forest", inputs, depth, np.nan) for depth in DEPTH_GRID]
        candidates += [("boosting", inputs, depth, rate) for rate in BOOST_RATES for depth in BOOST_DEPTHS]
    print("scoring", len(candidates), "candidates of the grids")
    scores = score_candidates(candidates)
    scores["round"] = "grid"

    # 2. Forest refinement, once, per input set: the 2 depths halfway to the
    # neighbours of the best depth.
    table = candidate_table(scores)
    extra = []
    for inputs in INPUT_SETS:
        forest = table[(table["family"] == "forest") & (table["inputs"] == inputs)]
        best_depth = forest.loc[forest["val_list_mae_mean"].idxmin(), "value"]
        print("forest", inputs, ": best depth of the grid:", best_depth, " added:", EXTRA_DEPTHS[best_depth])
        extra += [("forest", inputs, depth, np.nan) for depth in EXTRA_DEPTHS[best_depth]]
    more = score_candidates(extra)
    more["round"] = "forest refinement"
    scores = pd.concat([scores, more], ignore_index=True)

    # 3. Selection: the one standard error rule on all candidates together.
    table = candidate_table(scores)
    choice = one_se_rule(table)
    first_choice = choice["chosen"]
    edge = grid_edge(table[table["name"] == first_choice].iloc[0])
    edge_rule = "not used"
    stop_message = None
    if isinstance(edge, list):
        # 4. Grid edge: extend the grid of this family by 2 steps, for both input sets, once.
        family = table[table["name"] == first_choice]["family"].iloc[0]
        edge_rule = "used: " + family + " grid extended by " + ", ".join(format(v, "g") for v in edge)
        print("grid edge:", edge_rule)
        more = score_candidates([(family, inputs, value, np.nan) for inputs in INPUT_SETS for value in edge])
        more["round"] = "grid edge"
        scores = pd.concat([scores, more], ignore_index=True)
        table = candidate_table(scores)
        choice = one_se_rule(table)              # the selection again, only this once
    elif edge == "undefined":
        edge_rule = "STOP: the choice sits at the edge of a grid where the extension is not defined"
        stop_message = edge_rule

    final = choice["chosen"]
    order = {name: number + 1 for number, name in enumerate(choice["eligible"])}
    table["eligible"] = table["name"].isin(choice["eligible"])
    table["simplicity_order"] = table["name"].map(order)
    table["lowest_mean"] = table["name"] == choice["best"]
    table["final_model"] = table["name"] == final
    selection = pd.DataFrame([{"best": choice["best"], "best_mean": choice["best_mean"],
                               "se": choice["se"], "threshold": choice["threshold"],
                               "n_candidates": len(table), "n_eligible": len(choice["eligible"]),
                               "final_model": final, "final_mean": choice["chosen_mean"],
                               "final_raw_rank": choice["chosen_raw_rank"],
                               "edge_rule": edge_rule, "choice_before_edge_rule": first_choice}])

    # 5. Reports that do not change the choice.
    groups = []
    for family in FAMILY_ORDER:
        for inputs in INPUT_SETS:
            name = best_name(table, family, inputs)
            line = table[table["name"] == name].iloc[0]
            groups.append({"family": family, "inputs": inputs, "best": name,
                           "val_list_mae_mean": line["val_list_mae_mean"], "val_list_mae_std": line["val_list_mae_std"],
                           "val_spearman_mean": line["val_spearman_mean"], "val_top10_mean": line["val_top10_mean"],
                           "train_list_mae_mean": line["train_list_mae_mean"], "gap_val_minus_train": line["gap_val_minus_train"],
                           "val_list_mae_no_history_mean": line["val_list_mae_no_history_mean"],
                           "val_list_mae_with_history_mean": line["val_list_mae_with_history_mean"]})
    groups = pd.DataFrame(groups)
    comparisons = pd.DataFrame([compare(scores, table, best_name(table, family, "FULL"),
                                        best_name(table, family, "NO_LOAD"),
                                        "load, " + family + ": with minus without")
                                for family in FAMILY_ORDER])
    history = groups[["family", "inputs", "best", "val_list_mae_mean",
                      "val_list_mae_no_history_mean", "val_list_mae_with_history_mean"]]
    depth_curve = table[table["family"] == "forest"][
        ["inputs", "value", "name", "train_list_mae_mean", "val_list_mae_mean", "val_list_mae_se"]]
    depth_curve = depth_curve.sort_values(["inputs", "value"])
    boost = scores[(scores["family"] == "boosting") & (scores["part"] == "validation")]
    boosting_trees = boost.pivot(index="name", columns="val_season", values="trees_used")
    boosting_trees["mean"] = boosting_trees.mean(axis=1)
    boosting_trees = boosting_trees.reset_index()

    # Learning curve: the final model and the best candidate of each other family.
    final_family = table[table["name"] == final]["family"].iloc[0]
    names = [final] + [best_name(table, family) for family in FAMILY_ORDER if family != final_family]
    print("learning curve:", ", ".join(names))
    curve_folds = learning_curve(table, names)
    curve = curve_folds.groupby(["name", "share"])[["n_train_races", "n_train_rows",
                                                    "train_list_mae", "val_list_mae"]].mean().reset_index()
    curve["role"] = np.where(curve["name"] == final, "final model", "best of another family")

    importance = importance_table(best_name(table, family="forest"), table)

    # 6. Save and print.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tables = {"models_scores_per_fold": scores, "models_candidates": table,
              "models_selection": selection, "models_groups": groups,
              "models_comparisons": comparisons, "models_history": history,
              "models_depth_curve": depth_curve, "models_boosting_trees": boosting_trees,
              "models_learning_curve": curve, "models_learning_curve_per_fold": curve_folds,
              "models_importance": importance}
    for name, one in tables.items():
        one.to_csv(os.path.join(OUTPUT_DIR, name + ".csv"), index=False)

    print("\n--- all candidates, best mean validation list MAE first ---")
    show = ["raw_rank", "name", "simplicity_inputs", "train_list_mae_mean", "val_list_mae_mean",
            "val_list_mae_std", "val_spearman_mean", "val_top10_mean", "eligible", "final_model"]
    print(table[show].round(5).to_string(index=False))
    print("\n--- selection: the one standard error rule on all", len(table), "candidates ---")
    print("best     :", choice["best"], " mean:", round(choice["best_mean"], 5))
    print("SE       :", round(choice["se"], 5), "  threshold = best + SE:", round(choice["threshold"], 5))
    print("eligible :", len(choice["eligible"]), "of", len(table))
    print("simplest eligible first:")
    print(simplest_first(table[table["eligible"]])[["name", "simplicity_inputs", "val_list_mae_mean", "trees_used_mean"]]
          .round(5).to_string(index=False))
    print("grid edge:", edge_rule, "  (choice before: " + first_choice + ")")
    print("\n==============================================")
    print("THE FINAL MODEL:", final, "  mean validation list MAE:", round(choice["chosen_mean"], 5),
          "  raw rank:", choice["chosen_raw_rank"], "of", len(table))
    print("==============================================")
    print("\n--- best of each family and input set ---")
    print(groups.round(5).to_string(index=False))
    print("\n--- with load minus without load, per family ---")
    print(comparisons.round(5).to_string(index=False))
    print("\n--- boosting: trees used per fold ---")
    print(boosting_trees.round(1).to_string(index=False))
    print("\n--- learning curve: mean over the 5 folds ---")
    print(curve.round(5).to_string(index=False))
    print("\n--- forest importances, FIRST LOOK ONLY:", importance["forest"].iloc[0], "---")
    print(importance[["feature", "importance"]].round(4).to_string(index=False))
    print("\nwarnings: convergence", int(scores["convergence_warnings"].sum() / 2),
          "  other", int(scores["other_warnings"].sum() / 2), "(fits, each counted once)")
    print("\nsaved in " + OUTPUT_DIR + "/:", ", ".join(name + ".csv" for name in tables))
    if stop_message is not None:
        sys.exit(stop_message + ". Ask before going on.")


# ------------------------------------------------------------------ robustness
# The robustness run: training and validation seasons only. It changes
# nothing: it only tells us if the conclusions of step 5 hold with another
# split. Each validation season is predicted from the 5 seasons just before
# it (a rolling window), not from all earlier seasons.

def run_robustness():
    """Score exactly the 64 candidates of step 5 on the rolling window folds,
    and apply the same selection rule. No refinement round, no grid extension."""
    step5 = load_output("models_candidates")
    # The same 4 parts as in step 5: family, input set, value, learning rate.
    candidates = [(line["family"], line["inputs"], line["value"], line["learning_rate"])
                  for index, line in step5.iterrows()]
    print("scoring", len(candidates), "candidates of step 5 on the rolling window folds")
    scores = score_candidates(candidates, folds_function=make_window_folds)

    table = candidate_table(scores)
    choice = one_se_rule(table)
    final_step5 = step5[step5["final_model"]]["name"].iloc[0]
    order = {name: number + 1 for number, name in enumerate(choice["eligible"])}
    table["eligible"] = table["name"].isin(choice["eligible"])
    table["simplicity_order"] = table["name"].map(order)
    table["lowest_mean"] = table["name"] == choice["best"]
    table["one_se_choice"] = table["name"] == choice["chosen"]
    table["final_model_of_step_5"] = table["name"] == final_step5
    step5_selection = load_output("models_selection").iloc[0]
    selection = pd.DataFrame([{"best": choice["best"], "best_mean": choice["best_mean"],
                               "se": choice["se"], "threshold": choice["threshold"],
                               "n_candidates": len(table), "n_eligible": len(choice["eligible"]),
                               "one_se_choice": choice["chosen"], "one_se_choice_mean": choice["chosen_mean"],
                               "one_se_choice_raw_rank": choice["chosen_raw_rank"],
                               "final_model_of_step_5": final_step5,
                               "same_model": choice["chosen"] == final_step5,
                               "step5_best": step5_selection["best"], "step5_best_mean": step5_selection["best_mean"],
                               "step5_se": step5_selection["se"], "step5_threshold": step5_selection["threshold"],
                               "step5_n_eligible": step5_selection["n_eligible"]}])

    # The best of each family (over both input sets), next to step 5.
    groups = []
    for family in FAMILY_ORDER:
        name = best_name(table, family=family)
        line = table[table["name"] == name].iloc[0]
        name5 = best_name(step5, family=family)
        line5 = step5[step5["name"] == name5].iloc[0]
        groups.append({"family": family, "best": name, "val_list_mae_mean": line["val_list_mae_mean"],
                       "val_list_mae_std": line["val_list_mae_std"],
                       "step5_best": name5, "step5_val_list_mae_mean": line5["val_list_mae_mean"],
                       "step5_val_list_mae_std": line5["val_list_mae_std"]})
    groups = pd.DataFrame(groups)

    # With load minus without load, per family.
    comparisons = pd.DataFrame([compare(scores, table, best_name(table, family, "FULL"),
                                        best_name(table, family, "NO_LOAD"),
                                        "load, " + family + ": with minus without")
                                for family in FAMILY_ORDER])

    # The final model of step 5: rolling window against all earlier seasons, per fold.
    step5_scores = load_output("models_scores_per_fold")
    per_fold = pd.DataFrame({
        "val_season": VALIDATION_SEASONS,
        "all_earlier_list_mae": fold_values(step5_scores, final_step5),
        "rolling_window_list_mae": fold_values(scores, final_step5)})
    per_fold["window_minus_all_earlier"] = per_fold["rolling_window_list_mae"] - per_fold["all_earlier_list_mae"]
    train_rows = scores[(scores["name"] == final_step5) & (scores["part"] == "train")].sort_values("fold")
    per_fold["rolling_window_train_rows"] = train_rows["n_rows"].to_numpy()
    train5 = step5_scores[(step5_scores["name"] == final_step5) & (step5_scores["part"] == "train")].sort_values("fold")
    per_fold["all_earlier_train_rows"] = train5["n_rows"].to_numpy()
    per_fold["model"] = final_step5

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tables = {"robustness_scores_per_fold": scores, "robustness_candidates": table,
              "robustness_selection": selection, "robustness_groups": groups,
              "robustness_comparisons": comparisons, "robustness_final_per_fold": per_fold}
    for name, one in tables.items():
        one.to_csv(os.path.join(OUTPUT_DIR, name + ".csv"), index=False)

    print("\n--- rolling window: the one standard error rule (step 5 in brackets) ---")
    print("best      :", choice["best"], round(choice["best_mean"], 5),
          " (", step5_selection["best"], round(step5_selection["best_mean"], 5), ")")
    print("SE        :", round(choice["se"], 5), " (", round(step5_selection["se"], 5), ")")
    print("threshold :", round(choice["threshold"], 5), " (", round(step5_selection["threshold"], 5), ")")
    print("eligible  :", len(choice["eligible"]), "of", len(table), " (", step5_selection["n_eligible"], ")")
    print("choice    :", choice["chosen"], round(choice["chosen_mean"], 5), " raw rank", choice["chosen_raw_rank"],
          " (", final_step5, ")   same model:", choice["chosen"] == final_step5)
    print("\n--- best of each family ---")
    print(groups.round(5).to_string(index=False))
    print("\n--- with load minus without load ---")
    print(comparisons.round(5).to_string(index=False))
    print("\n--- final model of step 5: rolling window against all earlier seasons ---")
    print(per_fold.round(5).to_string(index=False))
    print("\nsaved in " + OUTPUT_DIR + "/:", ", ".join(name + ".csv" for name in tables))


# ------------------------------------------------------------------ final test
# Step 6: the final test, walk forward like the validation folds of make_folds.
# Each test season is predicted by models fitted on every season before it:
#   2023/24: fit on 2013/14 to 2022/23
#   2024/25: fit on 2013/14 to 2023/24
#   2025/26: fit on 2013/14 to 2024/25
# The history season 2012/13 is never trained on. The functions below get the
# training rows and the test rows as 2 tables. Only run_test() and
# load_test_data() load the real test rows.

def load_test_rows():
    """Load the rows of the TEST seasons from data/features.csv.
    Only the command 'python pipeline.py test' may call this function."""
    if not TEST_UNLOCKED:
        sys.exit("Step 6 is locked: TEST_UNLOCKED is False in pipeline.py. The test seasons were not read.")
    df = pd.read_csv(FEATURES_FILE, dtype={"season": str})
    return df[df["season"].isin(TEST_SEASONS)].reset_index(drop=True)


def load_test_predictions():
    """Load the per athlete predictions that the test command saved in data/."""
    return pd.read_csv(TEST_PREDICTIONS_FILE, dtype={"season": str})


def fit_final_model(train):
    """Fit the frozen final model on the given training rows. make_model
    builds the same pipeline as in step 5: median fill, scale, lasso with
    alpha 0.01 and at most 10,000 rounds."""
    model = make_model(FINAL_FAMILY, FINAL_VALUE)
    model.fit(train[INPUT_SETS[FINAL_INPUTS]], train[TARGET])
    return model


def fit_baselines(train):
    """The 2 simple models learn their single number from the same training
    rows as the final model, as in step 4: the constant takes the mean of the
    target, the naive rule fills empty cells with the median of last3_finish_pct."""
    return {"constant_value": train[TARGET].mean(),
            "naive_fill": train["last3_finish_pct"].median()}


def add_predictions(model, baselines, rows):
    """Add the predictions of the 3 models to the rows, then the predicted
    list of each model. The list is made on the WHOLE race, before any
    slice is cut."""
    rows = rows.copy()
    rows["pred_final"] = model.predict(rows[INPUT_SETS[FINAL_INPUTS]])
    rows["pred_constant"] = predict_constant(rows, baselines["constant_value"])
    rows["pred_naive"] = predict_naive(rows, baselines["naive_fill"])
    # The real place, found again from the percentile: pct = (place - 1) / (n - 1).
    rows["n_finishers"] = rows.groupby("race_id")[TARGET].transform("count")
    rows["real_place"] = (rows[TARGET] * (rows["n_finishers"] - 1)).round().astype(int) + 1
    # The predicted list of each model, written as a percentile (the same
    # function as the list metrics, so the same tie break by the race seed).
    for name, column in TEST_MODELS.items():
        rows["list_pct_" + name] = list_percentile(rows, rows[column])
    # The place of the athlete in the predicted list of the final model, 1 to n.
    rows["pred_place"] = (rows["list_pct_final"] * (rows["n_finishers"] - 1)).round().astype(int) + 1
    # The error of the final model: predicted percentile from the list minus the real one.
    rows["error"] = rows["list_pct_final"] - rows[TARGET]
    rows["abs_error"] = rows["error"].abs()
    return rows


def test_score_table(rows):
    """The 5 scores of the 3 models: all seasons together, then each season."""
    scopes = [("all", rows)]
    for season in sorted(rows["season"].unique()):
        scopes.append((season, rows[rows["season"] == season]))
    lines = []
    for scope, part in scopes:
        for model, column in TEST_MODELS.items():
            line = {"scope": scope, "model": model, "rows": len(part),
                    "races": part["race_id"].nunique()}
            line.update(all_metrics(part, part[column]))
            lines.append(line)
    return pd.DataFrame(lines)


def test_validation_table(scores):
    """The 5 scores on the 3 test seasons together, next to the validation
    scores (mean over the 5 folds). Final model: from step 5. Constant and
    naive rule: from step 4 (the same folds). Only open seasons are read here."""
    candidates = load_output("models_candidates").set_index("name")
    final = candidates.loc[candidate_name(FINAL_FAMILY, FINAL_INPUTS, FINAL_VALUE)]
    ladder = load_output("baselines_ladder")
    ladder = ladder[ladder["part"] == "validation"].set_index("model")
    test = scores[scores["scope"] == "all"].set_index("model")
    lines = []
    for model in ["constant", "naive", "final"]:
        for score in SCORES:
            if model == "final":
                validation = final["val_" + score + "_mean"]
            else:
                validation = ladder.loc[model, score + "_mean"]
            lines.append({"model": model, "score": score, "validation": validation,
                          "test": test.loc[model, score],
                          "test_minus_validation": test.loc[model, score] - validation})
    return pd.DataFrame(lines)


def test_race_table(rows):
    """1 line per race: the list MAE of the 3 models. No athlete in this table."""
    lines = []
    for race_id, race in rows.groupby("race_id"):
        line = {"race_id": race_id, "season": race["season"].iloc[0],
                "day": race["start"].iloc[0][:10], "place": race["place"].iloc[0],
                "gender": race["gender"].iloc[0],
                "is_major_event": race["is_major_event"].iloc[0], "rows": len(race)}
        for model in TEST_MODELS:
            # The list of the whole race was made in add_predictions.
            line["list_mae_" + model] = (race["list_pct_" + model] - race[TARGET]).abs().mean()
        lines.append(line)
    table = pd.DataFrame(lines)
    # Number the races from the lowest list MAE of the final model to the highest.
    table = table.sort_values(["list_mae_final", "race_id"]).reset_index(drop=True)
    table["number_by_list_mae"] = table.index + 1
    table["story_race"] = table["number_by_list_mae"] == STORY_NUMBER
    return table.sort_values(["day", "race_id"]).reset_index(drop=True)


def test_slice_table(rows):
    """For each group of the 5 slices: the mean absolute error and the mean
    signed error (predicted percentile from the list minus real percentile)
    of the final model and of the naive rule. The lists were made on the
    whole race first, then the rows are split into groups."""
    real = rows[TARGET]
    sprints = rows["n_sprints_window"]
    slices = [
        ("1) championship", "World Championships and Olympics", rows["is_major_event"] == True),
        ("1) championship", "other sprints", rows["is_major_event"] == False),
        ("2) history", "with history (no_history = 0)", rows["no_history"] == 0),
        ("2) history", "without history (no_history = 1)", rows["no_history"] == 1),
        ("3) gender", "men", rows["gender"] == "SM"),
        ("3) gender", "women", rows["gender"] == "SW"),
        ("4) real finish position", "top 10 % (up to 0.10)", real <= PCT_FRONT),
        ("4) real finish position", "middle", (real > PCT_FRONT) & (real < PCT_BACK)),
        ("4) real finish position", "bottom 10 % (0.90 and above)", real >= PCT_BACK),
        ("5) n_sprints_window", "0 to 3", sprints <= SPRINTS_FEW),
        ("5) n_sprints_window", "4 to 9", (sprints > SPRINTS_FEW) & (sprints < SPRINTS_MANY)),
        ("5) n_sprints_window", "10 or more", sprints >= SPRINTS_MANY),
    ]
    lines = []
    for slice_name, group, keep in slices:
        part = rows[keep]
        for model in SLICE_MODELS:
            signed = part["list_pct_" + model] - part[TARGET]
            lines.append({"slice": slice_name, "group": group, "model": model,
                          "rows": len(part), "races": part["race_id"].nunique(),
                          "mean_abs_error": signed.abs().mean(),
                          "mean_signed_error": signed.mean()})
    return pd.DataFrame(lines)


def predict_by_season(models, rows, X):
    """Predict every row with the model of its own test season.
    models: 1 fitted model per test season. X: the inputs of the rows."""
    pred = np.zeros(len(rows))
    for season, model in models.items():
        keep = (rows["season"] == season).to_numpy()     # the rows of this season
        pred[keep] = model.predict(X[keep])
    return pred


def test_permutation_table(models, rows):
    """Permutation importance of the final model on the test rows: shuffle
    1 input (or 1 group of inputs) between the rows of all 3 test seasons,
    so that it no longer belongs to the right athlete, and see how much the
    list MAE goes up. Every row is predicted by the model of its own season,
    before and after the shuffle. A big increase means the model needs this
    input. 20 repeats, sample standard deviation (ddof = 1)."""
    columns = INPUT_SETS[FINAL_INPUTS]
    X = rows[columns]
    # An input counts as dropped only if the lasso set its weight to 0 in all 3 fits.
    weights = pd.DataFrame({season: model.named_steps["lasso"].coef_
                            for season, model in models.items()}, index=columns)
    zero_in_all = (weights == 0).all(axis=1)
    base = list_mae(rows, predict_by_season(models, rows, X))      # nothing shuffled

    # a) each input alone, b) each group: the columns of a group get the SAME
    # new row order, so the values of the group stay together.
    parts = [("a) each input alone", name, [name]) for name in columns]
    parts = parts + [("b) by group", group, members) for group, members in INPUT_GROUPS.items()]
    lines = []
    for version, name, members in parts:
        rng = np.random.RandomState(PERM_SEED)             # the same seed for every input and group
        increases = []
        for repeat in range(PERM_REPEATS):
            order = rng.permutation(len(X))                # a random order of the rows
            shuffled = X.copy()
            shuffled[members] = X[members].to_numpy()[order]
            increases.append(list_mae(rows, predict_by_season(models, rows, shuffled)) - base)
        increases = np.array(increases)
        lines.append({"version": version, "name": name, "inputs": ", ".join(members),
                      # True if every input here has a weight of 0 in all 3 fits.
                      "lasso_weight_zero": bool(zero_in_all[members].all()),
                      "list_mae_increase_mean": increases.mean(),
                      "list_mae_increase_std": increases.std(ddof=1)})

    table = pd.DataFrame(lines)
    table["list_mae_not_shuffled"] = base
    table["repeats"] = PERM_REPEATS
    return table.sort_values(["version", "list_mae_increase_mean"], ascending=[True, False]).reset_index(drop=True)


def final_weights_table(models):
    """The weights of the 3 fitted final lassos (1 per test season), 1 block
    per season, each block sorted by size (the zeros last)."""
    blocks = []
    for season, model in models.items():
        table = season_weights_table(model)
        table.insert(0, "test_season", season)          # the season this fit predicts
        blocks.append(table)
    return pd.concat(blocks, ignore_index=True)


def season_weights_table(model):
    """The weights of 1 fitted final lasso, on the scaled inputs, sorted by
    size (the zeros last), with the numbers that the fill step and the scale
    step learned."""
    lasso = model.named_steps["lasso"]
    table = pd.DataFrame({"feature": INPUT_SETS[FINAL_INPUTS],
                          # + 0.0 turns a weight of -0.0 into 0.0, so that no "-0" is printed.
                          "weight": lasso.coef_ + 0.0,
                          "fill_median": model.named_steps["fill"].statistics_,
                          "scale_mean": model.named_steps["scale"].mean_,
                          "scale_spread": model.named_steps["scale"].scale_})
    table["size"] = table["weight"].abs()
    table["dropped"] = table["weight"] == 0               # the lasso set this weight to 0
    table = table.sort_values(["size", "feature"], ascending=[False, True]).reset_index(drop=True)
    table["alpha"] = lasso.alpha
    table["intercept"] = lasso.intercept_
    table["rounds_used"] = lasso.n_iter_                  # rounds the lasso needed (at most 10,000)
    return table


def draw_story(race, title, path=None):
    """The story figure: place in the predicted list against real place in
    1 race, 1 dot per athlete, no names. If a path is given, the figure is saved."""
    import matplotlib.pyplot as plt
    n = len(race)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([1, n], [1, n], color="#52514e", linewidth=1, linestyle="--",
            label="perfect prediction")
    ax.scatter(race["real_place"], race["pred_place"], s=28, color="#2a78d6",
               edgecolor="white", linewidth=0.6, label="1 athlete")
    ax.set_xlabel("real place")
    ax.set_ylabel("place in the predicted list")
    ax.set_title(title)
    ax.set_xlim(0, n + 2)
    ax.set_ylim(0, n + 2)
    ax.set_aspect("equal")
    ax.grid(color="#e5e4e0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def story_title(race_line):
    """The title of the story figure, made from the line of the race table."""
    who = "men" if race_line["gender"] == "SM" else "women"
    return (race_line["place"] + ", " + race_line["day"] + ", " + who
            + "\nlist MAE " + format(race_line["list_mae_final"], ".3f")
            + " (number " + str(STORY_NUMBER) + " of the races sorted by list MAE)")


def test_tables(train, test):
    """Everything of step 6 for the given training rows and test rows, walk
    forward: for each test season, fit the final model, the constant and the
    naive rule on the training rows plus the test seasons before it, and
    predict the rows of that season only. Then the 3 seasons are put together
    and every table is built. Every warning is caught and counted. Nothing is
    saved here."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models = {}                                   # 1 fitted final model per test season
        baselines = {}                                # 1 constant value and naive fill per test season
        fit_rows = {}                                 # how many rows each fit used
        parts = []
        for number, season in enumerate(TEST_SEASONS):
            earlier = test[test["season"].isin(TEST_SEASONS[:number])]   # the test seasons before it
            rows_before = pd.concat([train, earlier])
            models[season] = fit_final_model(rows_before)
            baselines[season] = fit_baselines(rows_before)
            fit_rows[season] = len(rows_before)
            parts.append(add_predictions(models[season], baselines[season],
                                         test[test["season"] == season]))
        rows = pd.concat(parts, ignore_index=True)    # the 3 seasons together
        scores = test_score_table(rows)
        tables = {"models": models,
                  "baselines": baselines,
                  "fit_rows": fit_rows,
                  "predictions": rows,
                  "scores": scores,
                  "versus_validation": test_validation_table(scores),
                  "races": test_race_table(rows),
                  "slices": test_slice_table(rows),
                  "permutation": test_permutation_table(models, rows),
                  "weights": final_weights_table(models)}
    tables["warnings"] = [w.category.__name__ + ": " + str(w.message) for w in caught]
    return tables


def load_test_data():
    """Load the training rows (2013/14 to 2022/23) and the test rows of the
    final test, and check them. In the walk forward test the rows of a test
    season are also used to fit the models of the later test seasons, so the
    time check is done for each test season. If a check fails the command
    stops here, before any score exists."""
    df = load_features()                                   # open seasons only
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = load_test_rows()                                # THE TEST SEASONS
    # The name of the place of each race comes from the file with all races.
    places = load_races().drop_duplicates("race_id")[["race_id", "place"]]
    test = test.merge(places, on="race_id", how="left")

    problems = []
    if len(train) != EXPECTED_TRAIN_ROWS:
        problems.append("training rows: " + str(len(train)))
    if len(test) != EXPECTED_TEST_ROWS:
        problems.append("test rows: " + str(len(test)))
    if test["race_id"].nunique() != EXPECTED_TEST_RACES:
        problems.append("test races: " + str(test["race_id"].nunique()))
    # Each test season: every row its models are fitted on (the training rows
    # and the test seasons before it) must start before its first race.
    for number, season in enumerate(TEST_SEASONS):
        rows_before = pd.concat([train, test[test["season"].isin(TEST_SEASONS[:number])]])
        if not rows_before["start"].max() < test[test["season"] == season]["start"].min():
            problems.append("a race used to fit the models of " + season + " starts after its first race")
    if sorted(test["season"].unique()) != TEST_SEASONS:
        problems.append("test seasons: " + str(sorted(test["season"].unique())))
    # The groups of the permutation importance must hold each input exactly once.
    grouped = sorted(name for members in INPUT_GROUPS.values() for name in members)
    if grouped != sorted(INPUT_SETS[FINAL_INPUTS]):
        problems.append("INPUT_GROUPS do not match the final inputs")
    if len(problems) > 0:
        sys.exit("STOPPED before any test score was computed: " + "; ".join(problems))
    return train, test


def run_test():
    """Step 6: the final test, walk forward: each test season is predicted by
    the frozen final model (and the constant and the naive rule) refitted on
    every season before it. A rerun of the same code may only reproduce the
    same numbers."""
    train, test = load_test_data()
    print("training rows:", len(train), "  seasons", train["season"].min(), "to", train["season"].max(),
          "  latest start:", train["start"].max()[:10])
    print("test rows    :", len(test), "  races:", test["race_id"].nunique(),
          "  seasons", test["season"].min(), "to", test["season"].max(),
          "  earliest start:", test["start"].min()[:10])

    tables = test_tables(train, test)
    rows = tables["predictions"]
    races = tables["races"]

    # Save. outputs/ gets the tables without athletes and the figure.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = {"test_scores": tables["scores"], "test_vs_validation": tables["versus_validation"],
             "test_list_mae_per_race": races, "test_slices": tables["slices"],
             "test_permutation": tables["permutation"], "test_weights": tables["weights"]}
    for name, one in saved.items():
        one.to_csv(os.path.join(OUTPUT_DIR, name + ".csv"), index=False)
    # The per athlete predictions go to data/, which is not in git.
    keep = ["race_id", "season", "start", "place", "gender", "IBUId", "Name", "Nat",
            "is_major_event", TARGET, "no_history", "n_sprints_window", "n_finishers",
            "real_place", "pred_final", "list_pct_final", "pred_place",
            "pred_naive", "list_pct_naive", "pred_constant", "list_pct_constant",
            "error", "abs_error"]
    rows[keep].to_csv(TEST_PREDICTIONS_FILE, index=False)
    # The story figure: the race in the middle of the list MAE order.
    story = races[races["story_race"]].iloc[0]
    draw_story(rows[rows["race_id"] == story["race_id"]], story_title(story),
               os.path.join(OUTPUT_DIR, STORY_FIGURE))

    weights = tables["weights"]
    print("final model  :", candidate_name(FINAL_FAMILY, FINAL_INPUTS, FINAL_VALUE), ",",
          len(INPUT_SETS[FINAL_INPUTS]), "inputs, refitted for each test season")
    for season in TEST_SEASONS:
        one = weights[weights["test_season"] == season]
        print("  " + season + ": fitted on", tables["fit_rows"][season], "rows",
              "  nonzero weights:", int((~one["dropped"]).sum()),
              "  constant value:", round(tables["baselines"][season]["constant_value"], 5),
              "  naive fill:", round(tables["baselines"][season]["naive_fill"], 5))
    print("warnings     :", len(tables["warnings"]))
    for text in tables["warnings"]:
        print("   ", text)

    print("\n--- test (3 seasons together) next to validation (mean over the 5 folds) ---")
    print(tables["versus_validation"].round(5).to_string(index=False))
    print("\n--- test scores per season ---")
    print(tables["scores"].round(5).to_string(index=False))
    print("\n--- list MAE per race (the final model): summary of the", len(races), "races ---")
    print(races["list_mae_final"].describe().round(5).to_string())
    print("championship sprints:", int(races["is_major_event"].sum()), "(expected 6)")
    print("\n--- slices: mean absolute and mean signed error (list percentile minus real) ---")
    print(tables["slices"].round(5).to_string(index=False))
    permutation = tables["permutation"]
    print("\n--- permutation importance on the test rows: increase in list MAE ---")
    print(permutation[["version", "name", "lasso_weight_zero", "list_mae_increase_mean",
                       "list_mae_increase_std"]].round(5).to_string(index=False))
    zero = permutation[permutation["lasso_weight_zero"]]
    print("inputs with a lasso weight of 0 show exactly 0:",
          bool((zero["list_mae_increase_mean"] == 0).all() and (zero["list_mae_increase_std"] == 0).all()),
          " (" + str(len(zero)) + " lines)")
    print("\n--- weights of the 3 fitted final lassos, on the scaled inputs ---")
    for season in TEST_SEASONS:
        one = weights[weights["test_season"] == season]
        print("test season", season, "  intercept:", round(one["intercept"].iloc[0], 5),
              "  rounds used:", int(one["rounds_used"].iloc[0]))
        print(one[["feature", "weight", "dropped"]].round(5).to_string(index=False))
    print("\nstory race:", story["race_id"], story["place"], story["day"],
          " list MAE:", round(story["list_mae_final"], 5))
    print("\nsaved in " + OUTPUT_DIR + "/:", ", ".join(name + ".csv" for name in saved),
          "and", STORY_FIGURE)
    print("saved in data/ (not in git):", TEST_PREDICTIONS_FILE)


# ------------------------------------------------------------------ main step

def make_features():
    """The whole step 2: load, clean, target, features, save, checks."""
    sprints = load_sprints()
    races = load_races()

    fin = build_target(sprints)
    fin = add_sprint_history(fin)
    fin = add_window_history(fin)
    fin = add_race_history(fin, races)

    # Keep the columns of the brief, in race and rank order. Missing values
    # stay empty: they are filled later inside the model pipeline.
    fin = fin.sort_values(["start", "race_id", "Rank"])
    out = fin[["race_id", "season", "start", "gender", "IBUId", "Name", "Nat",
               "is_major_event", TARGET] + FEATURES]
    out.to_csv(FEATURES_FILE, index=False)

    print_checks(out, fin, races)


if __name__ == "__main__":
    # The word after "python pipeline.py" says which step to run.
    if len(sys.argv) == 2 and sys.argv[1] == "features":
        make_features()
    elif len(sys.argv) == 2 and sys.argv[1] == "split":
        show_split()
    elif len(sys.argv) == 2 and sys.argv[1] == "baselines":
        run_baselines()
    elif len(sys.argv) == 2 and sys.argv[1] == "models":
        run_models()
    elif len(sys.argv) == 2 and sys.argv[1] == "robustness":
        run_robustness()
    elif len(sys.argv) == 2 and sys.argv[1] == "test":
        run_test()
    else:
        print("usage: python pipeline.py features | split | baselines | models | robustness | test")
