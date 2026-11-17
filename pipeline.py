# pipeline.py
# The data pipeline of the project. The steps so far:
#
#   python pipeline.py features    build data/features.csv
#   python pipeline.py split       print the season split and the validation folds
#   python pipeline.py baselines   score the 3 baseline models, save the tables in outputs/
#   python pipeline.py models      ridge and random forest, choose the final model
#   python pipeline.py robustness  step 6, part A: the step 5 candidates on a sliding window
#   python pipeline.py test        step 6, part B: THE FINAL TEST. Opens the test seasons.
#
# About "features":
# It reads data/sprints_raw.csv and data/races_all.csv, cleans them, builds
# the target and the 13 features, and writes data/features.csv with
# 1 row per finisher per sprint.
#
# Golden rule: a feature of a race only uses races that started BEFORE it.

import os
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
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

# Step 4, the ridge model. Alpha says how strongly the weights are pulled
# towards 0. These rules were fixed BEFORE looking at any result:
# - try every alpha of the grid,
# - take the alpha with the lowest mean validation MAE over the 5 folds,
# - tie rule: round the mean MAE to 3 decimals; among the alphas that share
#   the best rounded value, take the largest (the simplest model),
# - edge rule: if the chosen alpha is the smallest or the largest of the
#   grid, add 2 more steps of 10 on that side and choose again. Only once.
ALPHA_GRID = [0.01, 0.1, 1, 10, 100, 1000]
MAE_DECIMALS = 3        # the rounding of the tie rule
EDGE_STEPS = 2          # how many steps of 10 the edge rule adds

# The value the simple models predict when they know nothing: the middle.
MIDDLE = 0.5

# The 5 scores of step 4, in the order of the tables.
SCORES = ["mae", "rmse", "r2", "spearman", "top10"]

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


# Step 5: 2 model families on 2 input sets. All rules were fixed BEFORE
# looking at any result.
# The 4 load features: how much and how recently the athlete raced.
LOAD_FEATURES = ["days_since_race", "race_days_21d",
                 "missed_prev_event", "first_start_major"]
# FULL = all 13 inputs. NO_LOAD = FULL without the 4 load features (9 inputs).
INPUT_SETS = {
    "FULL": FEATURES,
    "NO_LOAD": [name for name in FEATURES if name not in LOAD_FEATURES],
}

# The random forest: 300 trees, each split looks at a third of the inputs,
# a leaf may hold 1 single row, and the seed makes every run the same.
FOREST_TREES = 300
FOREST_MAX_FEATURES = 1 / 3
FOREST_MIN_LEAF = 1
FOREST_SEED = 42
# The depths we try. NO_LIMIT stands for "max_depth=None": the trees grow
# until the end. It is written as infinity, so that it sorts as the deepest.
NO_LIMIT = np.inf
DEPTH_GRID = [2, 4, 6, 8, 12, 16, NO_LIMIT]
# Completing the grid: around the best depth, the 2 depths halfway to its
# neighbours are added. Once, no second round.
EXTRA_DEPTHS = {2: [1, 3], 4: [3, 5], 6: [5, 7], 8: [7, 10],
                12: [10, 14], 16: [14, 20], NO_LIMIT: [20, 24]}
# Edge rule of step 5: if the final choice is ridge with the largest alpha
# of the grid, these 2 alphas are added for both input sets. Once.
EDGE_ALPHAS = [10000, 100000]

# Step 6, part A: the robustness run. Each validation season is predicted
# from the WINDOW_SEASONS seasons just before it only (a sliding window),
# and not from all earlier seasons as in step 5.
WINDOW_SEASONS = 5

# Step 6, part B: the final test. The final model was chosen in step 5 and
# is frozen: ridge, the NO_LOAD inputs (9), alpha 1000.
FINAL_INPUTS = "NO_LOAD"
FINAL_ALPHA = 1000
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
# The groups of inputs that are shuffled together.
INPUT_GROUPS = {
    "results": ["last3_finish_pct", "last_finish_pct", "last_place_any"],
    "skiing": ["last3_ski_pct"],
    "shooting": ["last3_range_pct", "last3_misses"],
    "experience and timing": ["n_prev_sprints", "days_since_sprint", "no_history"],
}
# The limits of the slices.
PCT_FRONT = 0.10        # real finish percentile up to here = the front of the race
PCT_BACK = 0.90         # from here on = the back of the race
DAYS_SHORT = 30         # days since the last sprint: up to 30, 31 to 180, above 180
DAYS_LONG = 180
STORY_NUMBER = 24       # the story race: number 24 of the 48 races sorted by MAE
# The 3 models that are scored on the test seasons, and their column of predictions.
TEST_MODELS = {"final": "pred_final", "constant": "pred_constant", "naive": "pred_naive"}

# The learning curve: the shares of the training rows, and the seed that
# picks the random subset.
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
# Each metric gets the rows of some races (with race_id and the true finish
# percentile) and 1 prediction per row, in the same order.

def mae(rows, pred):
    """Mean absolute error: the mean distance between prediction and truth.
    This is the primary metric, used for all choices."""
    return np.mean(np.abs(np.asarray(pred) - rows[TARGET].to_numpy()))


def rmse(rows, pred):
    """Root mean squared error: like the MAE, but big errors count more,
    because every error is squared before the mean is taken."""
    errors = np.asarray(pred) - rows[TARGET].to_numpy()
    return np.sqrt(np.mean(errors ** 2))


def r2(rows, pred):
    """R2: the share of the spread of the truth that the prediction explains.
    1 = perfect. 0 = as good as predicting the mean of the truth for all.
    Below 0 = worse than that."""
    true = rows[TARGET].to_numpy()
    left_over = np.sum((true - np.asarray(pred)) ** 2)     # squared errors of the model
    spread = np.sum((true - true.mean()) ** 2)             # squared errors of "always the mean"
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
    """The 5 metrics together, as a small dictionary."""
    return {"mae": mae(rows, pred),
            "rmse": rmse(rows, pred),
            "r2": r2(rows, pred),
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
# Step 4: 3 simple models, scored on the 5 walk forward folds.

def predict_constant(rows):
    """Model 1: the same value, 0.5, for everybody. It knows nothing."""
    return np.full(len(rows), MIDDLE)


def predict_naive(rows):
    """Model 2: the mean finish percentile of the last 3 sprints, as it is.
    Where this value is empty (no previous sprint), 0.5. Nothing is learned."""
    return rows["last3_finish_pct"].fillna(MIDDLE).to_numpy()


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


def score_line(model, alpha, fold, val_season, part, rows, pred):
    """1 line of the score table: who was scored, on which rows, and the 5 scores."""
    line = {"model": model, "alpha": alpha, "fold": fold,
            "val_season": val_season, "part": part, "n_rows": len(rows)}
    line.update(all_metrics(rows, pred))
    # The MAE again, separately for the rows without and with a previous sprint.
    pred = np.asarray(pred)
    new = (rows["no_history"] == 1).to_numpy()
    line["n_no_history"] = int(new.sum())
    line["mae_no_history"] = mae(rows[new], pred[new])
    line["mae_with_history"] = mae(rows[~new], pred[~new])
    return line


def score_models(alphas, with_simple=True):
    """Score the models on the training part and the validation part of each
    fold. Returns 1 long table with 1 line per model, alpha, fold and part.
    with_simple=False scores only the ridge models (used by the edge rule)."""
    df = load_features()            # the test seasons are already removed here
    lines = []
    fold = 0
    for val_season, train_part, val_part in make_folds(df):
        fold = fold + 1
        parts = [("train", train_part), ("validation", val_part)]

        if with_simple:
            for part, rows in parts:
                lines.append(score_line("constant", np.nan, fold, val_season,
                                        part, rows, predict_constant(rows)))
                lines.append(score_line("naive", np.nan, fold, val_season,
                                        part, rows, predict_naive(rows)))

        for alpha in alphas:
            model = make_ridge(alpha)
            # fit sees the training part of this fold only. The medians, the
            # scaling and the weights all come from these rows.
            model.fit(train_part[FEATURES], train_part[TARGET])
            for part, rows in parts:
                pred = model.predict(rows[FEATURES])
                lines.append(score_line("ridge", alpha, fold, val_season,
                                        part, rows, pred))
    return pd.DataFrame(lines)


def alpha_table(scores):
    """Mean MAE of the ridge model for each alpha, on train and on validation."""
    ridge = scores[scores["model"] == "ridge"]
    table = ridge.pivot_table(index="alpha", columns="part", values="mae", aggfunc="mean")
    table = table.rename(columns={"train": "train_mae", "validation": "val_mae"})
    table["val_mae_rounded"] = table["val_mae"].round(MAE_DECIMALS)
    return table.sort_index().reset_index()


def choose_alpha(table):
    """Apply the choice rule and the tie rule to an alpha table.
    Returns the chosen alpha and the alphas that share the best rounded MAE."""
    best = table["val_mae_rounded"].min()
    tied = table[table["val_mae_rounded"] == best]["alpha"].tolist()
    return max(tied), tied          # the tie rule: the largest of them


def find_alpha():
    """Score all models and choose the alpha, with the rules fixed in advance.
    Returns the score table, the alpha table and the chosen alpha."""
    scores = score_models(ALPHA_GRID)
    scores["round"] = "grid"                    # scored in the first round
    table = alpha_table(scores)
    alpha, tied = choose_alpha(table)
    first_choice = alpha
    edge_rule = "not used"

    # Edge rule: the chosen alpha sits at an end of the grid, so the best
    # value could lie outside. Add 2 steps of 10 on that side, choose again.
    if alpha == max(ALPHA_GRID) or alpha == min(ALPHA_GRID):
        if alpha == max(ALPHA_GRID):
            extra = [alpha * 10 ** step for step in range(1, EDGE_STEPS + 1)]
            edge_rule = "used at the largest alpha"
        else:
            extra = [alpha / 10 ** step for step in range(1, EDGE_STEPS + 1)]
            edge_rule = "used at the smallest alpha"
        more = score_models(extra, with_simple=False)
        more["round"] = "edge rule"
        scores = pd.concat([scores, more], ignore_index=True)
        table = alpha_table(scores)
        alpha, tied = choose_alpha(table)       # choose again, only this once

    # Write down how the choice was made, next to the numbers.
    table["first_grid"] = table["alpha"].isin(ALPHA_GRID)
    table["chosen"] = table["alpha"] == alpha
    table["shares_best_rounded_mae"] = table["alpha"].isin(tied)
    table["tie_rule"] = "used" if len(tied) > 1 else "not used"
    table["edge_rule"] = edge_rule
    table["first_choice"] = first_choice
    return scores, table, alpha


def summary_table(scores):
    """Mean and standard deviation over the 5 folds, for every model, alpha
    and part. The standard deviation is the usual one (divided by 5 - 1)."""
    keys = ["model", "alpha", "part"]
    numbers = SCORES + ["mae_no_history", "mae_with_history"]
    # dropna=False keeps the constant and the naive model, which have no alpha.
    table = scores.groupby(keys, dropna=False)[numbers].agg(["mean", "std"])
    table.columns = [score + "_" + what for score, what in table.columns]
    return table.reset_index()


def ladder_table(summary, alpha):
    """The 3 models next to each other: constant, naive, ridge at the chosen alpha."""
    keep = (summary["model"] != "ridge") | (summary["alpha"] == alpha)
    ladder = summary[keep].copy()
    # Put the lines in the order of the ladder: constant, naive, ridge.
    order = {"constant": 1, "naive": 2, "ridge": 3}
    ladder["order"] = ladder["model"].map(order)
    ladder = ladder.sort_values(["order", "part"]).drop(columns="order")
    columns = ["model", "alpha", "part"] + [s + "_" + w for s in SCORES for w in ["mean", "std"]]
    return ladder[columns].reset_index(drop=True)


def history_table(summary, alpha):
    """Validation MAE of the 3 models for the rows without and with history."""
    ladder = summary[(summary["model"] != "ridge") | (summary["alpha"] == alpha)]
    val = ladder[ladder["part"] == "validation"]
    order = {"constant": 1, "naive": 2, "ridge": 3}
    val = val.sort_values("model", key=lambda col: col.map(order))
    return val[["model", "alpha", "mae_mean", "mae_std",
                "mae_no_history_mean", "mae_no_history_std",
                "mae_with_history_mean", "mae_with_history_std"]].reset_index(drop=True)


def weights_table(alpha):
    """Fit the ridge model on ALL training seasons and list its weights.
    The inputs were scaled, so the weights can be compared with each other:
    a weight says how much the prediction moves when the feature goes up by
    1 standard deviation."""
    df = load_features()
    train = df[df["season"].isin(TRAIN_SEASONS)]
    model = make_ridge(alpha)
    model.fit(train[FEATURES], train[TARGET])
    table = pd.DataFrame({"feature": FEATURES,
                          "weight": model.named_steps["ridge"].coef_,
                          "fill_median": model.named_steps["fill"].statistics_})
    table["size"] = table["weight"].abs()
    table = table.sort_values("size", ascending=False).reset_index(drop=True)
    table["alpha"] = alpha
    table["intercept"] = model.named_steps["ridge"].intercept_
    return table


def load_output(name):
    """Load 1 of the tables that 'python pipeline.py baselines' saved in outputs/."""
    return pd.read_csv(os.path.join(OUTPUT_DIR, name + ".csv"), dtype={"val_season": str})


def run_baselines():
    """The whole step 4: score, choose the alpha, save the tables, print."""
    scores, alphas, alpha = find_alpha()
    summary = summary_table(scores)
    ladder = ladder_table(summary, alpha)
    history = history_table(summary, alpha)
    weights = weights_table(alpha)

    # Save every table as a file.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tables = {"baselines_scores_per_fold": scores,
              "baselines_summary": summary,
              "baselines_alpha": alphas,
              "baselines_ladder": ladder,
              "baselines_history": history,
              "baselines_weights": weights}
    for name, table in tables.items():
        table.to_csv(os.path.join(OUTPUT_DIR, name + ".csv"), index=False)

    print("--- alpha: mean MAE over the 5 folds ---")
    print(alphas[["alpha", "train_mae", "val_mae", "val_mae_rounded", "first_grid", "chosen"]]
          .round(5).to_string(index=False))
    print("chosen alpha:", alpha, "  tie rule:", alphas["tie_rule"].iloc[0],
          "  edge rule:", alphas["edge_rule"].iloc[0])

    print("\n--- the ladder: mean and standard deviation over the 5 folds ---")
    print(ladder.round(3).to_string(index=False))

    print("\n--- validation MAE, rows without and with history ---")
    print(history.round(3).to_string(index=False))

    print("\n--- ridge weights at the chosen alpha, fitted on all training seasons ---")
    print(weights[["feature", "weight"]].round(4).to_string(index=False))

    print("\nsaved in " + OUTPUT_DIR + "/:", ", ".join(name + ".csv" for name in tables))


# ------------------------------------------------------------------ models
# Step 5: ridge and random forest on 2 input sets, and the final choice.
# A "candidate" is 1 model family + 1 input set + 1 value of its setting
# (alpha for ridge, depth for the forest).

def make_forest(depth):
    """The random forest pipeline: median fill, then the forest. No scaler:
    a tree only asks "is the value above or below a limit", so the scale of
    an input does not matter. n_jobs=-1 only uses all processor cores, the
    result is the same."""
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


def make_model(family, value):
    """Build the pipeline of a candidate: ridge with this alpha, or forest with this depth."""
    if family == "ridge":
        return make_ridge(value)
    return make_forest(value)


def candidate_name(family, inputs, value):
    """A short name for printing, like 'ridge NO_LOAD alpha=10' or 'forest FULL depth=8'."""
    if family == "ridge":
        return "ridge " + inputs + " alpha=" + format(value, "g")
    if value == NO_LIMIT:
        return "forest " + inputs + " depth=None"
    return "forest " + inputs + " depth=" + str(int(value))


def score_candidates(candidates, folds_function=make_folds):
    """Score a list of candidates (family, inputs, value) on the training part
    and the validation part of each fold. 1 line per candidate, fold and part.
    folds_function builds the folds: the walk forward folds of step 3, or the
    sliding window folds of the robustness run."""
    df = load_features()            # the test seasons are already removed here
    lines = []
    fold = 0
    for val_season, train_part, val_part in folds_function(df):
        fold = fold + 1
        for family, inputs, value in candidates:
            columns = INPUT_SETS[inputs]
            model = make_model(family, value)
            # fit sees the training part of this fold only.
            model.fit(train_part[columns], train_part[TARGET])
            for part, rows in [("train", train_part), ("validation", val_part)]:
                pred = model.predict(rows[columns])
                line = {"family": family, "inputs": inputs, "value": value,
                        "name": candidate_name(family, inputs, value),
                        "fold": fold, "val_season": val_season,
                        "part": part, "n_rows": len(rows)}
                line.update(all_metrics(rows, pred))
                # The MAE again, for the rows without and with a previous sprint.
                new = (rows["no_history"] == 1).to_numpy()
                line["mae_no_history"] = mae(rows[new], pred[new])
                line["mae_with_history"] = mae(rows[~new], pred[~new])
                lines.append(line)
        print("  fold", fold, "of", len(VALIDATION_SEASONS), "done")
    return pd.DataFrame(lines)


def candidate_table(scores):
    """1 line per candidate: mean and standard deviation over the 5 folds of
    every score, on train and on validation. The standard deviation is the
    sample one (ddof = 1, divided by 5 - 1). SE = standard deviation / sqrt(5)."""
    keys = ["family", "inputs", "value", "name"]
    numbers = SCORES + ["mae_no_history", "mae_with_history"]
    parts = []
    for part, short in [("validation", "val"), ("train", "train")]:
        rows = scores[scores["part"] == part]
        table = rows.groupby(keys)[numbers].agg(["mean", "std"])     # std: ddof = 1
        table.columns = [short + "_" + score + "_" + what for score, what in table.columns]
        parts.append(table)
    table = pd.concat(parts, axis=1).reset_index()
    n_folds = scores["fold"].nunique()
    table["val_mae_se"] = table["val_mae_std"] / np.sqrt(n_folds)
    return table.sort_values("val_mae_mean").reset_index(drop=True)


def simplest_first(table):
    """Sort candidates from the simplest to the most complex:
    a) NO_LOAD before FULL, b) ridge before forest,
    c) ridge: the larger alpha first; forest: the smaller depth first."""
    table = table.copy()
    table["key_inputs"] = (table["inputs"] != "NO_LOAD").astype(int)     # NO_LOAD = 0, FULL = 1
    table["key_family"] = (table["family"] != "ridge").astype(int)       # ridge = 0, forest = 1
    # For ridge a LARGER alpha is simpler, so we turn the sign around.
    table["key_value"] = np.where(table["family"] == "ridge", -table["value"], table["value"])
    table = table.sort_values(["key_inputs", "key_family", "key_value"])
    return table.drop(columns=["key_inputs", "key_family", "key_value"])


def one_se_rule(table):
    """The one standard error rule on a table of candidates.
    Returns a small dictionary with the best candidate, the threshold, the
    names of the eligible candidates and the chosen (simplest eligible) one."""
    best = table.loc[table["val_mae_mean"].idxmin()]          # lowest mean validation MAE
    threshold = best["val_mae_mean"] + best["val_mae_se"]     # best mean + its SE
    eligible = table[table["val_mae_mean"] <= threshold]
    chosen = simplest_first(eligible).iloc[0]
    return {"best": best["name"], "best_mean": best["val_mae_mean"],
            "se": best["val_mae_se"], "threshold": threshold,
            "eligible": eligible["name"].tolist(), "chosen": chosen["name"]}


def fold_values(scores, name, column="mae"):
    """The 5 validation values of 1 candidate, 1 per fold, in fold order."""
    rows = scores[(scores["name"] == name) & (scores["part"] == "validation")]
    return rows.sort_values("fold")[column].to_numpy()


def compare(scores, table, name_a, name_b, what):
    """Put 2 candidates next to each other: the 2 mean validation MAEs, the
    difference (a minus b) in each fold, its mean and its standard error."""
    diff = fold_values(scores, name_a) - fold_values(scores, name_b)
    line = {"comparison": what, "a": name_a, "b": name_b,
            "a_mean": table.set_index("name").loc[name_a, "val_mae_mean"],
            "b_mean": table.set_index("name").loc[name_b, "val_mae_mean"],
            "a_se": table.set_index("name").loc[name_a, "val_mae_se"],
            "b_se": table.set_index("name").loc[name_b, "val_mae_se"],
            "diff_mean": diff.mean(),
            # SE of the difference: sample standard deviation of the 5 fold differences / sqrt(5).
            "diff_se": diff.std(ddof=1) / np.sqrt(len(diff))}
    for number, value in enumerate(diff):
        line["diff_fold_" + str(number + 1)] = value
    return line


def best_name(table, family=None, inputs=None):
    """The name of the candidate with the lowest mean validation MAE,
    inside 1 family and / or 1 input set if they are given."""
    if family is not None:
        table = table[table["family"] == family]
    if inputs is not None:
        table = table[table["inputs"] == inputs]
    return table.loc[table["val_mae_mean"].idxmin(), "name"]


def learning_curve(table, names):
    """Train the given candidates on 20, 40, 60, 80 and 100 % of the training
    rows of each fold and score them on the same validation part."""
    df = load_features()
    lines = []
    fold = 0
    for val_season, train_part, val_part in make_folds(df):
        fold = fold + 1
        for name in names:
            cand = table.set_index("name").loc[name]
            columns = INPUT_SETS[cand["inputs"]]
            for share in CURVE_SHARES:
                if share < 1:
                    # A random subset, always the same one because of the seed.
                    # sort_index puts the rows back in their original order.
                    subset = train_part.sample(frac=share, random_state=CURVE_SEED).sort_index()
                else:
                    subset = train_part
                model = make_model(cand["family"], cand["value"])
                model.fit(subset[columns], subset[TARGET])
                lines.append({"name": name, "fold": fold, "val_season": val_season,
                              "share": share, "n_train_rows": len(subset),
                              "train_mae": mae(subset, model.predict(subset[columns])),
                              "val_mae": mae(val_part, model.predict(val_part[columns]))})
    return pd.DataFrame(lines)


def importance_table(depth):
    """Fit the forest with the FULL inputs on all training seasons and list
    its feature_importances_ (they add up to 1), sorted."""
    df = load_features()
    train = df[df["season"].isin(TRAIN_SEASONS)]
    columns = INPUT_SETS["FULL"]
    model = make_forest(depth)
    model.fit(train[columns], train[TARGET])
    table = pd.DataFrame({"feature": columns,
                          "importance": model.named_steps["forest"].feature_importances_})
    table["depth"] = depth
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def run_models():
    """The whole step 5. Every rule is applied in the order it was written down."""
    # 1. The 2 families on the 2 input sets, with the grids fixed in advance.
    candidates = []
    for inputs in INPUT_SETS:
        for alpha in ALPHA_GRID:
            candidates.append(("ridge", inputs, alpha))
        for depth in DEPTH_GRID:
            candidates.append(("forest", inputs, depth))
    print("scoring", len(candidates), "candidates of the grids")
    scores = score_candidates(candidates)
    scores["round"] = "grid"

    # 2. Completing the forest grid: once, per input set, before any selection.
    table = candidate_table(scores)
    extra = []
    for inputs in INPUT_SETS:
        forest = table[(table["family"] == "forest") & (table["inputs"] == inputs)]
        best_depth = forest.loc[forest["val_mae_mean"].idxmin(), "value"]
        print("forest", inputs, ": best depth of the grid:", best_depth,
              " added:", EXTRA_DEPTHS[best_depth])
        for depth in EXTRA_DEPTHS[best_depth]:
            extra.append(("forest", inputs, depth))
    more = score_candidates(extra)
    more["round"] = "completed forest grid"
    scores = pd.concat([scores, more], ignore_index=True)

    # 3. Selection: the one standard error rule on ALL candidates together.
    table = candidate_table(scores)
    choice = one_se_rule(table)
    edge_rule = "not used"
    first_choice = choice["chosen"]

    # 4. Edge rule: chosen = ridge with the largest alpha of the grid.
    chosen = table.set_index("name").loc[choice["chosen"]]
    if chosen["family"] == "ridge" and chosen["value"] == max(ALPHA_GRID):
        edge_rule = "used"
        print("edge rule: adding the alphas", EDGE_ALPHAS, "for both input sets")
        extra = [("ridge", inputs, alpha) for inputs in INPUT_SETS for alpha in EDGE_ALPHAS]
        more = score_candidates(extra)
        more["round"] = "edge rule"
        scores = pd.concat([scores, more], ignore_index=True)
        table = candidate_table(scores)
        choice = one_se_rule(table)             # steps 1 to 4 again, only this once
    final = choice["chosen"]

    # Write down the result of the rule next to the numbers.
    table["eligible"] = table["name"].isin(choice["eligible"])
    table["lowest_mean"] = table["name"] == choice["best"]
    table["final_model"] = table["name"] == final
    selection = pd.DataFrame([{"best": choice["best"], "best_mean": choice["best_mean"],
                               "se": choice["se"], "threshold": choice["threshold"],
                               "n_candidates": len(table), "n_eligible": len(choice["eligible"]),
                               "final_model": final, "edge_rule": edge_rule,
                               "choice_before_edge_rule": first_choice}])

    # 5. Reports that do not change the choice.
    # Inside each family and input set: lowest mean, and the one SE pick of the group alone.
    lines = []
    for family in ["ridge", "forest"]:
        for inputs in INPUT_SETS:
            group = table[(table["family"] == family) & (table["inputs"] == inputs)]
            inside = one_se_rule(group)
            lines.append({"family": family, "inputs": inputs,
                          "lowest_mean": inside["best"], "lowest_mean_mae": inside["best_mean"],
                          "se": inside["se"], "threshold": inside["threshold"],
                          "one_se_pick": inside["chosen"],
                          "one_se_pick_mae": group.set_index("name").loc[inside["chosen"], "val_mae_mean"]})
    groups = pd.DataFrame(lines)

    # Load hypothesis (per family: best FULL against best NO_LOAD), and ridge against forest.
    best_ridge = best_name(table, family="ridge")
    best_forest = best_name(table, family="forest")
    comparisons = pd.DataFrame([
        compare(scores, table, best_name(table, "ridge", "FULL"),
                best_name(table, "ridge", "NO_LOAD"), "load, ridge: FULL minus NO_LOAD"),
        compare(scores, table, best_name(table, "forest", "FULL"),
                best_name(table, "forest", "NO_LOAD"), "load, forest: FULL minus NO_LOAD"),
        compare(scores, table, best_ridge, best_forest, "ridge minus forest"),
    ])

    # Rows without and with history, best ridge and best forest.
    history = table[table["name"].isin([best_ridge, best_forest])][
        ["name", "val_mae_mean", "val_mae_std",
         "val_mae_no_history_mean", "val_mae_no_history_std",
         "val_mae_with_history_mean", "val_mae_with_history_std"]]

    # 6. Looks that choose nothing.
    depth_curve = table[table["family"] == "forest"][
        ["inputs", "value", "name", "train_mae_mean", "val_mae_mean", "val_mae_se"]]
    depth_curve = depth_curve.sort_values(["inputs", "value"])

    # Learning curve: the final model and the best candidate of the other family.
    final_family = table.set_index("name").loc[final, "family"]
    other = best_forest if final_family == "ridge" else best_ridge
    print("learning curve:", final, "and", other)
    curve_folds = learning_curve(table, [final, other])
    curve = curve_folds.groupby(["name", "share"])[["n_train_rows", "train_mae", "val_mae"]].mean().reset_index()
    curve["role"] = np.where(curve["name"] == final, "final model", "best of the other family")

    # Forest importances: FULL inputs, the depth with the lowest mean.
    forest_full = table[(table["family"] == "forest") & (table["inputs"] == "FULL")]
    importance = importance_table(forest_full.loc[forest_full["val_mae_mean"].idxmin(), "value"])

    # 7. Save and print.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tables = {"models_scores_per_fold": scores, "models_candidates": table,
              "models_selection": selection, "models_groups": groups,
              "models_comparisons": comparisons, "models_history": history,
              "models_depth_curve": depth_curve, "models_learning_curve": curve,
              "models_learning_curve_per_fold": curve_folds,
              "models_importance": importance}
    for name, one in tables.items():
        one.to_csv(os.path.join(OUTPUT_DIR, name + ".csv"), index=False)

    print("\n--- all candidates, best mean validation MAE first ---")
    show = ["name", "train_mae_mean", "val_mae_mean", "val_mae_std", "val_mae_se",
            "val_rmse_mean", "val_r2_mean", "val_spearman_mean", "val_top10_mean",
            "eligible", "final_model"]
    print(table[show].round(4).to_string(index=False))

    print("\n--- selection: the one standard error rule on all", len(table), "candidates ---")
    print("lowest mean validation MAE:", choice["best"], " mean:", round(choice["best_mean"], 5))
    print("SE:", round(choice["se"], 5), "  threshold = mean + SE:", round(choice["threshold"], 5))
    print("eligible candidates:", len(choice["eligible"]))
    print("simplest eligible first:")
    print(simplest_first(table[table["eligible"]])[["name", "val_mae_mean"]].round(5).to_string(index=False))
    print("edge rule:", edge_rule)
    print("\n==============================================")
    print("THE FINAL MODEL:", final)
    print("==============================================")

    print("\n--- inside each family and input set ---")
    print(groups.round(5).to_string(index=False))
    print("\n--- comparisons: validation MAE, difference = a minus b ---")
    print(comparisons.round(5).T.to_string(header=False))
    print("\n--- validation MAE, rows without and with history ---")
    print(history.round(4).to_string(index=False))
    print("\n--- depth curve of the forest ---")
    print(depth_curve.round(4).to_string(index=False))
    print("\n--- learning curve: mean over the 5 folds ---")
    print(curve.round(4).to_string(index=False))
    print("\n--- forest importances (FULL inputs, depth", importance["depth"].iloc[0], "), all training seasons ---")
    print(importance[["feature", "importance"]].round(4).to_string(index=False))
    print("\nsaved in " + OUTPUT_DIR + "/:", ", ".join(name + ".csv" for name in tables))


# ------------------------------------------------------------------ robustness
# Step 6, part A. Training and validation seasons only. It changes nothing:
# it only tells us if the conclusions of step 5 hold with another split.

def run_robustness():
    """Score the 34 candidates of step 5 again on the sliding window folds
    and apply the one standard error rule with the same simplicity order."""
    # The same candidates as in the final table of step 5. No new round.
    step5 = load_output("models_candidates")
    candidates = [(line["family"], line["inputs"], line["value"])
                  for index, line in step5.iterrows()]
    print("scoring", len(candidates), "candidates on the sliding window folds")
    scores = score_candidates(candidates, folds_function=make_window_folds)

    table = candidate_table(scores)
    choice = one_se_rule(table)
    table["eligible"] = table["name"].isin(choice["eligible"])
    table["lowest_mean"] = table["name"] == choice["best"]
    table["one_se_choice"] = table["name"] == choice["chosen"]
    final_step5 = step5[step5["final_model"]]["name"].iloc[0]
    table["final_model_of_step_5"] = table["name"] == final_step5
    selection = pd.DataFrame([{"best": choice["best"], "best_mean": choice["best_mean"],
                               "se": choice["se"], "threshold": choice["threshold"],
                               "n_candidates": len(table), "n_eligible": len(choice["eligible"]),
                               "one_se_choice": choice["chosen"],
                               "final_model_of_step_5": final_step5}])

    comparisons = pd.DataFrame([
        compare(scores, table, best_name(table, family="ridge"),
                best_name(table, family="forest"), "ridge minus forest"),
        compare(scores, table, best_name(table, "ridge", "FULL"),
                best_name(table, "ridge", "NO_LOAD"), "load, ridge: FULL minus NO_LOAD"),
        compare(scores, table, best_name(table, "forest", "FULL"),
                best_name(table, "forest", "NO_LOAD"), "load, forest: FULL minus NO_LOAD"),
    ])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tables = {"robustness_scores_per_fold": scores, "robustness_candidates": table,
              "robustness_selection": selection, "robustness_comparisons": comparisons}
    for name, one in tables.items():
        one.to_csv(os.path.join(OUTPUT_DIR, name + ".csv"), index=False)

    print("\n--- sliding window: all candidates, best mean validation MAE first ---")
    show = ["name", "train_mae_mean", "val_mae_mean", "val_mae_std", "val_mae_se",
            "eligible", "one_se_choice", "final_model_of_step_5"]
    print(table[show].round(5).to_string(index=False))
    print("\n--- the one standard error rule, same simplicity order ---")
    print("best:", choice["best"], " mean:", round(choice["best_mean"], 5),
          " SE:", round(choice["se"], 5), " threshold:", round(choice["threshold"], 5))
    print("eligible:", len(choice["eligible"]), "of", len(table))
    print("the rule would choose:", choice["chosen"])
    print("final model of step 5:", final_step5)
    print("\n--- comparisons: validation MAE, difference = a minus b ---")
    print(comparisons.round(5).T.to_string(header=False))
    print("\nsaved in " + OUTPUT_DIR + "/:", ", ".join(name + ".csv" for name in tables))


# ------------------------------------------------------------------ final test
# Step 6, part B. The functions below get the training rows and the test rows
# as 2 tables. Only run_test() loads the real test rows.

def load_test_rows():
    """Load the rows of the TEST seasons from data/features.csv.
    Only the command 'python pipeline.py test' may call this function."""
    df = pd.read_csv(FEATURES_FILE, dtype={"season": str})
    return df[df["season"].isin(TEST_SEASONS)].reset_index(drop=True)


def load_test_predictions():
    """Load the per athlete predictions that the test command saved in data/."""
    return pd.read_csv(TEST_PREDICTIONS_FILE, dtype={"season": str})


def fit_final_model(train):
    """Fit the frozen final pipeline on the given training rows."""
    model = make_ridge(FINAL_ALPHA)
    model.fit(train[INPUT_SETS[FINAL_INPUTS]], train[TARGET])
    return model


def add_predictions(model, rows):
    """Add the predictions of the 3 models to the rows, and the places."""
    rows = rows.copy()
    rows["pred_final"] = model.predict(rows[INPUT_SETS[FINAL_INPUTS]])
    rows["pred_constant"] = predict_constant(rows)
    rows["pred_naive"] = predict_naive(rows)
    # The real place, found again from the percentile: pct = (place - 1) / (n - 1).
    rows["n_finishers"] = rows.groupby("race_id")[TARGET].transform("count")
    rows["real_place"] = (rows[TARGET] * (rows["n_finishers"] - 1)).round().astype(int) + 1
    # The predicted rank: the order of the predictions inside the race, 1 = lowest.
    rows["pred_rank"] = rows.groupby("race_id")["pred_final"].rank(method="min").astype(int)
    # The same prediction written as a place: place = pct * (n - 1) + 1.
    rows["pred_place"] = rows["pred_final"] * (rows["n_finishers"] - 1) + 1
    rows["error"] = rows["pred_final"] - rows[TARGET]          # prediction minus real
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


def test_race_table(rows):
    """1 line per race: the MAE of the 3 models. No athlete in this table."""
    lines = []
    for race_id, race in rows.groupby("race_id"):
        line = {"race_id": race_id, "season": race["season"].iloc[0],
                "day": race["start"].iloc[0][:10], "place": race["place"].iloc[0],
                "gender": race["gender"].iloc[0],
                "is_major_event": race["is_major_event"].iloc[0], "rows": len(race)}
        for model, column in TEST_MODELS.items():
            line["mae_" + model] = mae(race, race[column])
        lines.append(line)
    table = pd.DataFrame(lines)
    # Number the races from the lowest MAE of the final model to the highest.
    table = table.sort_values(["mae_final", "race_id"]).reset_index(drop=True)
    table["number_by_mae"] = table.index + 1
    table["story_race"] = table["number_by_mae"] == STORY_NUMBER
    return table.sort_values(["day", "race_id"]).reset_index(drop=True)


def test_slice_table(rows):
    """The MAE and the mean signed error (prediction minus real) of the
    3 models in each group of the 5 slices."""
    real = rows[TARGET]
    days = rows["days_since_sprint"]
    slices = [
        ("a) championship", "World Championships and Olympics", rows["is_major_event"] == True),
        ("a) championship", "other sprints", rows["is_major_event"] == False),
        ("b) history", "no_history = 1", rows["no_history"] == 1),
        ("b) history", "no_history = 0", rows["no_history"] == 0),
        ("c) gender", "men", rows["gender"] == "SM"),
        ("c) gender", "women", rows["gender"] == "SW"),
        ("d) real finish percentile", "up to 0.10", real <= PCT_FRONT),
        ("d) real finish percentile", "between", (real > PCT_FRONT) & (real < PCT_BACK)),
        ("d) real finish percentile", "0.90 and above", real >= PCT_BACK),
        ("e) days since the last sprint", "up to 30", days <= DAYS_SHORT),
        ("e) days since the last sprint", "31 to 180", (days > DAYS_SHORT) & (days <= DAYS_LONG)),
        ("e) days since the last sprint", "above 180", days > DAYS_LONG),
        ("e) days since the last sprint", "no history (apart)", days.isna()),
    ]
    lines = []
    for slice_name, group, keep in slices:
        part = rows[keep]
        for model, column in TEST_MODELS.items():
            lines.append({"slice": slice_name, "group": group, "model": model,
                          "rows": len(part), "races": part["race_id"].nunique(),
                          "mae": mae(part, part[column]) if len(part) > 0 else np.nan,
                          "mean_signed_error": (part[column] - part[TARGET]).mean()})
    return pd.DataFrame(lines)


def test_permutation_table(model, rows):
    """Permutation importance of the final model: shuffle 1 input (or 1 group
    of inputs) between the rows, and see how much the MAE goes up. A big
    increase means the model needs this input. 20 repeats, sample standard
    deviation (ddof = 1)."""
    columns = INPUT_SETS[FINAL_INPUTS]
    X = rows[columns]
    y = rows[TARGET]
    lines = []

    # a) Each input alone, with the function of scikit-learn. Its score is
    # "minus MAE" (higher = better), so its importance is the increase in MAE.
    result = permutation_importance(model, X, y, scoring="neg_mean_absolute_error",
                                    n_repeats=PERM_REPEATS, random_state=PERM_SEED)
    for number, name in enumerate(columns):
        increases = result.importances[number]             # the 20 increases of this input
        lines.append({"version": "a) each input alone", "name": name, "inputs": name,
                      "mae_increase_mean": increases.mean(),
                      "mae_increase_std": increases.std(ddof=1)})

    # b) By group: the columns of a group get the SAME new row order, so the
    # values of the group stay together and only their link to the athlete is cut.
    base = mae(rows, model.predict(X))
    for group, members in INPUT_GROUPS.items():
        rng = np.random.RandomState(PERM_SEED)             # the same seed for every group
        increases = []
        for repeat in range(PERM_REPEATS):
            order = rng.permutation(len(X))                # a random order of the rows
            shuffled = X.copy()
            shuffled[members] = X[members].to_numpy()[order]
            increases.append(mae(rows, model.predict(shuffled)) - base)
        increases = np.array(increases)
        lines.append({"version": "b) by group", "name": group, "inputs": ", ".join(members),
                      "mae_increase_mean": increases.mean(),
                      "mae_increase_std": increases.std(ddof=1)})

    table = pd.DataFrame(lines)
    table["mae_not_shuffled"] = base
    table["repeats"] = PERM_REPEATS
    return table.sort_values(["version", "mae_increase_mean"], ascending=[True, False]).reset_index(drop=True)


def final_weights_table(model):
    """The weights of the fitted final model, sorted by size, with the numbers
    that the fill step and the scale step learned."""
    table = pd.DataFrame({"feature": INPUT_SETS[FINAL_INPUTS],
                          "weight": model.named_steps["ridge"].coef_,
                          "fill_median": model.named_steps["fill"].statistics_,
                          "scale_mean": model.named_steps["scale"].mean_,
                          "scale_spread": model.named_steps["scale"].scale_})
    table["size"] = table["weight"].abs()
    table = table.sort_values("size", ascending=False).reset_index(drop=True)
    table["alpha"] = model.named_steps["ridge"].alpha
    table["intercept"] = model.named_steps["ridge"].intercept_
    return table


def draw_story(race, title, path=None):
    """The story figure: predicted rank against real rank in 1 race,
    1 dot per athlete, no names. If a path is given, the figure is saved."""
    import matplotlib.pyplot as plt
    n = len(race)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([1, n], [1, n], color="#52514e", linewidth=1, linestyle="--",
            label="perfect prediction")
    ax.scatter(race["real_place"], race["pred_rank"], s=28, color="#2a78d6",
               edgecolor="white", linewidth=0.6, label="1 athlete")
    ax.set_xlabel("real rank")
    ax.set_ylabel("predicted rank")
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
            + "\nMAE " + format(race_line["mae_final"], ".3f")
            + " (number " + str(STORY_NUMBER) + " of the races sorted by MAE)")


def test_tables(train, test):
    """Everything of part B for the given training rows and test rows:
    fit the final model on the training rows, predict the test rows, and
    build every table. Nothing is saved here."""
    model = fit_final_model(train)
    rows = add_predictions(model, test)
    return {"model": model,
            "predictions": rows,
            "scores": test_score_table(rows),
            "races": test_race_table(rows),
            "slices": test_slice_table(rows),
            "permutation": test_permutation_table(model, rows),
            "weights": final_weights_table(model)}


def load_test_data():
    """Load the training rows and the test rows of the final test, and check
    them. If a check fails the command stops here, before any score exists."""
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
    if not train["start"].max() < test["start"].min():
        problems.append("a training race starts after a test race")
    if sorted(test["season"].unique()) != TEST_SEASONS:
        problems.append("test seasons: " + str(sorted(test["season"].unique())))
    if len(problems) > 0:
        sys.exit("STOPPED before any test score was computed: " + "; ".join(problems))
    return train, test


def run_test():
    """Step 6, part B: the final test."""
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
    saved = {"test_scores": tables["scores"], "test_mae_per_race": races,
             "test_slices": tables["slices"], "test_permutation": tables["permutation"],
             "test_weights": tables["weights"]}
    for name, one in saved.items():
        one.to_csv(os.path.join(OUTPUT_DIR, name + ".csv"), index=False)
    # The per athlete predictions go to data/, which is not in git.
    keep = ["race_id", "season", "start", "place", "gender", "IBUId", "Name", "Nat",
            "is_major_event", TARGET, "no_history", "days_since_sprint", "n_finishers",
            "real_place", "pred_final", "pred_rank", "pred_place", "pred_constant",
            "pred_naive", "error", "abs_error"]
    rows[keep].to_csv(TEST_PREDICTIONS_FILE, index=False)
    # The story figure: the race with the middle MAE.
    story = races[races["story_race"]].iloc[0]
    draw_story(rows[rows["race_id"] == story["race_id"]], story_title(story),
               os.path.join(OUTPUT_DIR, STORY_FIGURE))

    model = tables["model"]
    print("final model  : ridge, alpha", model.named_steps["ridge"].alpha, ",",
          len(INPUT_SETS[FINAL_INPUTS]), "inputs:", ", ".join(INPUT_SETS[FINAL_INPUTS]))

    print("\n--- test scores: the 3 seasons together, then each season ---")
    print(tables["scores"].round(4).to_string(index=False))
    print("\n--- MAE per race (the final model): summary of the", len(races), "races ---")
    print(races["mae_final"].describe().round(4).to_string())
    print("championship sprints:", int(races["is_major_event"].sum()), "(expected 6)")
    print("\n--- slices: MAE and mean signed error (prediction minus real) ---")
    print(tables["slices"].round(4).to_string(index=False))
    print("\n--- permutation importance on the test rows: increase in MAE ---")
    print(tables["permutation"][["version", "name", "mae_increase_mean", "mae_increase_std"]]
          .round(5).to_string(index=False))
    print("\n--- weights of the refitted final model ---")
    print(tables["weights"][["feature", "weight"]].round(4).to_string(index=False))
    print("\nstory race:", story["race_id"], story["place"], story["day"],
          " MAE:", round(story["mae_final"], 4))
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
