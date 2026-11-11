# pipeline.py
# The data pipeline of the project. The steps so far:
#
#   python pipeline.py features    build data/features.csv
#   python pipeline.py split       print the season split and the validation folds
#
# About "features":
# It reads data/sprints_raw.csv and data/races_all.csv, cleans them, builds
# the target and the 13 features, and writes data/features.csv with
# 1 row per finisher per sprint.
#
# Golden rule: a feature of a race only uses races that started BEFORE it.

import sys

import numpy as np
import pandas as pd

SPRINT_FILE = "data/sprints_raw.csv"    # sprints with split times (from 01_explore)
RACES_FILE = "data/races_all.csv"       # all formats, results only (from download_races.py)
FEATURES_FILE = "data/features.csv"     # what this step creates

RELAY_CODES = ["RL", "SR"]              # relay and single mixed relay

TARGET = "finish_pct"

# Caps: values above the cap are set to the cap.
# A few athletes come back after years away (up to about 3200 days). Such a
# huge number would weigh too much in the model. After 1 year without racing,
# a longer break does not tell us more, so we stop counting at 365 days.
MAX_DAYS_SINCE = 365
# The number of previous sprints goes up to 121. After 20 sprints an athlete
# is simply "experienced", so we stop counting at 20. A second reason: the
# data starts in 2012, so high counts are not complete anyway.
MAX_PREV_SPRINTS = 20

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

# The 13 features, in the order of the project brief.
FEATURES = [
    "last3_finish_pct",    # 1  mean finish percentile, last 3 sprints
    "last3_ski_pct",       # 2  mean ski percentile, last 3 sprints
    "last3_range_pct",     # 3  mean range percentile, last 3 sprints
    "last3_misses",        # 4  mean number of misses, last 3 sprints
    "last_finish_pct",     # 5  finish percentile of the last sprint
    "n_prev_sprints",      # 6  number of previous sprints
    "last_place_any",      # 7  finish place in the last race of any format, no relays
    "days_since_sprint",   # 8  days since the last sprint
    "days_since_race",     # 9  days since the last race day, relays included
    "race_days_21d",       # 10 number of race days in the last 21 days, relays included
    "missed_prev_event",   # 11 flag: did not start at the previous event of the season
    "first_start_major",   # 12 flag: first start at this World Championships / Olympics
    "no_history",          # 13 flag: no previous sprint
]


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

    # Same idea with the skiing time (CRST) and the time on the range (RNGT).
    fin["ski_pct"] = percentile(fin["CRST_s"], fin["race_id"])
    fin["range_pct"] = percentile(fin["RNGT_s"], fin["race_id"])

    fin["misses"] = fin["ShootingTotal"]                   # number of missed targets
    return fin


# ------------------------------------------------------------------ features

def add_sprint_history(fin):
    """Features 1, 2, 3, 4, 5, 6, 8 and 13: they only need the earlier sprints
    where the athlete had a valid finish. 'fin' holds exactly those sprints."""
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
    fin["last3_misses"] = by_athlete["misses"].transform(mean_last3)

    # Feature 5: the result of the sprint just before.
    fin["last_finish_pct"] = by_athlete[TARGET].shift(1)

    # Feature 6: cumcount() numbers the lines of an athlete 0, 1, 2, ...
    # which is the number of sprints before this one.
    fin["n_prev_sprints"] = by_athlete.cumcount()

    # Feature 8: days between this sprint and the sprint before.
    # clip(upper=...) sets every value above the cap to the cap. Empty stays empty.
    fin["days_since_sprint"] = (fin["date"] - by_athlete["date"].shift(1)).dt.days
    fin["days_since_sprint"] = fin["days_since_sprint"].clip(upper=MAX_DAYS_SINCE)

    # Feature 13: 1 when there is no earlier sprint.
    fin["no_history"] = (fin["n_prev_sprints"] == 0).astype(int)

    # Cap feature 6 last, after it was used for the flag above.
    fin["n_prev_sprints"] = fin["n_prev_sprints"].clip(upper=MAX_PREV_SPRINTS)
    return fin


def add_race_history(fin, races):
    """Features 7, 9, 10, 11 and 12: they need the races of all formats."""
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

    # Feature 9: days since the most recent race day.
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

    fin["days_since_race"] = (fin["date"] - fin["last_race_date"]).dt.days
    fin["days_since_race"] = fin["days_since_race"].clip(upper=MAX_DAYS_SINCE)
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
    """Print the checks of step 2."""
    print("rows:", len(out), "(expected 24957)   races:", out["race_id"].nunique(), "(expected 252)")

    print("\n--- share of missing values, min, mean, max ---")
    table = pd.DataFrame({"missing": out[[TARGET] + FEATURES].isna().mean(),
                          "min": out[[TARGET] + FEATURES].min(),
                          "mean": out[[TARGET] + FEATURES].mean(),
                          "max": out[[TARGET] + FEATURES].max()})
    print(table.round(3).to_string())

    # Leak test on 1 athlete: raw results next to the features.
    name = "BOE Johannes Thingnes"
    one = fin[fin["Name"] == name].sort_values("start")
    one = one.assign(day=one["date"].dt.date)       # the date in a short form for printing
    print("\n--- leak test:", name, "---")
    print("\nHis last 9 valid sprints, raw results (the features of the last 6 are below):")
    raw_cols = ["day", "place", "Rank", TARGET, "ski_pct", "range_pct", "misses"]
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
        # Mix the rows of the race in a random order (always the same one,
        # because of random_state). The file is sorted by real rank, so
        # without this, equal predictions would be picked in the real order
        # and a model that predicts the same value for all would look perfect.
        race = race.sample(frac=1, random_state=0)
        # The real place in the race, found again from the true percentile.
        place = race["true"].rank(method="min")
        picked = race["pred"].nsmallest(TOP_N).index      # our predicted top 10
        values.append((place[picked] <= TOP_N).mean())    # share of real top 10
    return np.mean(values)


def all_metrics(rows, pred):
    """The 3 metrics together, as a small dictionary."""
    return {"mae": mae(rows, pred),
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


# ------------------------------------------------------------------ main step

def make_features():
    """The whole step 2: load, clean, target, features, save, checks."""
    sprints = load_sprints()
    races = load_races()

    fin = build_target(sprints)
    fin = add_sprint_history(fin)
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
    else:
        print("usage: python pipeline.py features   or   python pipeline.py split")
