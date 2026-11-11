# download_races.py
# Downloads the results of every level 1 biathlon race (World Cup, World
# Championships, Olympics) from season 2012/13 to 2025/26 and saves them in
# data/races_all.csv, with 1 row per athlete per start.
#
# Run it with:  conda activate biathlon  then  python download_races.py
# If it stops in the middle, run it again: it continues where it stopped.

import os
import time

import pandas as pd
import biathlonresults as br

# ---------------------------------------------------------------- settings

SEASONS = ["1213", "1314", "1415", "1516", "1617", "1718", "1819",
           "1920", "2021", "2122", "2223", "2324", "2425", "2526"]

LEVEL = "1"            # 1 = World Cup, World Championships, Olympics (no IBU Cup)
PAUSE = 0.3            # seconds to wait between 2 calls, to be polite with the server
MAX_TRIES = 4          # how many times we try a call before giving up

PARTS_DIR = "data/races_parts"       # 1 small file per race = our saved progress
OUT_FILE = "data/races_all.csv"      # the final file
SPRINT_FILE = "data/sprints_raw.csv" # the old sprint file, only read, never written

# The API gives short codes for the formats. This table is only for printing.
FORMAT_NAMES = {"SP": "sprint", "PU": "pursuit", "IN": "individual",
                "SI": "short individual", "MS": "mass start",
                "RL": "relay", "SR": "single mixed relay"}

RELAY_CODES = ["RL", "SR"]           # the 2 formats that are raced in teams

# Events with one of these codes are World Championships (CH) or Olympics (OG).
MAJOR_CODES = ["BTSWRLCH", "BTSWRLOG"]

# The columns of the final file, in order.
COLUMNS = ["IBUId", "Name", "Nat", "race_id", "event_id", "season", "place",
           "start", "discipline", "category", "is_major_event",
           "Rank", "IRM", "ShootingTotal", "TotalTime", "Behind",
           "leg", "team_rank", "LegRank", "TeamRankAfterLeg"]


# ---------------------------------------------------------------- helpers

def call(function, *args):
    """Call the API, wait a little, and try again if the call fails."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            answer = function(*args)
            time.sleep(PAUSE)
            return answer
        except Exception as error:
            print("  call failed (try", attempt, "of", MAX_TRIES, "):", error)
            time.sleep(5 * attempt)      # wait longer after each failure
    # All tries failed: stop the script. A rerun continues from here.
    raise RuntimeError("the API did not answer, run the script again later")


def list_races():
    """Build the list of all races by going through seasons -> events -> competitions."""
    rows = []
    for season in SEASONS:
        for event in call(br.events, season, LEVEL):
            for comp in call(br.competitions, event["EventId"]):
                rows.append({
                    "race_id": comp["RaceId"],
                    "event_id": event["EventId"],
                    "season": season,
                    "place": event["ShortDescription"],
                    "start": comp["StartTime"],
                    "discipline": comp["DisciplineId"],
                    "category": comp["catId"],
                    "is_major_event": event["EventClassificationId"] in MAJOR_CODES,
                    "status": comp["StatusText"],
                })
        print("listed season", season)
    return pd.DataFrame(rows)


def get_race(race):
    """Download the results of 1 race and return a table with our columns."""
    res = pd.DataFrame(call(br.results, race["race_id"])["Results"])

    # A race without any result row: return an empty table with the right columns.
    if len(res) == 0:
        return pd.DataFrame(columns=COLUMNS)

    if race["discipline"] in RELAY_CODES:
        # A relay has 1 row per team (IsTeam is True) and 1 row per leg.
        # We keep only the leg rows, because those are the athletes.
        res = res[res["IsTeam"] == False].copy()
        res["leg"] = res["Leg"]            # leg number 1, 2, 3 or 4
        res["team_rank"] = res["Rank"]     # on a leg row, Rank is the rank of the team
    else:
        # In the other formats these 4 columns have no meaning: leave them empty.
        res["leg"] = None
        res["team_rank"] = None
        res["LegRank"] = None
        res["TeamRankAfterLeg"] = None

    # Add the information about the race itself to every row.
    for col in ["race_id", "event_id", "season", "place", "start",
                "discipline", "category", "is_major_event"]:
        res[col] = race[col]

    return res[COLUMNS]


# ---------------------------------------------------------------- download

os.makedirs(PARTS_DIR, exist_ok=True)

races = list_races()

# Split the list: races that took place, and cancelled races (nothing to download).
cancelled = races[races["status"] != "Final"]
races = races[races["status"] == "Final"].reset_index(drop=True)
print(len(races), "races to get,", len(cancelled), "cancelled")

for i, race in races.iterrows():
    part_file = os.path.join(PARTS_DIR, race["race_id"] + ".csv")

    # Saved progress: if the file of this race exists, it was downloaded before.
    if os.path.exists(part_file):
        continue

    table = get_race(race)

    # Write to a temporary name first and rename after. Like this a file with
    # the final name is always complete, even if the script is stopped here.
    table.to_csv(part_file + ".tmp", index=False)
    os.replace(part_file + ".tmp", part_file)

    if i % 25 == 0:
        print(i, "of", len(races))

# ---------------------------------------------------------------- put together

# Read every small file as text (dtype=str) so that nothing is changed,
# for example the season "1213" stays "1213" and does not become a number.
parts = []
empty = []
for race_id in races["race_id"]:
    part = pd.read_csv(os.path.join(PARTS_DIR, race_id + ".csv"), dtype=str)
    if len(part) == 0:
        empty.append(race_id)
    else:
        parts.append(part)

all_rows = pd.concat(parts, ignore_index=True)
all_rows.to_csv(OUT_FILE, index=False)

# ---------------------------------------------------------------- checks

# Check 1: number of races per season and per format.
print("\n--- races per season and format (races with results) ---")
done = all_rows.drop_duplicates("race_id")      # 1 line per race
done = done.assign(format=done["discipline"].map(FORMAT_NAMES))
print(pd.crosstab(done["season"], done["format"], margins=True).to_string())

# Check 2: cancelled races and races without rows.
print("\n--- cancelled races:", len(cancelled), "---")
print(cancelled[["race_id", "place", "start", "status"]].to_string(index=False))
print("\n--- races marked Final but without result rows:", len(empty), "---")
print(races[races["race_id"].isin(empty)][["race_id", "place", "start"]].to_string(index=False))

# Check 3: the sprint rows must match the old sprint file.
print("\n--- sprint check against", SPRINT_FILE, "---")
old = pd.read_csv(SPRINT_FILE, dtype=str)
old_counts = old.groupby("race_id").size()      # athletes per race in the old file
new_counts = all_rows[all_rows["discipline"] == "SP"].groupby("race_id").size()
same_ids = set(old_counts.index) == set(new_counts.index)
print("sprint races old:", len(old_counts), " new:", len(new_counts), " same race IDs:", same_ids)
# Put the 2 counts next to each other and keep the races where they differ.
both = pd.concat([old_counts, new_counts], axis=1, keys=["old", "new"])
different = both[both["old"] != both["new"]]
print("races with a different number of athletes:", len(different))
if len(different) > 0:
    print(different)

# Check 4: total rows and file size.
print("\n--- total ---")
print("rows:", len(all_rows), " columns:", all_rows.shape[1], " races:", all_rows["race_id"].nunique())
print("file size:", round(os.path.getsize(OUT_FILE) / 1e6, 2), "MB")
