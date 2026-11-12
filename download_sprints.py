# download_sprints.py
# Downloads every level 1 sprint (World Cup, World Championships, Olympics)
# from season 2012/13 to 2025/26, with the split times of each athlete, and
# saves them in data/sprints_raw.csv, with 1 row per athlete per sprint.
#
# Run it with:  conda activate biathlon  then  python download_sprints.py
#
# This is the code that was in 01_explore.ipynb, moved here without changes
# to the logic. The file data/sprints_raw.csv of the project was made by that
# notebook code on 6 October 2026.
#
# CAREFUL: running this script writes data/sprints_raw.csv again. The API can
# give slightly different data today (for example an athlete who changed her
# name), so keep a copy of the old file before running it.

import os
import time

import pandas as pd
import biathlonresults as br

# ---------------------------------------------------------------- settings

SEASONS = ["1213", "1314", "1415", "1516", "1617", "1718", "1819",
           "1920", "2021", "2122", "2223", "2324", "2425", "2526"]

LEVEL = "1"            # 1 = World Cup, World Championships, Olympics (no IBU Cup)
PAUSE = 0.2            # seconds to wait between 2 races, to be polite with the server

OUT_FILE = "data/sprints_raw.csv"    # the file this script creates

# The split times we ask for. Each code is 1 separate call to the API.
# CRST = ski time, RNGT = range time, STTM = time standing on the shooting mat,
# CRS1 to CRS3 = ski time of lap 1, 2, 3, RNG1 and RNG2 = range time of shooting 1, 2.
SPLIT_CODES = ["CRST", "RNGT", "STTM", "CRS1", "CRS2", "CRS3", "RNG1", "RNG2"]


# ---------------------------------------------------------------- helpers

def list_sprints():
    """Build the list of all sprints by going through seasons -> events -> competitions."""
    rows = []
    for s in SEASONS:
        for ev in br.events(s, LEVEL):
            for c in br.competitions(ev["EventId"]):
                if c["DisciplineId"] == "SP":        # SP = sprint
                    rows.append({
                        "season": s,
                        "place": ev["ShortDescription"],
                        "race_id": c["RaceId"],
                        "gender": c["catId"],        # SM = men, SW = women
                        "start": c["StartTime"],
                        "status": c["StatusText"],
                    })
    return pd.DataFrame(rows)


def get_race(race_id):
    """Download 1 sprint and attach the split times to each athlete."""
    # The normal result list: 1 row per athlete.
    res = pd.DataFrame(br.results(race_id)["Results"])
    res = res[["IBUId", "Name", "Nat", "Bib", "Rank", "IRM",
               "Shootings", "ShootingTotal", "TotalTime", "Behind"]]
    # Each split time comes as its own list. In that list the column
    # "TotalTime" holds the split time, so we rename it to the code.
    for code in SPLIT_CODES:
        ana = pd.DataFrame(br.analytic_results(race_id, code)["Results"])
        if len(ana) > 0:
            ana = ana[["IBUId", "TotalTime"]].drop_duplicates("IBUId")
            # how="left" keeps every athlete, also those without this split time.
            res = res.merge(ana.rename(columns={"TotalTime": code}),
                            on="IBUId", how="left")
        else:
            res[code] = None                 # the API has nothing for this code
    res["race_id"] = race_id
    return res


# ---------------------------------------------------------------- download

races = list_sprints()

# Keep only the races that took place (a cancelled race has no results).
races = races[races["status"] == "Final"].reset_index(drop=True)
print(len(races), "sprints to get")

frames = []        # the tables of the races that worked
failed = []        # the IDs of the races that did not work
for i, rid in enumerate(races["race_id"]):
    try:
        frames.append(get_race(rid))
    except Exception as e:
        failed.append(rid)
        print("failed:", rid, e)
    if i % 25 == 0:
        print(i, "of", len(races))
    time.sleep(PAUSE)

# Put all races under each other, then add the information about each race
# (season, place, gender, start, status) to every row.
raw = pd.concat(frames, ignore_index=True)
raw = raw.merge(races, on="race_id", how="left")

print(raw.shape, "failed:", len(failed))

# ---------------------------------------------------------------- save

os.makedirs("data", exist_ok=True)
raw.to_csv(OUT_FILE, index=False)
