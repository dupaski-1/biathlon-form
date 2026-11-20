"""Check for report section 9.2 (20 Nov 2026). NOT part of the pipeline.

Question: are the test seasons easier to predict because Russian and
Belarusian athletes no longer start (banned since 2022/23)?

Check: take these athletes (Nat RUS, RBU, BLR) out of every sprint, give the
remaining finishers new ranks 1 to n, rebuild the finish percentile and the
"last 3 sprints" feature with the functions of pipeline.py, and score the
naive rule season by season. If the ban explains the easier test seasons, the
older seasons should become about as easy as the test seasons (0.159 to 0.165).

Control: take out the same number of other athletes per season, picked at
random (20 different picks), to see what a smaller field alone does.

2 models are scored: the naive rule, which learns nothing, and the frozen
final model (lasso, alpha 0.01, 10 inputs), refitted walk forward on every
season before the one it predicts, as in validation and in the test. Its
settings are not changed, so no choice can be made here. The control is scored
with the naive rule only. The script writes nothing to the outputs folder.

Run:  python check_field_change.py     (about 2 minutes)
Result on 20 Nov: without RUS and BLR the naive rule goes from 0.1765 to
0.1734 on the 9 older seasons (test seasons: 0.1624), and the final model from
0.1608 to 0.1583 on the 4 older validation seasons (test seasons: 0.1495).
Both times the ban explains about a fifth of the gap. All 20 random picks
score between 0.1765 and 0.1796, so none is as low as 0.1734.
"""
import numpy as np
import pandas as pd
import pipeline as p

OUT = ["RUS", "RBU", "BLR"]      # RBU: the Russian team's name in 2020/21
SEASONS = p.TRAIN_SEASONS + p.TEST_SEASONS
CONTROL_PICKS = 20
CONTROL_SEED = 42


def rerank(table, rows):
    """New ranks 1 to n inside each race for the given rows (the finishers)."""
    table = table.copy()
    table.loc[rows, "Rank"] = table[rows].groupby("race_id")["Rank"].rank(method="min")
    return table


def build(sprints, races=None):
    """Re-rank the finishers and rebuild the target and the features with the
    functions of pipeline.py. Without 'races' only the sprint history is built
    (enough for the naive rule)."""
    sprints = rerank(sprints, sprints["IRM"].isna())
    fin = p.add_sprint_history(p.build_target(sprints))
    if races is not None:
        solo = ~races["discipline"].isin(p.RELAY_CODES) & races["Rank"].notna()
        fin = p.add_race_history(p.add_window_history(fin), rerank(races, solo))
    return fin


def naive_per_season(fin):
    scores = {}
    for season in SEASONS:
        rows = fin[fin["season"] == season]
        before = fin[fin["season"].isin(SEASONS) & (fin["season"] < season)]
        # As in the pipeline: empty cells get the median of the earlier seasons.
        fill = before["last3_finish_pct"].median() if len(before) > 0 else p.MIDDLE
        scores[season] = p.list_mae(rows, p.predict_naive(rows, fill))
    return pd.Series(scores)


def lasso_per_season(fin):
    """The frozen final model, walk forward, from the first validation season on."""
    columns = p.INPUT_SETS[p.FINAL_INPUTS]
    scores = {}
    for season in LASSO_SEASONS:
        rows = fin[fin["season"] == season]
        before = fin[fin["season"].isin(SEASONS) & (fin["season"] < season)]
        model = p.make_model(p.FINAL_FAMILY, p.FINAL_VALUE)
        model.fit(before[columns], before[p.TARGET])
        scores[season] = p.list_mae(rows, model.predict(rows[columns]))
    return pd.Series(scores)


LASSO_SEASONS = p.VALIDATION_SEASONS + p.TEST_SEASONS

sprints = p.load_sprints()
races = p.load_races()
banned = sprints["Nat"].isin(OUT)

fin_all = build(sprints, races)
fin_without = build(sprints[~banned], races[~races["Nat"].isin(OUT)])

table = pd.DataFrame({"naive_all": naive_per_season(fin_all),
                      "naive_without": naive_per_season(fin_without),
                      "lasso_all": lasso_per_season(fin_all),
                      "lasso_without": lasso_per_season(fin_without)})

# Control: the same number of athletes per season, but picked at random
# among the others. Naive rule only.
rng = np.random.RandomState(CONTROL_SEED)
picks = []
for pick in range(CONTROL_PICKS):
    drop = pd.Series(False, index=sprints.index)
    for season in SEASONS:
        in_season = sprints["season"] == season
        n_out = sprints[in_season & banned]["IBUId"].nunique()
        others = sprints[in_season & ~banned]["IBUId"].unique()
        chosen = rng.choice(others, size=n_out, replace=False)
        drop |= in_season & sprints["IBUId"].isin(chosen)
    picks.append(naive_per_season(build(sprints[~drop])))
picks = pd.concat(picks, axis=1)
table["naive_random_out_mean"] = picks.mean(axis=1)

table["rus_blr_athletes"] = [sprints[(sprints["season"] == s) & banned]["IBUId"].nunique()
                             for s in SEASONS]
print(table.round(4).to_string())

old = [s for s in SEASONS if s < "2223"]              # RUS and BLR still start
old_lasso = [s for s in LASSO_SEASONS if s < "2223"]
test = p.TEST_SEASONS
print("\nnaive rule, mean of the 9 seasons with RUS and BLR (2013/14 to 2021/22):")
print("  all athletes      :", round(table.loc[old, "naive_all"].mean(), 4))
print("  without RUS, BLR  :", round(table.loc[old, "naive_without"].mean(), 4))
print("  random others out :", round(table.loc[old, "naive_random_out_mean"].mean(), 4),
      "  lowest and highest of the", CONTROL_PICKS, "picks:",
      round(picks.loc[old].mean().min(), 4), "and", round(picks.loc[old].mean().max(), 4))
print("  the 3 test seasons:", round(table.loc[test, "naive_all"].mean(), 4))
print("final model, mean of the 4 validation seasons with RUS and BLR (2018/19 to 2021/22):")
print("  all athletes      :", round(table.loc[old_lasso, "lasso_all"].mean(), 4))
print("  without RUS, BLR  :", round(table.loc[old_lasso, "lasso_without"].mean(), 4))
print("  the 3 test seasons:", round(table.loc[test, "lasso_all"].mean(), 4))
print("controls: final model, 5 validation seasons, all athletes:",
      round(table.loc[p.VALIDATION_SEASONS, "lasso_all"].mean(), 4), "(step 5: 0.1597)")
