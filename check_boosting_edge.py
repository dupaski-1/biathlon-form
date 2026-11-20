"""Check of the boosting grid edge (19 Nov 2026). NOT part of the pipeline.

The best boosting candidate (learning rate 0.02, depth 3) sits at the lowest
learning rate and the lowest depth of its grid. This script scores smaller
values on the same 5 validation folds, to see if anything beats it.
It reads data/features.csv through pipeline.py (test seasons stay closed)
and writes nothing to the outputs folder.

Run:  python check_boosting_edge.py     (about 2 minutes)
Result on 19 Nov: learning rate 0.02 with depth 3 stays the best (0.15856).
"""
import pipeline as p

RATES = [0.005, 0.01, 0.02]     # 0.02 with depth 3 is the control: it must give 0.158558
DEPTHS = [1, 2, 3]

candidates = [("boosting", inputs, depth, rate)
              for inputs in p.INPUT_SETS for rate in RATES for depth in DEPTHS]
scores = p.score_candidates(candidates)
validation = scores[scores["part"] == "validation"]
table = validation.groupby("name").agg(list_mae_mean=("list_mae", "mean"),
                                       list_mae_std=("list_mae", "std"),
                                       trees_used_mean=("trees_used", "mean"))
print(table.sort_values("list_mae_mean").to_string())
