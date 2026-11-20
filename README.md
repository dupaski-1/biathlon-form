# Predicting Biathlon World Cup Sprint Results

Semester project for MST0052 Predictive Modelling with Machine Learning at BI Norwegian Business School, autumn 2026.

Author: Arnaud Du Pasquier

## What this project does

Before a biathlon World Cup sprint, the start list is known. This project predicts the result list of that sprint from each starter's earlier races.

The model gives every starter a predicted finish percentile (0 is the winner, 1 is the last finisher). Sorting these numbers inside the race gives a predicted result list.

The full report is in `Biathlon_report.pdf`.

## Main result

The final model is a lasso regression that keeps 6 inputs. It was tested on 3 seasons it was not trained on (2023/24 to 2025/26, 48 sprints, 4,778 results).

| Model | Validation list MAE | Test list MAE |
|---|---|---|
| Random order (knows nothing) | 0.337 | 0.339 |
| Naive rule (average of the last 3 sprints) | 0.175 | 0.162 |
| Final model (lasso, 6 inputs) | 0.160 | 0.150 |

A list MAE of 0.150 means the predicted place is on average about 15 places off per 100 starters.

4 model families (ridge, lasso, random forest, gradient boosting) score within 0.0004 of each other. The limit is what the features know, not the type of model.

## What is in this repository

| File or folder | Content |
|---|---|
| `Biathlon_report.pdf` | The report |
| `pipeline.py` | The whole pipeline, from the raw files to every table and score |
| `download_sprints.py`, `download_races.py` | Download the 2 raw files from biathlonresults.com |
| `check_boosting_edge.py`, `check_field_change.py` | 2 extra checks outside the pipeline (report sections 5.6 and 9.2) |
| `figures.py` | Makes the figures of the report from the tables in `outputs/` |
| `lab_book.ipynb` | Notebook with every data check in a cell of its own |
| `outputs/` | The score tables behind the numbers in the report |
| `environment.yml` | The conda environment with pinned versions |

## How to reproduce

```
conda env create -f environment.yml
conda activate biathlon
python download_sprints.py      # sprint file
python download_races.py        # all races file
python pipeline.py features     # target and features
python pipeline.py split        # split and folds
python pipeline.py baselines    # constant, naive rule, ridge, lasso
python pipeline.py models       # 4 families, with and without load; selection
python pipeline.py robustness   # second split
python pipeline.py test         # the final test, walk forward
python check_boosting_edge.py   # extra check: boosting grid edge
python check_field_change.py    # extra check: field change
```

The `test` command only runs when `TEST_UNLOCKED = True` is set in `pipeline.py`. It was set by hand before the final test.

Seeds are fixed (42 everywhere), so the same data gives the same numbers. The download scripts read the live results site, which can change slightly over time.

## Data

All data comes from the official results of the International Biathlon Union, read from biathlonresults.com through the Python package `biathlonresults` (version 0.1.1).

The IBU terms of use do not allow publishing the data, so the `data/` folder is not in this repository. The 2 download scripts rebuild it from the public source.

## Version of the report

The report describes the code at commit `6d0c450`. Later commits only add this README, the final report PDF and `figures.py`, the script that draws the figures. The pipeline and the outputs are unchanged.

## AI use

The course allows AI use with disclosure. I used Claude in chat, Claude Code and the course AI tutor. The report (section 11) states what was decided by me, what was built with AI and how it was checked.
