# Predicting biathlon sprint results from recent form

MST0052 semester project report · Nov 18, 2026 · @Arnaud

## Summary

I predict the result list of a biathlon World Cup sprint from each starter's earlier races. The final model is a lasso regression that keeps 6 inputs. On 3 seasons it was not trained on (48 sprints, 4,778 results), its list MAE is 0.150: the predicted place is on average about 15 places off per 100 starters.

| Model | Validation list MAE | Test list MAE |
| --- | --- | --- |
| Random order (knows nothing) | 0.337 | 0.339 |
| Simple average (sort by the last 3 sprints) | 0.175 | 0.162 |
| Final model (lasso, 6 inputs) | 0.160 | 0.150 |

4 findings matter more than the headline number:

1. **An athlete's recent results already say most of what can be said.** Sorting by the average of the last 3 sprints is about 16 places off per 100 starters, against 15 for the final model, which wins in every test season and in 43 of 48 races. The gain is largest for newcomers, whom the simple average cannot place.
2. **The limit is what the features know, not the type of model.** 4 families, from ridge to gradient boosting, score within 0.04 places of each other. Skiing form repeats from race to race; whether the shots hit in 1 sprint says little about the next.
3. **But not just any new information helps.** I added 3 features for World Cup load, meaning how much someone has raced in recent weeks, expecting long stretches on the circuit to wear athletes down. They did not improve the prediction in any measurable way.
4. **Sprints are harder to predict for women than for men.** On the test seasons the error is about 1 place per 100 starters larger for them, and the simple average shows the same gap. The cause was not investigated.

**Main weakness.** The model cannot see who will have an exceptional day: it ranks the favourites well, but places the real winners a little too far back and the weakest athletes too far forward. It gives expected places, not chances of winning.

**How to read the tags.** \[Expert\] marks a choice based on my experience as a biathlete. \[Outside\] marks a method that is not part of the course.

## 1. Problem and use

**The question.** Before a biathlon World Cup sprint, the start list is known. From each starter's recent races, can we predict the result list of that sprint?

**What the model does.** It gives every starter a predicted finish percentile, where 0 is the winner and 1 is the last finisher. Sorting these numbers inside the race gives a predicted result list, from place 1 to place n. That list is the output a user would see.

**Who would use it.** Media and fans who want to judge form before a race, and team staff who select athletes. Betting is a useful thought experiment: a bookmaker needs the chance that a given athlete wins or reaches the podium, which a list of expected places cannot give. That would need a classifier with probabilities, and I name it as a next step.

**Why sprints only.** A sprint has an interval start and a full field, so its result does not depend on an earlier race the way a pursuit does. Individual races are too few and a different game, with more shooting and a heavier penalty per miss. The other formats are not predicted, but their results feed the features. \[Expert\]

**Why a percentile.** Time behind the winner depends on the course, the snow and the format. A percentile can be compared across races and field sizes. \[Expert\]

**How the model is judged.** All choices are made with 1 number, the list MAE. In each race the predicted result list is turned back into percentiles and compared with the real ones. A list MAE of 0.15 means the predicted place is on average about 15 places off per 100 starters. I score the list, not the raw predictions, because the list is what a user would act on. Section 4 explains the metric in full.

## 2. Data

**Source.** All data comes from the official results of the International Biathlon Union, read from biathlonresults.com through the Python package `biathlonresults` (version 0.1.1, pinned). It is the only structured source with split times. The package is a small hobby project, so its outputs were checked before use.

**2 raw files, 14 seasons (2012/13 to 2025/26).**

| File | Rows | Races | Content | Feeds |
| --- | --- | --- | --- | --- |
| Sprint file | 25,507 | 252 sprints | Results plus ski time and range time | The target; ski and range features |
| All races file | 73,348 | 935 races | Results only, every World Cup format, relays included | Load features; last place in any format |

The sprint rows are identical in both files. 16 cancelled races are skipped.

**Why start in 2012/13.** Range time only exists from that season on.

**Championships.** World Championship and Olympic sprints are kept. They are the races athletes peak for, so section 8 reports them as a separate slice.

**Did not start, did not finish, disqualified.** These rows have no valid place, so they are not targets. They can still count in later features: a DNF, for example, is a race day for the load features.

**Is the data clean?** I separate 2 things. Documented cases (DNS, DNF, DSQ, lapped relay teams, cancelled races) are not errors; they only need a rule. Dirty data is a finisher with an empty or impossible value. On the 20,179 sprint finisher rows of the seasons outside the test set:

- 209 rows are dirty, and 203 of them are lap times in 2 men's sprints of 2014/15, which no feature uses.
- Only 4 rows touch a feature: 3 finishers without ski or range time, and 1 range time of 47 minutes. The last 2 rows are in columns no feature uses.

**The range time rule.** The 47 minute value was not fixed by hand. Instead, a sprint range time above 10 minutes counts as missing. A rule also protects seasons nobody had looked at, and 10 minutes leaves room for real incidents on the range \[Expert\]. Removing that value re-ranks the rest of its race, so 274 rows changed in the range feature, 271 of them by at most 0.0045.

**Odd times kept.** 18 finishers have 77 to 132 seconds too much in their total time once ski and range time are removed. The most likely reason is a wrong number of penalty loops: 1 loop too few gives a 2 minute penalty, which is about 95 seconds once the loop time is taken off, and 13 of the 18 sit near that value \[Expert, not verified\]. The penalty is part of the official result, so these rows stay.

**Columns dropped.** Lap times (wrong in 2 races), the athlete's own start time, the hand timed shooting time, and bib and start group. Seeding information would overlap with form, and its rules changed over the years; it is a next step.

**Licence.** The IBU terms of use have no educational exception. No raw data is published: the data folder is excluded from the repository, and this report names no athlete. The 2 download scripts rebuild both files from the public source, and I can show the data to the instructor and the examiner on request.

## 3. Target and features

**Unit.** 1 row is 1 finisher in 1 sprint: 24,957 rows in 252 races. The same athlete appears in many rows, so the rows are not independent. I do not claim they are; the time ordered validation in section 4 handles it.

**Target.** The finish percentile inside the race: (rank minus 1) divided by (finishers minus 1). Athletes with the same time share the better rank.

**The golden rule.** Every feature uses only races that started before the race being predicted. History carries over from season to season, also into the test seasons.

| # | Feature | Group |
| --- | --- | --- |
| 1 | Average finish percentile, last 3 sprints | Form |
| 2 | Average ski percentile, last 3 sprints | Form |
| 3 | Average range percentile, last 3 sprints | Form |
| 4 | Average miss percentile, last 3 sprints | Form |
| 5 | Finish percentile of the last sprint | Form |
| 6 | Valid sprints in a 2 season window | Experience |
| 7 | Finish place in the last race of any format, relays excluded | Recent result |
| 8 | Mean misses per sprint over the window | Shooting |
| 9 | Miss dispersion over the window, shrunk | Shooting |
| 10 | Race days in the last 21 days, relays included | Load |
| 11 | Did not start at the previous World Cup event of the season | Load |
| 12 | First start at this season's World Championships or Olympics | Load |
| Flag | No previous sprint with a valid finish | Helper |

**Form (1 to 5).** How the athlete finished, skied and shot lately. Feature 4 uses the athlete's rank on misses within each race, not the raw count. On a windy day everybody misses, so 2 misses can be a good day; the percentile compares the athlete with the same field on the same day \[Expert\].

**Experience (6).** The window is the current season so far plus the whole previous season. The count goes down again when an athlete stops racing, which a lifetime count cannot do.

**Recent result (7).** A place, not a percentile, because a percentile does not mean the same in every format: last in a mass start is still top 30 in the world \[Expert\].

**Shooting over the window (8 and 9).** Both use sprints only, where every athlete fires the same 10 shots. Feature 9 asks a question I care about as a biathlete: is this shooter steady, or streaky? \[Expert\]

- If each shot were a coin flip with the athlete's own miss rate, we know how much the number of misses should vary from race to race.
- The dispersion is the real variance divided by that expected variance. 1 means pure luck, above 1 means streaky, below 1 steadier than luck.
- With few sprints the value is unreliable, so it is pulled toward 1, more strongly when there are fewer sprints.

This ratio is the standard check for overdispersion in count data, and the pull toward 1 follows Vasicek (1973), who did the same for stock betas in finance \[Outside\].

**World Cup load (10 to 12).** My hypothesis: weeks on the top circuit wear athletes down mentally \[Expert\]. I call it load, not fatigue, because I can measure race days, not tiredness. Relay days count, since a relay takes the whole day. IBU Cup races do not count, because a call up to the World Cup gives a boost; this is an assumption I cannot test. Section 6 tests the hypothesis by running every model with and without these 3 features.

**Missing history.**

- A past sprint without a valid finish is skipped, and the code reaches back to the next valid one.
- With fewer than 3 past sprints, the average uses those that exist.
- With no past sprint at all, the form features stay empty and the flag is 1. The filling happens later, inside the model pipeline (section 5).

## 4. Validation design and metric

**The split.** The model will be used on races that have not happened yet, so training data must come before evaluation data (L7). A random split would train on 2025 and test on 2019.

| Set | Seasons | Rows | Sprints |
| --- | --- | --- | --- |
| History only | 2012/13 | 1,996 | 20 |
| Training | 2013/14 to 2022/23 | 18,183 | 184 |
| Test | 2023/24 to 2025/26 | 4,778 | 48 |

- **2012/13 is history only.** The data starts there, so 18 % of its rows look like debutants only because their past is missing.
- **3 test seasons** is a judgment call \[Outside\]: about 19 % of the sprints, enough to average out an odd season, and 6 championship sprints for a separate slice.

**Validation inside training.** 5 walk forward folds by season: train on all training seasons before season X, validate on season X, then move forward. The folds validate on 2018/19 to 2022/23.

| Validation season | Training rows | Validation rows |
| --- | --- | --- |
| 2018/19 | 9,066 | 1,813 |
| 2019/20 | 10,879 | 1,655 |
| 2020/21 | 12,534 | 2,060 |
| 2021/22 | 14,594 | 2,017 |
| 2022/23 | 16,611 | 1,572 |

Every model uses the same folds. Rejected: random folds (they train on the future), 1 single split (it can be lucky, L7) and grouping by athlete (made for predicting people never seen, which is not our use). Appendix B lists them, and appendix C repeats the model choice on a second split.

**The metric: list MAE.** It scores the predicted result list, the output a user would see.

1. In each race the predictions are sorted into places 1 to n. Ties are broken at random, with 1 fixed seed per race.
2. Each predicted place becomes a percentile: (place minus 1) divided by (n minus 1).
3. List MAE is the mean absolute difference with the real percentile.

MAE because being 40 places off is about twice as bad as 20, not 4 times (L7). The list version judges only the order: a cautious prediction that squeezes everyone toward the middle but keeps the right order scores exactly like the truth.

**Testing the metric before trusting it.** I scored made up predictions whose right answer is known \[Outside\]:

| Made up prediction | Expected | List MAE |
| --- | --- | --- |
| The truth | about 0 (shared places) | 0.00019 |
| A cautious copy of the truth | same as the truth | same as the truth |
| The list reversed | 0.50496 | 0.50497 |
| A constant (random order) | about 0.337 | 0.33714 |

**Numbers reported but never used to choose.** List RMSE and list R² (L4), the rank correlation per race (Spearman) and the top 10 hit rate: of the 10 athletes ranked highest, how many really finish in the top 10. The last 2 are outside the course \[Outside\]; L7 allows extra numbers next to the deciding one.

**Where leakage could enter.**

| Place | Risk | Control |
| --- | --- | --- |
| Features | A feature uses a later race | Hand checks; 2 rebuild tests with 0 differences |
| Preprocessing | Fill values or scaling learned on validation rows | Fitted inside the pipeline, per fold (L3) |
| Evaluation | Row order hints at the result | Random tie break, 1 seed per race |
| Me | Looking at the test seasons | See below |

**Features are built before the split, and this is not leakage.** Building a feature is a calculation on 1 athlete's own earlier races; nothing is learned from other rows. Fitting is different: fill values, scaling and weights are learned on training folds only. 2 tests prove it. Deleting every race from 1 January 2019 leaves all earlier rows identical, and a rebuild without the test seasons gives identical rows for the other seasons.

**What I saw of the test seasons, stated openly.**

- On 11 Nov, before the split was fixed, the feature checks ran on all seasons. No model existed, so only input values were seen.
- On 16 Nov the test seasons were opened once with an earlier version of the pipeline. After that, the features, the metric and the set of models were changed, and the final test was run again with the new, frozen pipeline.
- The changes were made for reasons stated in sections 3 and 4: the wind (feature 4), a count that said nothing after its cap (feature 6), and scoring the list that a user would see. They were not made to raise the test score. Still, they came after seeing test numbers, so the test in section 7 is not fully blind. The git history shows both runs, and the lab book states it.

## 5. Models and selection

Every rule in this section was written down before the results it applies to.

**A ladder of baselines.** Each step must beat the one before it, or its extra complexity is not justified (L3, L4).

1. **Constant.** Predicts the training mean for everyone. It creates no order, so the tie break gives a random result list: the score for knowing nothing.
2. **Naive rule.** Sorts the athletes by their average finish percentile over the last 3 sprints. It learns nothing; it is what a fan with a results list would do. Empty values get the training median.
3. **Ridge and lasso.** Linear models with a penalty on large weights. Ridge uses an L2 penalty (the sum of the squared weights), which shrinks all weights but keeps them. Lasso uses an L1 penalty (the sum of the absolute weights), which can set weights exactly to 0 and so drops inputs (L4).

**The linear pipeline.** Median fill, then scaling, then the model, all fitted on the training part of each fold. The median alone would call a debutant an average athlete, which is wrong; the no history flag stays as an input so the model can learn 1 correction for all of them (L3). Scaling comes before the model because the penalty treats all weights alike (L4). Grids: ridge alpha 0.01 to 1000, lasso alpha 0.00001 to 0.1, in steps of 10.

**2 tree families.** A tree asks yes or no questions, so it can find curves and combinations a linear model cannot.

- **Random forest (L10):** 300 trees, a third of the inputs tried at each split, seed 42, median fill, no scaling. The knob is the tree depth: 2, 4, 6, 8, 12, 16 and no limit, plus 1 refinement round with the 2 depths halfway between the best and its neighbours. Depth is the bias and variance knob (L6).
- **Gradient boosting (L12):** scikit-learn's histogram version with early stopping, a budget of 2,000 trees and its own handling of missing values. Grid: learning rate 0.02, 0.05, 0.1 and 0.2, times depth 3, 4 and 6.

**The load test.** Every candidate runs twice: with the 3 load features (13 inputs) and without them (10 inputs). 1 claim, 1 change. This gives 64 candidates on the same 5 folds.

**The selection rule: one standard error (L7).**

1. Take the best mean list MAE over the 5 folds.
2. Its standard error is the standard deviation of its 5 fold scores divided by the square root of 5.
3. Every candidate within 1 standard error of the best counts as equally good.
4. Among those, the simplest wins, by an order written before any result: fewer inputs first; then linear before trees; ridge before lasso at the same number of inputs; forest before boosting; inside a family the larger penalty or the smaller trees.

For lasso, "inputs" means the largest number of nonzero weights over the 5 folds. If the choice sat at the edge of a grid, the grid would be extended once (L7).

**Considered and not used.**

| Method | Why not |
| --- | --- |
| SVM (L11) | Slow with kernels at this data size |
| Clustering (L13) | The problem has a clear target; grouping athletes by type could be a next step |
| PCA (L9) | Every feature should keep a meaning I can explain as a biathlete |
| Neural network (L14) | Allowed only as a side comparison, and networks need far more data: L14 says below about 100,000 rows; we have about 18,000 |

## 6. Validation results

All numbers are list MAE on the 5 validation seasons, as a mean (and standard deviation) over the folds. The test seasons play no role here.

**The ladder.**

| Model | List MAE | Spearman | Top 10 hit rate |
| --- | --- | --- | --- |
| Constant (random order) | 0.3373 ± 0.0046 | none | about 0.10 (chance) |
| Naive rule | 0.1751 ± 0.0050 | 0.690 | 0.499 |
| Ridge, any alpha | 0.1589 to 0.1590 | 0.743 | 0.534 |
| Lasso, alpha up to 0.001 | the same as ridge |  |  |

Random order is about 34 places off per 100 starters. Sorting by the last 3 sprints halves that to about 18. Ridge and lasso add a smaller step, to about 16.

**Where the linear models gain.** For athletes without history the naive rule scores 0.328, ridge 0.187 to 0.189: the no history flag lets the model learn that debutants usually finish in the back half.

**The penalty barely matters.** Ridge gives the same score for every alpha. Lasso is flat up to alpha 0.001, then rises to 0.1597 at 0.01 (6 or 7 inputs left) and 0.1686 at 0.1 (3 inputs left, and the flag is dropped). The differences between good alphas are about 0.0001, against a spread of 0.005 between seasons. Training and validation scores are almost equal, so the linear models do not overfit.

**What lasso drops first.** Race days in 21 days (a load feature), the sprints in the window and, in 2 folds, the last sprint's finish go at alpha 0.001. At 0.01 the miss percentile, the last sprint's finish, the dispersion and the 2 load flags follow. In ridge the miss percentile and the mean misses split their weight with opposite signs, a sign that they overlap (L4).

**4 families tie.**

| Family | Best candidate | List MAE |
| --- | --- | --- |
| Gradient boosting | learning rate 0.02, depth 3, no load | 0.15856 |
| Random forest | depth 8, no load | 0.15887 |
| Ridge | with load | 0.15895 |
| Lasso | no load | 0.15897 |

The best and the worst of these 4 differ by 0.0004, about 0.04 places per 100 starters. Trees can find curves and combinations, and they found almost nothing the linear models had missed. The limit is what the features know, not the type of model.

**Bias and variance in the trees.** The linear models score the same on training and validation. The forest learns its training races much better than new ones (a gap of 0.013 to 0.028 for the 2 best forests, and up to 0.109 without a depth limit), which grows with depth: that is variance (L6). Boosting stays close (0.003 to 0.006) and stopped early, using 24 to 480 trees, never the 2,000 limit.

**The load hypothesis: not supported.** With load minus without load, for the best candidate of each family: ridge −0.00001, lasso 0.00000, forest +0.00005, boosting +0.00002. All are far inside 1 standard error. My World Cup load features did not change the prediction in any way the data can show.

**The selection.**

1. Best candidate: boosting, 0.15856. Standard error 0.00198, so the threshold is 0.16054.
2. 56 of the 64 candidates are within the threshold.
3. The simplest of them: **lasso without load, alpha 0.01**, 0.15969. It uses at most 7 inputs over the folds. It is not at the edge of its grid.

The rule was lenient here, and I say so: the seasons differ more than 10 times more than the good candidates, so most count as equal and simplicity decides. The final model ranks 52 of 64 by raw score; the price is 0.0011, about 0.1 place per 100 starters. Taking boosting because it was first by 0.0011 would be the winner's curse (L7). What the rule bought is a model with a handful of inputs I can explain.

One tie the rule did not foresee: lasso at 0.01 scores the same with and without load. The table order picked "without load", which is also the version given fewer inputs.

**Would more data help?** I trained each family on 20 to 100 % of the training races. The lasso is flat (0.1597 at 20 % and at 100 %), and so is ridge. The forest and boosting still improve at 100 % (last steps −0.0003 and −0.0010). With more seasons, a tree model might pull ahead. For the linear models, more rows of the same kind would not help.

## 7. Test result

**How the final test was run.** The final model, lasso with alpha 0.01 on 10 inputs, was fitted once on all 10 training seasons (L7) and then predicted the 48 sprints of 2023/24 to 2025/26. Only the final model and the 2 baselines that learn nothing were scored; the other candidates were not, so nothing invites a second choice (L7). The weights were never updated during the 3 test seasons. Features still use each athlete's full history, including earlier test races, because those are facts known on race morning. Section 4 explains that the test seasons had been opened once before, with an earlier pipeline.

**The final model.** With alpha 0.01 the lasso keeps 6 of its 10 inputs. It sets the miss percentile, the last sprint's finish, the sprints in the window and the dispersion to exactly 0.

| Input | Weight (scaled inputs) |
| --- | --- |
| Ski percentile, last 3 sprints | 0.124 |
| Last place in any format | 0.049 |
| No history flag | 0.033 |
| Finish percentile, last 3 sprints | 0.027 |
| Range percentile, last 3 sprints | 0.018 |
| Mean misses over the window | 0.012 |

**The test result.**

| Model | Validation list MAE | Test list MAE |
| --- | --- | --- |
| Constant (random order) | 0.337 | 0.339 |
| Naive rule | 0.175 | 0.162 |
| **Final model** | 0.160 | **0.150** |

On 3 seasons it had not been trained on, the predicted place is on average about 15 places from the real one per 100 starters. The other scores: Spearman 0.770, top 10 hit rate 0.554 (5.5 of the real top 10 named on average), list R² 0.539.

**Per season.**

| Season | Naive rule | Final model |
| --- | --- | --- |
| 2023/24 | 0.159 | 0.145 |
| 2024/25 | 0.163 | 0.148 |
| 2025/26 | 0.165 | 0.156 |

The model beats the naive rule in every season and in 43 of the 48 races. The last season is the hardest for both. The model was trained on data up to March 2023, so in real use it should be retrained every year.

**Why the test is better than validation.** The test score (0.150) is better than validation (0.160). The naive rule, which learns nothing, improved even more (from 0.175 to 0.162), so the test seasons are simply easier to predict; the model did not get better. For the naive rule, each of the 3 is easier than any of the 10 training seasons, whose best is 0.168 in 2022/23. A likely reason, not proven: the field changed, as Russian and Belarusian athletes are absent from 2022/23 on \[Expert\]. The fair comparison is therefore with the naive rule on the same seasons.

## 8. What the model can and cannot do

The slices were chosen before the final test. The predicted list is built on the whole race; the errors are then averaged per group. The signed error is the predicted percentile minus the real one: positive means placed too far back.

| Slice | Group | Final model | Naive rule |
| --- | --- | --- | --- |
| Event | Championships | 0.143 | 0.161 |
| Event | Other sprints | 0.150 | 0.163 |
| History | With history | 0.148 | 0.158 |
| History | Without history | 0.197 | 0.289 |
| Gender | Men | 0.144 | 0.156 |
| Gender | Women | 0.155 | 0.169 |
| Real result | Top 10 % | 0.092 | 0.106 |
| Real result | Middle 80 % | 0.163 | 0.175 |
| Real result | Bottom 10 % | 0.104 | 0.127 |
| Sprints in the window | 0 to 3 | 0.161 | 0.184 |
| Sprints in the window | 4 to 9 | 0.150 | 0.161 |
| Sprints in the window | 10 or more | 0.144 | 0.153 |

The model beats the naive rule in every group.

**The real best and worst are predicted less extreme than they turn out.** The real top 10 % are placed too far back (signed error +0.074), and the real bottom 10 % too far forward (−0.088). The naive rule does the same, more strongly. This is true of any imperfect ranking: whoever wins had a good day, which no past race can show. Grouped by predicted place, the sign flips: the predicted top 10 % finish further back than the model said (−0.096), and the predicted bottom 10 % further forward (+0.086). The list itself is not squeezed, because it always uses places 1 to n.

**Newcomers.** Without history the model is far better than the naive rule (0.197 against 0.289), but still places them too far back on average (+0.076).

**Women's sprints are harder to predict.** The error is 0.155 for women and 0.144 for men, and the naive rule shows the same gap. So the gap comes from the races, not from the model. The cause was not studied.

**More recent history, better predictions.** The error falls from 0.161 for athletes with 0 to 3 sprints in the window to 0.144 for those with 10 or more.

**Championships are not harder to predict.** If anything the error is smaller, but 6 races are too few to claim more.

**Which inputs matter.** Permutation importance on the test seasons (L10, L7): shuffle 1 input 20 times and measure how much the list MAE gets worse. The features overlap, so I also shuffled them in groups \[Outside\].

The groups: results (features 1, 5 and 7), skiing (feature 2), shooting (features 3, 4, 8 and 9), experience (feature 6 and the flag).

| Group | Increase in list MAE |
| --- | --- |
| Skiing | 0.101 |
| Results | 0.033 |
| Experience | 0.007 |
| Shooting | 0.005 |

Alone, the ski percentile costs 0.101 when shuffled, the last place in any format 0.014, and every other input 0.007 or less. The prediction rests mostly on how fast the athlete skied lately.

**Why shooting barely helps.** On the training seasons I measured how strongly each measure repeats from 1 sprint to the athlete's next one.

| Measure | Correlation with the next sprint |
| --- | --- |
| Ski percentile | 0.83 |
| Finish percentile | 0.63 |
| Range percentile | 0.58 |
| Miss percentile | 0.18 |

Skiing form persists. How fast an athlete shoots is a habit, so range time repeats too; whether the shots hit is close to chance from 1 race to the next. Accuracy only shows over many sprints: the lasso dropped the miss percentile of the last 3 sprints and my steadiness feature, and kept the mean misses over the window with a small weight. \[Expert reading of our numbers\] Range time is a different case: it repeats, but it is a small part of the total time, so it moves the result little.

## 9. Limitations, fairness, next steps

**The biggest weakness.** The model gives expected places, not chances. It ranks the field, but it cannot say how likely an athlete is to win or reach the podium, and it cannot tell who will have an exceptional day. For the use that cares most about the top, such as betting, a classifier with probabilities would be needed.

**Limits of the result.**

- **The test is not fully blind.** The test seasons were opened once with an earlier pipeline before the features, the metric and the models were changed (section 4).
- **Not retrained.** The weights stayed frozen for 3 seasons, and the last season is the hardest.
- **Seasons differ in predictability,** and I do not know why. The changed field from 2022/23 on fits the data but was not tested.
- **Trees were still learning.** With more seasons, a tree model might beat the lasso.

**Why the load features may add nothing.** 3 possible reasons, none tested:

1. Recent form already shows a tired athlete.
2. Only World Cup race days are seen. Training, travel, illness and the IBU Cup are invisible.
3. The target is relative, so load shared by most of the field cancels out.

I find reason 2 the most likely: most of what tires an athlete happens outside World Cup race days. All 3 load features also partly measure level, not freshness. Race days in the last 21 days does so most: the best athletes qualify for pursuits, mass starts and relays, so they race more, and the feature has a correlation of −0.29 with the target (training rows of the last fold). The 2 flags do the same more weakly (0.17 and 0.04): newcomers and athletes left out of the relay are weaker on average. So this test shows that load adds nothing to the order beyond form. It cannot show that load has no effect on an athlete.

**Limits of the features.**

- No age limit on the last 3 sprints: in 277 training rows the oldest of them is more than 2 years old. Feature 6 shows when the window is thin.
- The first start at a major event flag depends on the race programme: in Sochi and PyeongChang it is 1 for every finisher, because the sprint was the first race.
- The IBU Cup assumption is my experience, not a measurement.

**Fairness.**

- **Women.** The model is less accurate for women's sprints. A user should trust those predictions a little less.
- **Newcomers.** They are placed too far back on average. For a strong young athlete this is too pessimistic, and a selection decision based on it would be unfair to them.
- **Use.** The numbers describe groups. They should not be read as a judgment on 1 athlete on 1 day.

**Interpretability.** The final model is 6 weights on inputs that each mean something in the sport, which is a reason to prefer it over a boosting model that scores the same.

**Next steps.**

1. A classifier with probabilities for win, podium and top 10.
2. Retraining after every season.
3. New information: IBU Cup results, seeding and cup standings, DNF or DSQ as a sign of tiredness.
4. A live prediction from a published start list.

## 10. Reproducibility and AI usage

**Workflow.** 1 script, `pipeline.py`, goes from the raw files to every table and figure, with the commands `features`, `split`, `baselines`, `models`, `robustness` and `test`. A lab book notebook imports its functions and holds every data check in a cell of its own; it looks, it never builds. This follows the course: notebooks to explore, a script for the final pipeline (L2).

| Item | How it is fixed |
| --- | --- |
| Environment | conda environment with Python 3.12, `environment.yml` with pinned versions, including `biathlonresults` 0.1.1 |
| Seeds | 42 everywhere (forest, boosting, learning curve subsets, permutation importance); tie break 1 seed per race from the race ID |
| Pipeline | Fill, scaling and model in 1 scikit-learn pipeline, fitted per fold |
| Code | Private GitHub repository, final state at commit `40d637c`; the data folder is excluded. Access is given to the instructor and the examiner on request |

**AI usage statement.** The course allows AI use, including drafting, with disclosure.

| Tool | Used for |
| --- | --- |
| Claude (chat) | Discussing every decision before coding; explaining course concepts; keeping the decision log; drafting this report text, which I directed, checked and corrected |
| Claude Code | Writing `pipeline.py`, the download scripts and the lab book cells from written briefs |
| Course AI tutor | Checking what the course material says |

**What was mine.** The question, the target, the choice of sprints, every feature idea, the load hypothesis, the steadiness feature and the switch to scoring the result list. Every decision was made in my words before code was written.

**How the AI's work was checked.**

- Features checked by hand against the raw results, and 2 rebuild tests for leakage.
- The metric tested on made up predictions with known answers (section 4).
- Every data check is a lab book cell that I ran and read.
- Several AI claims and my own guesses were tested and dropped when the data disagreed.

## References

- International Biathlon Union. Official competition results, biathlonresults.com. Downloaded in October and November 2026.
- `biathlonresults`, version 0.1.1. Python package that reads the results from biathlonresults.com.
- Vasicek, O. A. (1973). A note on using cross-sectional information in Bayesian estimation of security betas. The Journal of Finance, 28(5), 1233-1239.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

## Appendix A. Commands to reproduce

```
conda env create -f environment.yml
conda activate biathlon
python download_sprints.py      # sprint file
python download_races.py        # all races file
python pipeline.py features     # target and features
python pipeline.py split        # split and folds
python pipeline.py baselines    # constant, naive rule, ridge, lasso
python pipeline.py models       # 4 families, with and without load; selection
python pipeline.py robustness   # second split (appendix C)
# the last command only runs when TEST_UNLOCKED = True in pipeline.py; it was set by hand before the final test
python pipeline.py test         # the only command that opens the test seasons
```

## Appendix B. Validation strategies considered

| Strategy | Used? | Why |
| --- | --- | --- |
| Walk forward by season, 5 folds | Yes | Respects time like the real use; gives a mean and a spread |
| Random folds | No | Trains on the future (L7) |
| 1 single split | No | 1 number only, and 1 season can be lucky (L7) |
| Grouped by athlete | No | Made for predicting people never seen; not our use |
| Walk forward with a 5 season window | As a check | Appendix C |

## Appendix C. Robustness run

Each validation season predicted from only the 5 seasons before it, with the same 64 candidates and no new refinement round (L7: check whether the choice survives another suitable split).

| Question | Main split | Second split |
| --- | --- | --- |
| Best candidate (list MAE) | Boosting, 0.15856 | The same, 0.15854 |
| Standard error; threshold | 0.00198; 0.16054 | 0.00206; 0.16060 |
| Eligible candidates | 56 of 64 | 54 of 64 |
| Model chosen | Lasso without load, alpha 0.01 (0.15969) | The same (0.15964) |
| Load features | No clear gain | No clear gain |

On the second split, lasso with load is 0.0001 better than without, about 2 standard errors but on this split only, and far too small to matter.

## Appendix D. Candidates on validation

All 64 candidates, best first. The list MAE is the mean over the 5 folds, with the standard deviation after the ± sign. Inputs is the number given to the model; for lasso it is the largest number of nonzero weights over the folds. The threshold is 0.16054 (section 6).

| Rank | Candidate | Inputs | Validation list MAE | Within threshold |
| --- | --- | --- | --- | --- |
| 1 | Boosting, learning rate 0.02, depth 3, no load | 10 | 0.15856 ± 0.0044 | Yes |
| 2 | Boosting, learning rate 0.05, depth 4, with load | 13 | 0.15858 ± 0.0047 | Yes |
| 3 | Boosting, learning rate 0.02, depth 3, with load | 13 | 0.15858 ± 0.0044 | Yes |
| 4 | Boosting, learning rate 0.05, depth 4, no load | 10 | 0.15866 ± 0.0047 | Yes |
| 5 | Boosting, learning rate 0.05, depth 3, with load | 13 | 0.15867 ± 0.0045 | Yes |
| 6 | Boosting, learning rate 0.02, depth 4, no load | 10 | 0.15869 ± 0.0048 | Yes |
| 7 | Boosting, learning rate 0.05, depth 3, no load | 10 | 0.15869 ± 0.0046 | Yes |
| 8 | Boosting, learning rate 0.02, depth 6, with load | 13 | 0.15870 ± 0.0049 | Yes |
| 9 | Boosting, learning rate 0.1, depth 4, no load | 10 | 0.15872 ± 0.0049 | Yes |
| 10 | Boosting, learning rate 0.05, depth 6, with load | 13 | 0.15875 ± 0.0046 | Yes |
| 11 | Boosting, learning rate 0.05, depth 6, no load | 10 | 0.15877 ± 0.0049 | Yes |
| 12 | Boosting, learning rate 0.02, depth 4, with load | 13 | 0.15879 ± 0.0046 | Yes |
| 13 | Forest, depth 8, no load | 10 | 0.15887 ± 0.0049 | Yes |
| 14 | Forest, depth 10, with load | 13 | 0.15892 ± 0.0044 | Yes |
| 15 | Ridge, alpha 100, with load | 13 | 0.15895 ± 0.0048 | Yes |
| 16 | Boosting, learning rate 0.02, depth 6, no load | 10 | 0.15895 ± 0.0050 | Yes |
| 17 | Ridge, alpha 1000, with load | 13 | 0.15896 ± 0.0047 | Yes |
| 18 | Boosting, learning rate 0.1, depth 3, with load | 13 | 0.15896 ± 0.0043 | Yes |
| 19 | Ridge, alpha 10, no load | 10 | 0.15896 ± 0.0049 | Yes |
| 20 | Lasso, alpha 0.00001, no load | 10 | 0.15897 ± 0.0049 | Yes |
| 21 | Lasso, alpha 0.0001, with load | 13 | 0.15898 ± 0.0049 | Yes |
| 22 | Boosting, learning rate 0.1, depth 3, no load | 10 | 0.15898 ± 0.0046 | Yes |
| 23 | Forest, depth 10, no load | 10 | 0.15898 ± 0.0050 | Yes |
| 24 | Ridge, alpha 1, no load | 10 | 0.15898 ± 0.0049 | Yes |
| 25 | Ridge, alpha 100, no load | 10 | 0.15898 ± 0.0049 | Yes |
| 26 | Ridge, alpha 0.1, no load | 10 | 0.15899 ± 0.0049 | Yes |
| 27 | Ridge, alpha 0.01, no load | 10 | 0.15899 ± 0.0049 | Yes |
| 28 | Ridge, alpha 10, with load | 13 | 0.15900 ± 0.0049 | Yes |
| 29 | Lasso, alpha 0.00001, with load | 13 | 0.15900 ± 0.0049 | Yes |
| 30 | Ridge, alpha 0.01, with load | 13 | 0.15901 ± 0.0049 | Yes |
| 31 | Ridge, alpha 0.1, with load | 13 | 0.15901 ± 0.0049 | Yes |
| 32 | Lasso, alpha 0.001, with load | 13 | 0.15901 ± 0.0047 | Yes |
| 33 | Ridge, alpha 1, with load | 13 | 0.15901 ± 0.0049 | Yes |
| 34 | Lasso, alpha 0.0001, no load | 10 | 0.15901 ± 0.0049 | Yes |
| 35 | Forest, depth 8, with load | 13 | 0.15903 ± 0.0049 | Yes |
| 36 | Forest, depth 12, with load | 13 | 0.15907 ± 0.0046 | Yes |
| 37 | Boosting, learning rate 0.1, depth 4, with load | 13 | 0.15908 ± 0.0048 | Yes |
| 38 | Lasso, alpha 0.001, no load | 10 | 0.15909 ± 0.0048 | Yes |
| 39 | Forest, depth 7, with load | 13 | 0.15911 ± 0.0047 | Yes |
| 40 | Boosting, learning rate 0.2, depth 4, with load | 13 | 0.15914 ± 0.0051 | Yes |
| 41 | Forest, depth 7, no load | 10 | 0.15915 ± 0.0048 | Yes |
| 42 | Ridge, alpha 1000, no load | 10 | 0.15918 ± 0.0049 | Yes |
| 43 | Boosting, learning rate 0.2, depth 3, with load | 13 | 0.15925 ± 0.0049 | Yes |
| 44 | Boosting, learning rate 0.1, depth 6, with load | 13 | 0.15928 ± 0.0051 | Yes |
| 45 | Boosting, learning rate 0.2, depth 3, no load | 10 | 0.15930 ± 0.0048 | Yes |
| 46 | Forest, depth 6, with load | 13 | 0.15932 ± 0.0047 | Yes |
| 47 | Boosting, learning rate 0.1, depth 6, no load | 10 | 0.15934 ± 0.0052 | Yes |
| 48 | Boosting, learning rate 0.2, depth 6, with load | 13 | 0.15944 ± 0.0046 | Yes |
| 49 | Forest, depth 6, no load | 10 | 0.15951 ± 0.0049 | Yes |
| 50 | Boosting, learning rate 0.2, depth 4, no load | 10 | 0.15962 ± 0.0051 | Yes |
| 51 | Forest, depth 12, no load | 10 | 0.15962 ± 0.0048 | Yes |
| 52 | **Lasso, alpha 0.01, no load (final model)** | 7 | 0.15969 ± 0.0050 | Yes |
| 53 | Lasso, alpha 0.01, with load | 7 | 0.15970 ± 0.0050 | Yes |
| 54 | Forest, depth 16, with load | 13 | 0.15983 ± 0.0044 | Yes |
| 55 | Boosting, learning rate 0.2, depth 6, no load | 10 | 0.16013 ± 0.0051 | Yes |
| 56 | Forest, depth 16, no load | 10 | 0.16046 ± 0.0047 | Yes |
| 57 | Forest, depth no limit, no load | 10 | 0.16057 ± 0.0047 | No |
| 58 | Forest, depth no limit, with load | 13 | 0.16063 ± 0.0044 | No |
| 59 | Forest, depth 4, no load | 10 | 0.16095 ± 0.0048 | No |
| 60 | Forest, depth 4, with load | 13 | 0.16101 ± 0.0046 | No |
| 61 | Forest, depth 2, with load | 13 | 0.16522 ± 0.0044 | No |
| 62 | Forest, depth 2, no load | 10 | 0.16545 ± 0.0043 | No |
| 63 | Lasso, alpha 0.1, no load | 3 | 0.16865 ± 0.0047 | No |
| 64 | Lasso, alpha 0.1, with load | 3 | 0.16865 ± 0.0047 | No |

## Appendix E. How this project was built with AI

| Step | Decided by me, with Claude in chat | Built by Claude Code | How it was checked |
| --- | --- | --- | --- |
| 1 Data | Sprints only, seasons, championships | Download scripts | Row counts; sprint rows equal in both files |
| 2 Target and features | Target, 12 features, range time rule, redesign of 17 Nov | `features` command | Hand checks; 2 leak tests |
| 3 Split and metric | Test seasons, folds, list MAE | `split` command, metric functions | Metric tested on made up predictions |
| 4 Baselines | Ladder, fill rule, ridge and lasso grids | `baselines` command | Lab book section 4 |
| 5 Models | 4 families, load test, selection rule | `models` and `robustness` commands | Second split; lab book section 5 |
| 6 Test | What is scored, slices | `test` command | Lab book section 6 |
| 7 Report | Outline, content, corrections | Draft text by Claude | Every number checked against the project files |
