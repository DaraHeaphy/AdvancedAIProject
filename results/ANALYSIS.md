# Positional encoding with complete source seasons

Does fixed sinusoidal positional encoding improve next-fixture prediction compared with no positional encoding?

The experiment explicitly excludes the incomplete 2021-22 season before constructing histories or target splits. Training spans 2001-02 through 2018-19, validation is 2019-20 and testing is 2020-21. The original CSV remains unchanged. All twenty included seasons contain all 380 ordered home/away pairings. The excluded season has 367 unique completed fixtures. Its thirteen missing pairings do not affect this experiment.

**2020-21 previously served as validation and selected checkpoints in earlier runs. Moving it into the test role does not make it an untouched holdout. These results are a revised chronological evaluation.** We selected this split for complete source-season coverage, held the architecture and training settings fixed and retained all seeds. We did not choose a model, threshold or ensemble based on the new test outcome.

| Split | Target seasons | Source fixtures | Usable targets | History exclusions |
| --- | --- | ---: | ---: | ---: |
| train | 2001-2002 to 2018-2019 | 6840 | 6253 | 587 |
| val | 2019-2020 | 380 | 351 | 29 |
| test | 2020-2021 | 380 | 352 | 28 |

A complete source season and complete model coverage are different claims. The 2020-21 source has all 380 fixtures, but 28 fail the existing requirement that both teams have ten valid prior results. We evaluate the remaining 352. Promoted-team exclusions narrow the evaluated population.

Each team supplies its ten previous completed Premier League results: goals scored, goals conceded and home status. Inputs use earlier UTC dates only. Histories persist across consecutive seasons and reset after a missing season. This prevents stale records such as Leeds results from 2004 entering its 2020 inputs. No result from the excluded season enters either targets or histories.

The audit checks fixture pairing coverage, duplicate/abandoned-row handling, exact history lengths, missing-season gaps, history ages and disjoint chronological targets. The independent real-data test checks the exact previous ten eligible results and team perspective. Structural completeness does not independently verify every score or timestamp against ESPN. The source contains 7,979 raw rows, 7,967 unique completed fixtures after cleaning and 7,600 fixtures after excluding 2021-22. See `data_audit.json`.

The model architecture is unchanged: a shared projection from 3 to 32 channels, one transformer encoder layer with two heads, feedforward width 64, dropout 0.1 and mean pooling. Home and away summaries concatenate to 64 values and a linear head returns three logits. The same weights process both teams. There are 8,867 learned scalar weights/biases in both conditions. The 320 fixed sinusoidal values add no trainable parameters.

Parameter breakdown: projection 128; attention query/key/value 3,168; attention output 1,056; feedforward 4,192; two layer normalisations 128; classifier 195. Width, context length and budget are pragmatic small CPU-study choices rather than demonstrated optima. The position-free encoder followed by mean pooling is invariant to history reordering in evaluation mode. Position vectors allow order to matter but do not guarantee useful recency learning.

Both conditions train from scratch using Adam at 0.001, batch size 32, ten epochs, one CPU thread and paired seeds 42, 43 and 44. Lowest validation log loss on 2019-20 selects each checkpoint. All six runs remain in the analysis. The smaller training set has 196 batches per epoch. Cross-entropy consumes logits; exported probabilities use softmax. The primary metric is multiclass log loss using natural logs, with accuracy, Brier score and class behaviour as secondary measures. The additional constant predictor uses training class frequencies only.

| Condition | Seed | Selected epoch | Validation log loss | Test log loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 42 | 9 | 0.984636 | 1.066951 | 47.44% |
| sinusoidal | 42 | 9 | 0.984257 | 1.072184 | 46.31% |
| none | 43 | 8 | 0.988454 | 1.068002 | 47.16% |
| sinusoidal | 43 | 8 | 0.991329 | 1.066248 | 45.45% |
| none | 44 | 10 | 0.987293 | 1.066857 | 44.03% |
| sinusoidal | 44 | 10 | 0.990084 | 1.072594 | 44.89% |

| Condition | Mean test log loss | Sample SD | Mean test accuracy |
| --- | ---: | ---: | ---: |
| none | 1.067270 | 0.000635 | 46.21% |
| sinusoidal | 1.070342 | 0.003551 | 45.55% |

Sinusoidal encoding lowers test log loss in 1 of three seed pairs and validation log loss in 1 of three pairs. The test direction varies across seeds, so the comparison gives no consistent test advantage.

Mean test paired difference (sinusoidal minus none): +0.003072; sample SD 0.004187. Negative favours sinusoidal. Mean validation paired difference: +0.001762. The constant predictor has test log loss 1.097316 and accuracy 38.07%.

| Condition | Seed | Actual draws | Predicted draws | Correct draws | Draw recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 42 | 77 | 0 | 0 | 0.00% |
| sinusoidal | 42 | 77 | 0 | 0 | 0.00% |
| none | 43 | 77 | 0 | 0 | 0.00% |
| sinusoidal | 43 | 77 | 0 | 0 | 0.00% |
| none | 44 | 77 | 0 | 0 | 0.00% |
| sinusoidal | 44 | 77 | 1 | 0 | 0.00% |

Draw recall describes argmax classification. It does not mean that a model assigns zero draw probability, and it cannot establish calibration. Log loss scores the probability assigned to actual draws. We did not adjust class weights or thresholds after inspecting these results.

The no-position transformer supplies an informative reference for the isolated positional change. Beating the constant prior does not establish that attention is necessary or that a transformer is competitive with stronger football predictors. The final linear head combines additive home/away contributions and cannot add a further nonlinear interaction between summaries.

Only three seeds and one test season are available. Fixtures share teams and overlapping histories, so they are dependent observations. Seed SD measures training randomness on these fixtures rather than uncertainty across future seasons. Class frequencies and football conditions can change over time. Inputs omit opponent identity, squad changes and elapsed days. Validation-selected scores can be optimistic. Previous use of the test season for validation limits confirmatory claims.

Earlier completed test results can enter later histories while model weights stay frozen. This is sequential pre-match prediction, rather than forecasting every fixture before the season starts. Every input is in the past, so the encoder needs no causal attention mask. Checkpoints support inference but do not contain the full state for exact mid-run training resumption.

The previous experiment evaluated available 2021-22 fixtures. These new scores cannot establish the causal effect of changing the split because the training, validation and test populations all changed. Each reported positional comparison uses identical information and targets within its own experiment. Earlier evidence remains archived and is not counted as extra independent seeds.

Compute: six fresh ten-epoch CPU runs with 8,867 parameters each. Total training/validation-loop time 323.56 seconds; runner wall time 401.75 seconds. Environment and processor details are in `metrics.json` and the dependency lock.

All 31 automated tests passed. They cover exclusion of 2021-22 from all model inputs and targets, exact real-data histories, promotion resets, calendar-year parsing, permutation invariance, weighted validation loss, checkpoint round trips, corrupted evidence and paired aggregation. All six selected validation checkpoints and test prediction sets were reloaded and checked against independent probability metrics. The frozen protocol and hashes identify the executed source and data. These checks do not independently establish individual contributions or the history of human data inspection.

Dataset provenance and declared publisher terms are in `../dataset/README.md`. Joseph Mohr identifies ESPN as the source and declares CC0/Public Domain. We record these declared terms rather than independently verifying upstream rights or a particular publisher version. The educational model uses score/home-status inputs and excludes player fields. No pretrained model is used, and the results establish no betting value.

Codex assisted with code, audit, analysis and writing. Its stale-history policy needed correction, followed by regression checks. This further split change follows the team's preference to exclude incomplete source seasons. Both students must understand the full pipeline and confirm actual individual contributions. Lecturer gate decisions and acceptance of the revised evaluation require the team's records.

The root README documents reproduction. Compact configuration, audit and all individual results remain versioned. Generated checkpoints, source snapshots, reports and presentations stay outside Git. A verified archive preserves the preceding experiment and a separate ZIP preserves the complete-season experiment. Cross-platform numerical identity is not guaranteed.
