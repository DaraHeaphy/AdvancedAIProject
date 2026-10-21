# Revised positional-encoding study

Prepared 21 October 2026.

Does fixed sinusoidal positional encoding improve next-fixture prediction compared with no positional encoding?

The earlier data loader retained histories after missing Premier League seasons, including Leeds results from 2004 in 2020 inputs. The corrected loader clears a returning team's history and waits for ten new results. Ordinary consecutive-season histories are retained. The architecture, optimiser, seeds, features and training budget are unchanged.

**The evaluation season was already inspected before this correction. This is a revised analysis, not a newly untouched test.** No model, seed, epoch budget, threshold or ensemble was selected using the revised test scores. The revision addresses a concrete input-data problem. Lecturer agreement with any final-week revision and recorded gate decisions must be established from the team's records.

| Split | Examples | Excluded for insufficient history | Maximum history age |
| --- | ---: | ---: | ---: |
| train | 6604 | 616 | 183 days |
| val | 352 | 28 | 197 days |
| test | 338 | 29 | 162 days |

No retained history contains a result over a year old. All inputs precede their target UTC date, and target splits are disjoint and chronological. Training spans 2001-02 through 2019-20, validation is 2020-21 and evaluation is the available 2021-22 season. The first twenty source seasons contain all 380 pairings; the final season has 367 and ends on 17 May 2022. See `data_audit.json` for the thirteen missing pairings and excluded fixture participations.

The shared encoder projects three per-match features (scored/conceded goals and home status) to width 32, applies one transformer layer with two heads, feedforward width 64 and dropout 0.1, then mean-pools each team's ten outputs. Concatenated home/away summaries map linearly to three logits. All weights are trained from scratch. There are 8,867 learned weights/biases; fixed position vectors add zero. The fixed sizes and ten-match context are pragmatic CPU-study choices, not demonstrated optima.

Parameter breakdown: projection 128; attention query/key/value 3,168; attention output 1,056; feedforward 4,192; two layer normalisations 128; classifier 195. The same encoder weights process both teams. Without positions, mean-pooled predictions are invariant to history permutations in evaluation mode. Sinusoidal encoding permits ordering to matter, but does not guarantee that the trained model learns useful recency.

Both conditions use Adam at 0.001, batch size 32, ten epochs, one CPU thread and seeds 42/43/44. Each checkpoint is selected by its lowest validation log loss. All six runs are retained; there is no best-seed selection or ensemble. The constant reference uses training class frequencies only. The no-position transformer is the informative reference for the isolated positional-encoding change.

| Condition | Seed | Selected epoch | Validation log loss | Test log loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 42 | 1 | 1.036403 | 1.000699 | 54.44% |
| sinusoidal | 42 | 1 | 1.034800 | 0.998422 | 54.73% |
| none | 43 | 8 | 1.041701 | 0.995733 | 52.37% |
| sinusoidal | 43 | 8 | 1.042392 | 0.997797 | 53.55% |
| none | 44 | 4 | 1.046301 | 0.995800 | 53.55% |
| sinusoidal | 44 | 4 | 1.048353 | 0.993092 | 55.33% |

| Condition | Mean test log loss | Sample SD | Mean test accuracy |
| --- | ---: | ---: | ---: |
| none | 0.997411 | 0.002848 | 53.45% |
| sinusoidal | 0.996437 | 0.002914 | 54.54% |

Sinusoidal encoding lowers test log loss in 2 of three seed pairs. Mean paired difference: -0.000974; sample SD: 0.002640. Negative favours sinusoidal. Three seeds on one incomplete season do not establish statistical significance or a general advantage.

Validation mean paired difference is +0.000380. The constant predictor's test log loss is 1.075881. The test log-loss direction varies across seeds, and the mean advantage is smaller than the paired sample SD. This provides no consistent advantage under the revised setup.

All six revised models predict zero draws by argmax, despite 80 actual draws. Their draw recall is therefore zero. This does not mean they assign zero draw probability or establish whether their probabilities are calibrated. It is a serious limitation for a three-way classification claim; log loss remains the primary probability metric. See `class_errors.csv` for all class counts. No class weighting or thresholds were adjusted after evaluation.

These scores must not be compared directly with the earlier experiment to estimate the causal effect of the history correction: the corrected train/validation/test populations changed. Within the revised comparison, both positional conditions receive identical targets and information.

Limitations include only three seeds, a single incomplete evaluation season, class-distribution changes over time, and excluded early promoted-team fixtures. Shared teams and overlapping histories make fixture errors dependent. Seed SD measures training randomness on these fixtures, not uncertainty across future seasons. Checkpoint selection can make validation scores optimistic. Inputs omit opponent identities/strength, squad changes and elapsed time; the final linear head has additive home/away contributions rather than a nonlinear interaction. Neither competitiveness against stronger football baselines nor betting value is established.

Earlier completed evaluation results can enter later histories with frozen weights. This represents sequential pre-match forecasting; it is not a whole-season forecast issued before the season starts. A causal mask is unnecessary because every input result is already in the past. Cross-entropy consumes logits, while exported probabilities use softmax. Exact mid-run training resumption is not supported; completed study runs can be reused through their manifest.

Compute: six fresh ten-epoch CPU runs, 8,867 parameters each; total training/validation-loop time 314.21 seconds, runner wall time 397.53 seconds. Historical pre-correction runs are separate, archived evidence and are not additional independent seeds. Environment versions and processor details are recorded in `metrics.json` and the dependency lock.

Verification: all 30 automated tests passed, including missing-season reset and Leeds regression checks, loss aggregation, permutation invariance, checkpoint round trips, corruption rejection and paired metrics. All six revised validation checkpoints and test prediction sets were reloaded and independently verified. The audit checked every retained history. Source/data hashes and selected epochs were checked. These checks validate implementation and reproducibility; they do not independently prove the history of human test inspection or individual contributions.

Dataset provenance and declared publisher terms are in `../dataset/README.md`. This educational study uses sporting results and excludes player fields from model inputs. The public availability of source material does not independently establish upstream rights. No pretrained model is used.

Codex assisted with coding, review, corrections, verification and prose. Its earlier stale-history policy and excessive generated artifacts required correction. Regression tests, fixture coverage checks, checkpoint reloads and independent probability calculations check those changes. Both students must review the results and understand the full pipeline. Commit authorship does not establish unaided authorship; the final contribution statement requires confirmation of actual work. A raw prompt log is not required by the supplied specification.

Run instructions are in the root README. Compact config, metrics and individual results are versioned here; generated run histories, snapshots, weights and documents remain local. A verified archive preserves the original state and a separate revised evidence ZIP contains the full corrected experiment. Cross-platform numerical identity is not guaranteed.
