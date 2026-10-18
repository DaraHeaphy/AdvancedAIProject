# Advanced AI project

Team Trivial: Dara Heaphy (23369914) and Tiernan Scully (23365528).
Track 3.3: a small transformer trained from scratch.

## Final study

**Question:** Does fixed sinusoidal positional encoding improve next-fixture
home/draw/away probability prediction compared with no positional encoding,
under the same small-model training budget?

The shared encoder has 8,867 trainable parameters. Inputs are each team's ten
previous completed Premier League matches, with goals scored, goals conceded
and home status. Both conditions use one layer, width 32, two heads, feedforward
width 64, dropout 0.1, mean pooling and a linear classifier. Fixed positional
vectors add no parameters. All other choices are held fixed.

Training: Adam, learning rate 0.001, batch size 32, ten epochs, CPU, one thread.
Seeds 42, 43 and 44 are paired across conditions. Minimum validation log loss
selects each checkpoint. Every seed is retained; there is no best-seed selection
or ensemble. The training-frequency constant predictor is a sanity baseline;
the no-position transformer is the informative reference for the main question.

## Submission material

- `research/week7/REPORT.md`: final written analysis and all-six-run tables.
- `research/week7/protocol.json`: protocol frozen before test evaluation.
- `research/week7/results.json`: full-precision held-out results.
- `research/week7/verification.json`: checkpoint and metric verification.
- `output/docx/Advanced AI Project - Positional Encoding Study.docx`: updated Week 7 document.
- `output/pptx/Advanced AI Project - Week 7.pptx`: group presentation.
- `research/week7_notes/DEFENCE.md`: individual preparation notes.
- `research/week7_notes/SUBMISSION_CHECKLIST.md`: final administrative checks.
- `output/submission/`: portable submission ZIP and its audit.

The final package preserves the earlier Week 6 analysis. Its original six
checkpoints and its named comparison ZIP were absent from this checkout, so
Week 7 uses a transparently labelled fresh reproduction of the same six-run
protocol in `research/week7_reproduction/`. No model or training hyperparameter
was changed to improve the outcome. The reproduction uses the available Python
3.11/PyTorch 2.6 CPU environment; earlier records used a different environment.
The final report distinguishes the historical results from the reproduction.

## Installation

From this repository's root in PowerShell, using Python 3.11:

```powershell
python -m venv .venv_submission
.\.venv_submission\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv_submission\Scripts\python.exe -m unittest discover -s tests -v
```

The root lock records the Week 7 training environment. Every run also records
its installed dependency closure, exact PyTorch build, hardware, configuration,
source snapshots and data SHA-256. Bitwise identity on different hardware or
software is not guaranteed. Analysis dependencies are pinned in the root lock.

## Verify the submitted results

Extract the complete submission ZIP if working from a fresh clone: binary
checkpoints and per-run source snapshots are intentionally ignored by Git.

```powershell
python -m unittest discover -s tests -v
python comparison.py report --output research/week7_reproduction
python final_evaluation.py verify --output research/week7
```

Verification reloads all six models and reproduces their saved predictions and
independent probability metrics. It does not train or change model selection.
The final evaluation verifier also checks the frozen source, data, manifest and
checkpoint hashes. Do not edit frozen experiment source before verifying.

## Reproduce training separately

Use new directories so submitted evidence is preserved:

```powershell
python comparison.py run --output research/reproduction_new
python final_evaluation.py freeze --study research/reproduction_new --output research/evaluation_new
python final_evaluation.py evaluate --output research/evaluation_new
python final_evaluation.py verify --output research/evaluation_new
```

Only run final evaluation after development is fixed. Do not use test outcomes
to choose architecture, features, epochs, checkpoints, seeds or thresholds.
Repeated verification checks the same frozen result; it is not another model
selection opportunity. An interrupted evaluation leaves its started record
rather than silently replacing partial results.

## Data preparation and information flow

`dataset/matches.csv` contains 7,979 rows. Excluding two abandoned fixtures and
ten duplicates leaves 7,967 unique completed fixtures. Histories use only the
previous ten completed fixtures from earlier UTC dates, oldest to newest.
All targets on a date are built before any result from that date enters a history.
The calendar year is resolved from the season and recorded weekday, including
the delayed 2020 matches. Invalid or conflicting completed fixtures raise errors.

Target seasons define the chronological split: 6,912 training examples
(2001-2002 to 2019-2020), 380 validation examples (2020-2021), and 357 test
examples (available 2021-2022 fixtures). Insufficient histories are skipped.
Histories may cross season and split boundaries. At final evaluation, a later
test fixture can use an earlier completed test fixture's result, consistent
with sequential pre-match forecasting. Weights stay frozen. This is not a
forecast of every fixture made simultaneously before the season starts.

Each item is `(home_history, away_history, label)`, tensors `[10,3]`, `[10,3]`
and a scalar integer; labels 0/1/2 mean home win/draw/away win. Team names and
fixture IDs are audit metadata, not features. No scaling, padding, team IDs or
elapsed-time inputs are used. The final season is incomplete.

## Model and evaluation

`model.py` applies a shared `Linear(3,32)` and `TransformerEncoderLayer` to each
team, averages each sequence, concatenates the summaries and applies
`Linear(64,3)`. No causal mask is needed: all input matches precede the target.
Without positional encoding the encoder is permutation equivariant; averaging
makes the summary invariant to history order in evaluation mode. Position lets
order affect predictions, but does not guarantee useful recency learning.

`train.py` uses cross-entropy directly on logits. Validation disables dropout
and gradients and weights loss by examples, including the final partial batch.
Prediction uses softmax. Primary evaluation is multiclass log loss (natural
logarithm); accuracy is secondary. Brier score and confusion matrices describe
probability and class behaviour. Standard deviation across three seeds measures
training variation on the same fixtures, not uncertainty across future seasons.

```powershell
python train.py train --positional-encoding none --seed 42 --epochs 10 --device cpu
python dataset.py --report tmp/data_report.json
python train.py overfit --positional-encoding none --seed 42 --steps 300 --device cpu
```

Dataset provenance and publisher-declared terms: `dataset/README.md`.
AI assistance, corrections and remaining team confirmations are recorded in
the final analysis and submission checklist. Both members must understand the
full experiment. The original Week 5/6 instructions remain archived in
`research/week7_notes/README_before_week7.md`.
