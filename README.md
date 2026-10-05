# AdvancedAIProject
Team Members:
Dara Heaphy 23369914
Tiernan Scully 23365528

This git repository will track the development of our small transformer model for the Advanced AI module's group project.

This project will follow track 3.3 from the project spec: https://advanced-ai.eoin.ai/project/spec/

Our current experiment focuses on how positional information affects prediction
of the next fixture's home win / draw / away win probabilities using the previous
10 matches of each team.

The raw CSV dataset is included under [dataset/](dataset/README.md). Start with
`dataset/matches.csv`; event and aggregate statistics are not needed initially.

## Data preparation (Dara's half)

Install the dependencies as shown below. The loader otherwise uses only the
Python standard library. From the repo root:

```powershell
python dataset.py
python -m unittest discover -s tests -v
```

The first command writes [research/data_report.json](research/data_report.json),
including preprocessing counts, split seasons, class counts and a validation
example traced to all twenty historical fixtures. No manual preprocessing is
needed. To use a different CSV or report location:

```powershell
python dataset.py --csv dataset/matches.csv --report research/data_report.json
```

Tiernan can consume the datasets directly:

```python
from torch.utils.data import DataLoader
from dataset import load_datasets

datasets = load_datasets()
loader = DataLoader(datasets["train"], batch_size=32, shuffle=True)
home_history, away_history, labels = next(iter(loader))
# Histories: float32 [batch, 10, 3]; labels: int64 [batch].
# Features: goals_scored, goals_conceded, played_at_home.
# Labels: 0 = home win, 1 = draw, 2 = away win.

example = datasets["val"].metadata[0]  # Target and source fixtures for inspection.
report = datasets["train"].report
```

Only completed (`FT`) fixtures enter histories. Identical duplicate fixtures are
removed; conflicting duplicates and invalid completed rows raise errors rather
than silently changing the data. The source's `year` is the season start year;
the loader resolves the calendar year using the season and recorded weekday,
including the delayed July 2020 matches.

Histories contain the previous ten completed fixtures from **earlier UTC calendar
dates**, ordered oldest to newest. Completion times are absent, so all targets on
a date are constructed before adding any results from that date. Histories cross
season boundaries and retain earlier known results across split boundaries.
There is no scaling, padding, team ID feature or positional feature. Targets with
insufficient history are skipped.

Training uses seasons 2001-2002 through 2019-2020; validation uses 2020-2021 and
testing uses the available 2021-2022 data. After preprocessing, the splits contain
6,912 / 380 / 357 examples, respectively. Ten duplicate rows mean the raw final
season's 377 rows represent only 367 unique completed fixtures; ten of these
are skipped for insufficient history. The test set is reserved for final evaluation.

Data preparation was originally verified with Python 3.11 and PyTorch 2.6.0+cpu.
The model and training pipeline below preserve this loader and its interface.

## Installation and verification

In **PowerShell**, start in this repository. The implementation and commands were
verified on Windows with Python 3.14.2 and PyTorch 2.14.0+cpu. `requirements.txt`
pins the PyTorch package version; the actual build and dependency versions are
recorded for each run. The raw CSVs are already included.

```powershell
Set-Location C:\Users\User\repos\AdvancedAIProject
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The checks should finish with `Ran 12 tests` and `OK`: five existing data tests
and seven model/pipeline tests. The new checks cover both positional settings,
finite `[batch, 3]` logits, usable gradients, equal parameter counts, probability
sums, exact checkpoint reloads, independent history permutations, weighted
validation loss, a training-only frequency baseline and reproducible distinct
runs. Small pipeline tests supply only training and validation datasets.

## Model and training (Tiernan's half)

`model.py` projects each match's three raw features to 32 features. The same
projection and single `TransformerEncoderLayer` process the home and away
histories separately. The layer has two attention heads, a 64-feature feedforward
network, dropout 0.1 and `batch_first=True`. Each team's ten outputs are mean
pooled; the home summary is concatenated before the away summary and passed to
`Linear(64, 3)` for raw home win / draw / away win logits.

The only experimental setting exposed by training is `--positional-encoding`:
`none` (default) or `sinusoidal`. The latter adds fixed sine/cosine values at
positions 0 through 9 after projection. These values are a registered buffer,
move with the model and add no trainable parameters. Both conditions have
**8,867 trainable parameters**. There is no causal mask. Mean pooling means the
`none` model ignores within-history ordering in evaluation mode; it still knows
which summary belongs to the home team. Neither condition adds date or recency
features. Histories remain oldest to newest.

Run the first implementation from the repository root in **PowerShell**:

```powershell
python train.py train --positional-encoding none --seed 42 --epochs 10 --device cpu
```

These are also the defaults for `python train.py train`. Training uses Adam with
learning rate 0.001, batch size 32 and cross-entropy applied directly to logits.
Training batches shuffle with a separate seeded generator; validation batches
do not shuffle. Seeds are set before model construction or batch iteration.
Losses are averaged over examples, including the final smaller batch. Validation
uses evaluation mode and no gradients. The checkpoint with the lowest validation
multiclass log loss (natural logarithm) is retained, regardless of accuracy.

CPU is the default device. `--device cuda` requires a working CUDA-enabled
PyTorch installation. `--threads 1` is the default, used for this small model's
first run. Deterministic algorithms are enabled; exact numerical reproducibility
across other dependency versions or hardware is not guaranteed. For a later
sinusoidal run, change only `--positional-encoding sinusoidal` and use the same
seed and training settings.

Each invocation prints a new `runs/<UTC timestamp>_<condition>_seed<seed>_<id>/`
directory. Existing run directories are never reused. Each directory contains:

| File | Contents |
| --- | --- |
| `config.json` | Seed, model configuration and fixed training settings |
| `environment.json` | Python/dependency versions, PyTorch build, device, CPU and thread details |
| `data_report.json` | Source SHA-256, preparation counts and split definitions with a portable source path |
| `baseline.json` | Training class frequencies and their validation log loss/accuracy |
| `metrics.json` | Training loss, validation log loss and accuracy for every epoch |
| `summary.json` | Parameter count, training-loop time, best epoch and verified reload metrics |
| `validation_prediction.json` | Fixture metadata, true label and three probabilities from the reloaded best model |
| `best.pt` | Best model weights, fixed position buffer, model/run configuration and selected epoch |

The loader prepares all three datasets, but the training and prediction commands
use only training and validation. No test metrics or test predictions are
calculated. The constant baseline predicts the training class proportions for
every validation fixture and selects the training majority class for accuracy;
validation frequencies do not fit the baseline.

## Reload a checkpoint and predict

For the verified first run, this **PowerShell** command reloads the saved model
configuration and weights and predicts validation item zero:

```powershell
python train.py predict --checkpoint runs/20261005T151017Z_none_seed42_a6de1590/best.pt --index 0 --device cpu
```

The binary checkpoint is local and ignored by Git. After cloning, regenerate it
with the ten-epoch training command above. To predict from the newest regenerated
`none`, seed-42 run, use:

```powershell
$run = Get-ChildItem .\runs -Directory -Filter '*_none_seed42_*' | Sort-Object LastWriteTime | Select-Object -Last 1
python train.py predict --checkpoint (Join-Path $run.FullName 'best.pt') --index 0 --device cpu
```

The output includes fixture ID, home/away teams, kickoff, true label, checkpoint
epoch and named probabilities. Softmax is applied here for prediction. Other
validation fixtures can be selected with `--index` from 0 to 379. If the source
hash differs from the run's neighbouring data report, prediction raises an error.

## First run: 5 October 2026

The single real-data run used `none`, seed 42, ten epochs, CPU and one PyTorch
thread. It trained on all 6,912 training examples and selected its checkpoint
using all 380 validation examples. The training/validation loop took 10.15
seconds on the recorded machine. The selected epoch was **1**, using log loss.

| Validation result | Multiclass log loss | Accuracy |
| --- | ---: | ---: |
| Training-frequency constant baseline | 1.097809 | 37.89% |
| Selected transformer checkpoint | 1.048161 | 47.63% |

Checkpoint reload reproduced the validation metrics and saved prediction exactly.
Validation item 0 is fixture **578653**, **Fulham vs Arsenal**, true label **2
(away win)**. Its probabilities are home win **0.310528**, draw **0.286784** and
away win **0.402688**. On 32 validation examples, ten trials independently
permuting each team's history changed the trained `none` model's probabilities
by at most **1.19e-7**, within the specified numerical tolerance.

The compact record is committed under
[`runs/20261005T151017Z_none_seed42_a6de1590/`](runs/20261005T151017Z_none_seed42_a6de1590/):
[configuration](runs/20261005T151017Z_none_seed42_a6de1590/config.json),
[every epoch](runs/20261005T151017Z_none_seed42_a6de1590/metrics.json),
[summary](runs/20261005T151017Z_none_seed42_a6de1590/summary.json),
[prediction](runs/20261005T151017Z_none_seed42_a6de1590/validation_prediction.json)
and [verification](runs/20261005T151017Z_none_seed42_a6de1590/verification.json),
alongside dependency, data and baseline records. Only `runs/**/*.pt` is ignored;
JSON results remain visible to Git. Regeneration creates a distinct directory,
rather than overwriting this record.

The separate small-batch diagnostic repeatedly trained on the first eight
**training** examples for 300 Adam steps. Evaluation loss fell from **1.105211**
to **0.000902**, a **99.92%** reduction, passing the predefined 50% reduction
threshold. It did not select a checkpoint or use validation. Its compact result
is [research/overfit_none_seed42.json](research/overfit_none_seed42.json). Recheck
it from the repository root in **PowerShell**:

```powershell
python train.py overfit --positional-encoding none --seed 42 --steps 300 --device cpu
```

An optional `--output <new-file.json>` saves this diagnostic without overwriting
an existing file. The diagnostic is separate from the ten-epoch run.

This is a pipeline sanity check, not evidence of a positional effect. The main
study still needs at least three independent seeds per condition, the same
seeds across conditions, reporting every run and final held-out evaluation.
The logistic regression baseline is deferred. The final season remains incomplete.


