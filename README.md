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

Use Python 3.10 or later with PyTorch installed (`python -m pip install torch`).
The loader otherwise uses only the Python standard library. From the repo root:

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

Verified locally with Python 3.11 and PyTorch 2.6.0+cpu. Model construction and
training remain Tiernan's half of the work.


