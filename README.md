# Football match-history transformer

Team Trivial: Dara Heaphy (23369914) and Tiernan Scully (23365528).
Advanced AI project, track 3.3: a small transformer trained from scratch.

Does fixed sinusoidal positional encoding improve home/draw/away probability prediction compared with the same transformer without positions?

Both conditions share a one-layer team encoder: three inputs projected to width 32, two attention heads, feedforward width 64, dropout 0.1 and mean pooling. The home and away summaries feed a linear three-class head. Each model has 8,867 trainable parameters; fixed sinusoidal positions add none.

## Data and correction

Inputs are each team's ten previous completed Premier League matches: goals scored, goals conceded and home status. Only results from earlier UTC dates enter a history. Histories carry across consecutive seasons but reset after a missing season. Returning teams must accumulate ten new matches. Names and dates are audit metadata, rather than model inputs.

Training targets span 2001-02 through 2019-20 (6,604 examples); validation is 2020-21 (352); evaluation uses available 2021-22 fixtures (338). The final source season has 367 completed fixtures rather than 380. Missing source fixtures and insufficient-history exclusions are different issues. See `dataset/README.md` for provenance and `results/data_audit.json` for the audit.

The earlier experiment retained histories through relegation gaps, including Leeds results from 2004 in 2020 inputs. This revision corrects that policy. **The test season was already inspected. These revised results are a reanalysis, not a newly untouched holdout.** Architecture, seeds and training settings were held fixed.

## Run

Use Python 3.11 from the repository root. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe audit.py
.\.venv\Scripts\python.exe comparison.py run --output research/reproduction_validation
.\.venv\Scripts\python.exe final_evaluation.py freeze --study research/reproduction_validation --output research/reproduction_evaluation
.\.venv\Scripts\python.exe final_evaluation.py evaluate --output research/reproduction_evaluation
.\.venv\Scripts\python.exe final_evaluation.py verify --output research/reproduction_evaluation
```

Use new output directories for each fresh comparison. Both conditions use Adam at 0.001, batch size 32, ten epochs, CPU and one thread. Seeds 42, 43 and 44 are paired; every seed is retained. Minimum validation log loss selects each checkpoint. Primary comparison is paired multiclass log loss; accuracy, Brier score and class errors are secondary. A constant predictor uses training class frequencies only.

Earlier completed evaluation results may enter later histories while model weights stay frozen. This is sequential pre-match prediction. For one run use `python train.py train --positional-encoding none --seed 42`. `python train.py overfit --steps 300` checks learning on eight training examples. Checkpoints support inference; exact mid-run training resumption is not implemented.

Training creates ignored `runs/` weights, configurations, source snapshots, environments and predictions. The comparison creates ignored `research/` reports and figures. Verification checks executed source bytes and data, so run it before editing that source. Different hardware or software can produce numerical differences.

## Evidence and limits

`results/` retains the compact audit, configuration, individual run tables and analysis. Regenerate full local artifacts with the commands above. Old documents, unused CSVs, duplicated snapshots and run histories are excluded from Git. A verified local archive preserves the original project and original Git history.

Three seeds on one incomplete evaluation season provide limited evidence. Shared teams and overlapping histories make fixtures dependent; seed variation does not measure future-season uncertainty. Excluding early promoted-team fixtures narrows the evaluated population. Inputs omit opponent strength, squad changes and elapsed time. This comparison does not establish the best football predictor or betting value.

Codex assisted with implementation, review, tests, analysis and writing. Earlier stale histories and excessive generated material required correction. Checks include regression tests, a data audit, checkpoint reloads and independent probability metrics. Historical commit authorship is preserved, but is not proof of unaided contribution. Actual individual contributions and lecturer gate decisions require the team's records. Both students remain responsible for understanding the project. The project specification permits AI assistance and requires reflection on its use.
