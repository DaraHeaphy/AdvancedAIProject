# Football match-history transformer

Team Trivial: Dara Heaphy (23369914) and Tiernan Scully (23365528).
Advanced AI project, track 3.3: a small transformer trained from scratch.

Does fixed sinusoidal positional encoding improve home/draw/away probability prediction compared with the same transformer without positions?

Both conditions share a one-layer team encoder: three inputs projected to width 32, two attention heads, feedforward width 64, dropout 0.1 and mean pooling. The home and away summaries feed a linear three-class head. Each model has 8,867 trainable parameters; fixed sinusoidal positions add none.

## Data and correction

Inputs are each team's ten previous completed Premier League matches: goals scored, goals conceded and home status. Only results from earlier UTC dates enter a history. Histories carry across consecutive seasons but reset after a missing season. Returning teams must accumulate ten new matches. Names and dates are audit metadata, rather than model inputs.

Training targets span 2001-02 through 2018-19 (6,253 examples); validation is 2019-20 (351); testing is 2020-21 (352). Every included source season contains all 380 home/away pairings. We exclude the incomplete 2021-22 season entirely before constructing histories or splits, while preserving the original CSV. The test season supplies 380 fixtures, of which 28 lack sufficient recent history and are excluded. A complete source season does not mean every fixture qualifies for this model. See `dataset/README.md` and `results/data_audit.json`.

The earlier experiment retained histories through relegation gaps, including Leeds results from 2004 in 2020 inputs. This revision corrects that policy. We subsequently moved the chronological split back one season to use complete source seasons only. This chronological evaluation keeps the architecture, seeds and training settings fixed.

## Run

For an executed visual walkthrough, open [`notebooks/Project_Walkthrough.ipynb`](notebooks/Project_Walkthrough.ipynb). It explains the data audit, an actual input example, chronological splits, the shared model and parameter count, all six learning curves, paired results and confusion matrices. It imports the existing implementation and reads recorded evidence; **Run All does not retrain models or select new checkpoints**. Outputs are included for viewing without starting a kernel.

To run it interactively, install the optional notebook tools into the project environment and launch Jupyter from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m notebook notebooks/Project_Walkthrough.ipynb
```

In VS Code, select `.venv\Scripts\python.exe` as the notebook kernel. `results/notebook_evidence.json` contains the compact recorded curves, confusion matrices and validation example, with hashes identifying their original local artifacts. It lets the walkthrough run without ignored training folders. If the original checkpoint is available, the final example also checks its hash and reproduces its prediction. The existing scripts below remain the route for reproducing the complete training experiment.

Use Python 3.11 from the repository root. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe audit.py
.\.venv\Scripts\python.exe comparison.py run --output research/complete_seasons_reproduction_validation
.\.venv\Scripts\python.exe final_evaluation.py freeze --study research/complete_seasons_reproduction_validation --output research/complete_seasons_reproduction_evaluation
.\.venv\Scripts\python.exe final_evaluation.py evaluate --output research/complete_seasons_reproduction_evaluation
.\.venv\Scripts\python.exe final_evaluation.py verify --output research/complete_seasons_reproduction_evaluation
```

Use new output directories for each fresh comparison. Both conditions use Adam at 0.001, batch size 32, ten epochs, CPU and one thread. Seeds 42, 43 and 44 are paired; every seed is retained. Minimum validation log loss selects each checkpoint. Primary comparison is paired multiclass log loss; accuracy, Brier score and class errors are secondary. A constant predictor uses training class frequencies only.

Earlier completed evaluation results may enter later histories while model weights stay frozen. This is sequential pre-match prediction. For one run use `python train.py train --positional-encoding none --seed 42`. `python train.py overfit --steps 300` checks learning on eight training examples. Checkpoints support inference; exact mid-run training resumption is not implemented.

Training creates ignored `runs/` weights, configurations, source snapshots, environments and predictions. The comparison creates ignored `research/` reports and figures. Verification checks executed source bytes and data, so run it before editing that source. Different hardware or software can produce numerical differences.

## Evidence and limits

`results/` retains the compact audit, configuration, individual run tables and analysis. Individual contribution statements belong in the separate project write-up.

Regenerate full local experiment artifacts with the commands above. Existing ignored `runs/` and `research/` folders retain the six trained checkpoints and detailed evidence for verification. Historical archives, study guides, the HTML walkthrough export, the stale Word report and the specification reference were moved outside the repository to the sibling folder `AdvancedAIProject_support_archive_20261021`. Its cleanup manifest records SHA-256 hashes for all 22 moved files. The notebook remains the walkthrough format. The local virtual environment and generated caches can be recreated and are excluded from Git.

Three seeds on one evaluation season provide limited evidence. The 2020-21 season also differs from earlier training seasons. Shared teams and overlapping histories make fixtures dependent; seed variation does not measure future-season uncertainty. Excluding early promoted-team fixtures narrows the evaluated population. Inputs omit opponent strength, squad changes and elapsed time. This comparison does not establish the best football predictor or betting value.

Codex assisted with implementation, review, tests, analysis and writing. Earlier stale histories and excessive generated material required correction. Checks include regression tests, a data audit, checkpoint reloads and independent probability metrics. Historical commit authorship is preserved, but is not proof of unaided contribution. Actual individual contributions belong in the separate project write-up; lecturer gate decisions require the team's records. Both students remain responsible for understanding the project. The project specification permits AI assistance and requires reflection on its use.
