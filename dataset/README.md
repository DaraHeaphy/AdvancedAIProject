# Dataset

Only `matches.csv` is required. From 7,979 source rows, excluding two abandoned fixtures and ten duplicates leaves 7,967 completed fixtures. All twenty seasons from 2001-02 through 2020-21 have all 380 ordered home/away pairings. Available 2021-22 data has 367 fixtures and ends on 17 May 2022. The experiment excludes that entire incomplete 2021-22 season. All included seasons have all 380 pairings. Run `python audit.py` to inspect inclusion flags, coverage and history exclusions.

Publisher: Joseph Mohr, [English Premier League Match Events and Results](https://www.kaggle.com/datasets/josephvm/english-premier-league-game-events-and-results/metadata). The publisher identifies ESPN as the scraped source and declares CC0/Public Domain. This records declared publisher terms, rather than independently establishing upstream ESPN rights. The source was supplied with this project; identity with a particular Kaggle version has not been established.

Unmodified local CSV SHA-256: `b60a46d09867d8cb74e4eb48fe1ec3cbd1dd846a7eec87f6b1057133356abad1`.

The source year is the season start. Season labels and weekdays resolve calendar dates, including delayed July 2020 matches. Model inputs use goals scored/conceded and home status only. Player fields are unused.

Ten earlier-date results are required for each team. Histories cross consecutive seasons and reset after a missing season. Early promoted-team fixtures are excluded until enough new history exists. No padding, scaling, identity embeddings or elapsed-time features are applied. Training ends in 2018-19, validation is 2019-20 and testing is 2020-21. The original CSV remains unchanged; exclusion is explicit in the loader and data report.
