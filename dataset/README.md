# Raw dataset

Copied from the project's existing local `../dataset` directory. The six CSV
files are unchanged; use `matches.csv` for the first transformer attempt.

| File | Contents / first-attempt use |
| --- | --- |
| `matches.csv` | Fixtures, scores and other match fields; primary input |
| `events.csv` | Match events; not needed initially |
| `all_tables.csv` | Raw source table; not needed initially |
| `agg_stats/assist_leaders.csv` | Aggregate statistics; not needed initially |
| `agg_stats/goal_leaders.csv` | Aggregate statistics; not needed initially |
| `agg_stats/team_discipline.csv` | Aggregate statistics; not needed initially |

Initial inspection of `matches.csv`: 7,979 rows, 7,977 with status `FT` and two
with status `Abandoned`. There are 21 season labels spanning 2001-2002 through
2021-22; the final season contains 377 rows. This inspection is not a full data
quality audit.

The implemented loader found ten duplicate rows in 2021-22 and removes them,
leaving 367 unique completed fixtures in that final season and 7,967 across the
full dataset after excluding the two abandoned rows. See
`../research/data_report.json` for the preparation counts and example trace.

The project proposal identifies the source as the Kaggle English Premier League
Match Events and Results dataset, acquired from ESPN according to its publisher:
https://www.kaggle.com/datasets/josephvm/english-premier-league-game-events-and-results

The proposal reports CC0/Public Domain licensing. Source attribution and licence
are carried over from the proposal; they have not been independently verified
against the downloaded files in this step.
