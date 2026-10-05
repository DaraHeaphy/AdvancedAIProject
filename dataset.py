"""Prepare ten-match team histories; no training or positional encoding here."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

import torch
from torch.utils.data import Dataset


HISTORY_LENGTH = 10
FEATURES = ["goals_scored", "goals_conceded", "played_at_home"]
LABELS = ["home_win", "draw", "away_win"]
DEFAULT_CSV = Path(__file__).resolve().parent / "dataset" / "matches.csv"


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    home: str
    away: str
    kickoff: datetime
    season: str
    home_score: int
    away_score: int

    @property
    def label(self):
        if self.home_score > self.away_score:
            return 0
        return 1 if self.home_score == self.away_score else 2

    def features_for(self, team):
        if team == self.home:
            return [self.home_score, self.away_score, 1]
        if team == self.away:
            return [self.away_score, self.home_score, 0]
        raise ValueError(f"{team!r} did not play in fixture {self.fixture_id}")


@dataclass(frozen=True)
class ExampleMetadata:
    target: Fixture
    home_history: tuple
    away_history: tuple


class MatchHistoryDataset(Dataset):
    """Items are (home float32 [10,3], away float32 [10,3], int64 label).

    `metadata[index]` retains the target and source fixtures for auditing.
    `report` contains preparation counts shared by all three datasets.
    """

    def __init__(self, metadata, report):
        self.metadata = tuple(metadata)
        self.report = report
        self.home_histories = torch.tensor(
            [[f.features_for(m.target.home) for f in m.home_history] for m in metadata],
            dtype=torch.float32,
        ).reshape(-1, HISTORY_LENGTH, 3)
        self.away_histories = torch.tensor(
            [[f.features_for(m.target.away) for f in m.away_history] for m in metadata],
            dtype=torch.float32,
        ).reshape(-1, HISTORY_LENGTH, 3)
        self.labels = torch.tensor([m.target.label for m in metadata], dtype=torch.int64)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.home_histories[index], self.away_histories[index], self.labels[index]


def parse_kickoff(row):
    """The source year is a season start; weekday resolves the calendar year."""
    match = re.match(r"^(\d{4})[-/](\d{4}|\d{2})\b", row["league"])
    if not match:
        raise ValueError(f"Unrecognised season: {row['league']!r}")
    start = int(match[1])
    end = int(match[2])
    if end < 100:
        end += (start // 100) * 100
        if end <= start:
            end += 100
    if end != start + 1 or int(row["year"]) != start:
        raise ValueError("Season and source year disagree")
    weekday, month_day = row["date"].split(", ", 1)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    expected_weekday = weekdays.index(weekday)
    candidates = []
    for year in (start, end):
        try:
            day = datetime.strptime(f"{month_day} {year}", "%B %d %Y")
        except ValueError:
            continue  # February 29 can be valid in only one of the two years.
        if day.weekday() == expected_weekday:
            candidates.append(day)
    if len(candidates) != 1:
        raise ValueError(f"Cannot resolve calendar year for {row['date']!r}")
    time = datetime.strptime(row["time (utc)"], "%H:%M").time()
    kickoff = datetime.combine(candidates[0].date(), time, tzinfo=timezone.utc)
    return kickoff, f"{start}-{end}"


def read_fixtures(csv_path):
    fixtures = []
    counts = Counter()
    seen_ids = {}
    seen_fixtures = {}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "home", "away", "date", "year", "time (utc)", "league",
                    "game_status", "home_score", "away_score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, start=2):
            counts["raw_rows"] += 1
            if row["game_status"] != "FT":
                counts["excluded_non_completed"] += 1
                continue
            try:
                kickoff, season = parse_kickoff(row)
                home_score, away_score = int(row["home_score"]), int(row["away_score"])
                home, away = row["home"].strip(), row["away"].strip()
                if min(home_score, away_score) < 0 or not home or not away or home == away or not row["id"]:
                    raise ValueError("Invalid score, teams or fixture ID")
                fixture = Fixture(row["id"], home, away, kickoff, season, home_score, away_score)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid completed fixture at CSV line {line}: {error}") from error
            previous = seen_ids.get(fixture.fixture_id)
            if previous is not None:
                if previous != fixture:
                    raise ValueError(f"Conflicting rows for fixture ID {fixture.fixture_id}")
                counts["excluded_duplicate_rows"] += 1
                continue
            # One home/away pairing per Premier League season; also catches different IDs.
            key = (season, home, away)
            previous = seen_fixtures.get(key)
            if previous is not None:
                if (previous.kickoff, previous.home_score, previous.away_score) != (kickoff, home_score, away_score):
                    raise ValueError(f"Conflicting duplicate fixture: {key}")
                counts["excluded_duplicate_rows"] += 1
                seen_ids[fixture.fixture_id] = fixture
                continue
            seen_ids[fixture.fixture_id] = fixture
            seen_fixtures[key] = fixture
            fixtures.append(fixture)
    fixtures.sort(key=lambda f: (f.kickoff, f.fixture_id))
    return fixtures, {key: counts[key] for key in
                      ("raw_rows", "excluded_non_completed", "excluded_duplicate_rows")}


def load_datasets(csv_path=DEFAULT_CSV):
    """Return train/val/test datasets, splitting by target season.

    Histories cross season/split boundaries and use earlier UTC dates only.
    No scaling, random split, padding or positional features are applied.
    """
    csv_path = Path(csv_path)
    fixtures, counts = read_fixtures(csv_path)
    seasons = sorted({f.season for f in fixtures})
    if len(seasons) < 3:
        raise ValueError("Need at least three seasons for train/validation/test splits")
    split_seasons = {"train": seasons[:-2], "val": seasons[-2:-1], "test": seasons[-1:]}
    season_to_split = {season: split for split, values in split_seasons.items() for season in values}
    examples = {split: [] for split in split_seasons}
    eligible = Counter()
    skipped = Counter()
    histories = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))

    # Construct every target on a date before adding any results from that date.
    # Kickoff is not completion time: this conservative rule prevents same-day leakage.
    for _, day_fixtures in groupby(fixtures, key=lambda f: f.kickoff.date()):
        day_fixtures = list(day_fixtures)
        for fixture in day_fixtures:
            split = season_to_split[fixture.season]
            eligible[split] += 1
            home, away = histories[fixture.home], histories[fixture.away]
            if len(home) < HISTORY_LENGTH or len(away) < HISTORY_LENGTH:
                skipped[split] += 1
                continue
            examples[split].append(ExampleMetadata(fixture, tuple(home), tuple(away)))
        for fixture in day_fixtures:
            histories[fixture.home].append(fixture)
            histories[fixture.away].append(fixture)

    report = {
        "source": str(csv_path.resolve()),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        **counts,
        "unique_completed_fixtures": len(fixtures),
        "history_length": HISTORY_LENGTH,
        "features": FEATURES,
        "labels": LABELS,
        "history_policy": "Previous ten completed fixtures from earlier UTC dates; crosses seasons",
        "scaling": "none",
        "splits": {},
    }
    for split, items in examples.items():
        class_counts = Counter(m.target.label for m in items)
        report["splits"][split] = {
            "seasons": split_seasons[split],
            "completed_fixtures": eligible[split],
            "skipped_insufficient_history": skipped[split],
            "examples": len(items),
            "class_counts": {label: class_counts[i] for i, label in enumerate(LABELS)},
            "first_target": items[0].target.kickoff.isoformat() if items else None,
            "last_target": items[-1].target.kickoff.isoformat() if items else None,
        }
    return {split: MatchHistoryDataset(items, report) for split, items in examples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=Path("research/data_report.json"))
    args = parser.parse_args()
    datasets = load_datasets(args.csv)
    report = dict(datasets["train"].report)
    # Trace a validation example; do not inspect test predictions during development.
    val = datasets["val"]
    if len(val):
        home, away, label = val[0]
        report["validation_example"] = {
            "metadata": asdict(val.metadata[0]),
            "home_features": home.tolist(),
            "away_features": away.tolist(),
            "label": label.item(),
            "label_name": LABELS[label.item()],
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=lambda value: value.isoformat()) + "\n", encoding="utf-8")
    print(f"Read {report['raw_rows']} rows; kept {report['unique_completed_fixtures']} unique completed fixtures.")
    print(f"Excluded {report['excluded_non_completed']} non-completed rows and {report['excluded_duplicate_rows']} duplicates.")
    for split, summary in report["splits"].items():
        print(f"{split}: {summary['examples']} examples; {summary['skipped_insufficient_history']} skipped; {summary['class_counts']}")
    print(f"Saved preparation report and traced validation example to {args.report}")


if __name__ == "__main__":
    main()
