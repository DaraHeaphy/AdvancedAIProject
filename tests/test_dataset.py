"""Checks for time leakage, team perspective and the teammate handoff."""

import csv
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_CSV, load_datasets, parse_kickoff, read_fixtures


def row_for(fixture_id, year, month, day, home="A", away="B", hour=15):
    date = datetime(year, month, day)
    season_start = year if month >= 8 else year - 1
    return {
        "id": str(fixture_id), "home": home, "away": away,
        "date": date.strftime("%A, %B ") + str(day), "year": str(season_start),
        "time (utc)": f"{hour:02}:00", "league": f"{season_start}-{season_start + 1} Premier League",
        "game_status": "FT", "home_score": "2", "away_score": "1",
    }


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.csv_path = Path(self.temp.name) / "matches.csv"

    def write_rows(self, rows):
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def synthetic_rows(self):
        rows = []
        for year in (2018, 2019, 2020):
            for day in range(1, 11):
                home, away = ("A", f"C{day}") if day % 2 else (f"C{day}", "A")
                rows.append(row_for(f"{year}-A-{day}", year, 8, day, home, away))
                rows.append(row_for(f"{year}-B-{day}", year, 8, day, "B", f"D{day}"))
            # This earlier kickoff must not enter the same-date target's history.
            rows.append(row_for(f"{year}-same-day", year, 8, 11, "A", "E", hour=10))
            rows.append(row_for(f"{year}-target", year, 8, 11, hour=20))
        return rows

    def test_calendar_year_including_covid_summer_and_leap_day(self):
        for year, month, day in [(2020, 7, 1), (2020, 2, 29), (2021, 1, 2), (2021, 8, 13)]:
            kickoff, _ = parse_kickoff(row_for("1", year, month, day))
            self.assertEqual(kickoff, datetime(year, month, day, 15, tzinfo=timezone.utc))
        row = row_for("1", 2020, 7, 1)
        row["league"] = "2019-20 English Premier League"
        self.assertEqual(parse_kickoff(row)[1], "2019-2020")
        row["league"] = "2019/2020 English Premier League"
        self.assertEqual(parse_kickoff(row)[1], "2019-2020")

    def test_duplicates_abandoned_and_invalid_scores(self):
        rows = self.synthetic_rows()
        abandoned = dict(rows[0], id="abandoned", game_status="Abandoned")
        self.write_rows(rows + [rows[0].copy(), abandoned])
        fixtures, counts = read_fixtures(self.csv_path)
        self.assertEqual(len(fixtures), len(rows))
        self.assertEqual(counts["excluded_duplicate_rows"], 1)
        self.assertEqual(counts["excluded_non_completed"], 1)
        self.write_rows(rows + [dict(rows[0], home_score="3")])
        with self.assertRaisesRegex(ValueError, "Conflicting rows"):
            read_fixtures(self.csv_path)
        self.write_rows([dict(rows[0], home_score="-1")])
        with self.assertRaisesRegex(ValueError, "Invalid completed fixture"):
            read_fixtures(self.csv_path)

    def test_handoff_orientation_and_same_day_exclusion(self):
        self.write_rows(list(reversed(self.synthetic_rows())))
        splits = load_datasets(self.csv_path)
        train = splits["train"]
        index = next(i for i, m in enumerate(train.metadata) if m.target.fixture_id == "2018-target")
        home, away, label = train[index]
        self.assertEqual(home.shape, (10, 3))
        self.assertEqual(away.shape, (10, 3))
        self.assertEqual(home.dtype, torch.float32)
        self.assertEqual(away.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.int64)
        self.assertEqual(label.shape, torch.Size([]))
        self.assertEqual(home[0].tolist(), [2, 1, 1])
        self.assertEqual(home[1].tolist(), [1, 2, 0])
        self.assertNotIn("2018-same-day", [f.fixture_id for f in train.metadata[index].home_history])
        self.assertEqual(label.item(), 0)
        fixture = train.metadata[index].target
        self.assertEqual(replace(fixture, home_score=1, away_score=1).label, 1)
        self.assertEqual(replace(fixture, home_score=0, away_score=1).label, 2)
        batch = next(iter(DataLoader(train, batch_size=2)))
        self.assertEqual(batch[0].shape, (min(2, len(train)), 10, 3))
        self.assertEqual(batch[2].shape, (min(2, len(train)),))

    def test_split_seasons_and_cross_season_history(self):
        rows = [r for r in self.synthetic_rows() if r["id"] != "2019-target"]
        rows.append(row_for("2019-early-target", 2019, 8, 1, hour=10))
        self.write_rows(rows)
        splits = load_datasets(self.csv_path)
        for name, season in [("train", "2018-2019"), ("val", "2019-2020"), ("test", "2020-2021")]:
            self.assertEqual({m.target.season for m in splits[name].metadata}, {season})
        val = splits["val"]
        example = next(m for m in val.metadata if m.target.fixture_id == "2019-early-target")
        self.assertTrue(any(f.season == "2018-2019" for f in example.away_history))

    def test_real_data_all_histories_are_exact_previous_ten(self):
        fixtures, _ = read_fixtures(DEFAULT_CSV)
        splits = load_datasets()
        by_team = {}
        for fixture in fixtures:
            for team in (fixture.home, fixture.away):
                by_team.setdefault(team, []).append(fixture)
        targets = set()
        for split in splits.values():
            for i, metadata in enumerate(split.metadata):
                target = metadata.target
                self.assertNotIn(target.fixture_id, targets)
                targets.add(target.fixture_id)
                for team, history, tensor in [
                    (target.home, metadata.home_history, split.home_histories[i]),
                    (target.away, metadata.away_history, split.away_histories[i]),
                ]:
                    expected = [f for f in by_team[team] if f.kickoff.date() < target.kickoff.date()][-10:]
                    self.assertEqual(list(history), expected)
                    self.assertEqual(tensor.tolist(), [f.features_for(team) for f in expected])
                self.assertEqual(split.labels[i].item(), target.label)
        report = splits["train"].report
        self.assertEqual(sum(s["completed_fixtures"] for s in report["splits"].values()), report["unique_completed_fixtures"])
        for summary in report["splits"].values():
            self.assertEqual(summary["examples"] + summary["skipped_insufficient_history"], summary["completed_fixtures"])


if __name__ == "__main__":
    unittest.main()
