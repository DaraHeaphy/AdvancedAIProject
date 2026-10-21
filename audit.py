"""Audit fixture coverage, history ages and chronological separation."""

import argparse
import json
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

from dataset import DEFAULT_CSV, EVALUATION_STATUS, load_datasets, read_fixtures


def audit_data():
    fixtures, counts = read_fixtures(DEFAULT_CSV)
    datasets = load_datasets()
    seasons = defaultdict(list)
    for fixture in fixtures:
        seasons[fixture.season].append(fixture)
    coverage = []
    for season, matches in sorted(seasons.items()):
        teams = sorted({team for f in matches for team in (f.home, f.away)})
        missing = sorted(set(permutations(teams, 2)) - {(f.home, f.away) for f in matches})
        included = season not in datasets["train"].report["excluded_seasons"]
        if included:
            assert len(teams) == 20 and not missing, "Included season is incomplete"
        coverage.append({"season": season, "included_in_experiment": included,
                         "teams": len(teams), "fixtures": len(matches),
                         "missing_pairs": missing, "last_fixture": matches[-1].kickoff.isoformat()})
    ages = {}
    targets = set()
    for name, dataset in datasets.items():
        affected = 0
        max_age = 0
        skipped_teams = Counter()
        included = {m.target.fixture_id for m in dataset.metadata}
        split_seasons = set(dataset.report["splits"][name]["seasons"])
        for fixture in fixtures:
            if fixture.season in split_seasons and fixture.fixture_id not in included:
                skipped_teams.update((fixture.home, fixture.away))
        for metadata in dataset.metadata:
            target = metadata.target
            assert target.season not in dataset.report["excluded_seasons"]
            assert target.fixture_id not in targets, "Target appears in multiple splits"
            targets.add(target.fixture_id)
            old = False
            for history in (metadata.home_history, metadata.away_history):
                assert len(history) == 10
                assert all(f.season not in dataset.report["excluded_seasons"] for f in history)
                assert all(f.kickoff.date() < target.kickoff.date() for f in history)
                years = sorted({int(f.season[:4]) for f in (*history, target)})
                assert all(b == a + 1 for a, b in zip(years, years[1:])), "History spans missing season"
                age = (target.kickoff - history[0].kickoff).days
                max_age = max(max_age, age)
                old |= age > 365
            affected += old
        ages[name] = {"targets": len(dataset), "targets_with_history_over_365_days": affected,
                      "maximum_history_age_days": max_age,
                      "skipped_fixture_participations_by_team": dict(sorted(skipped_teams.items()))}
    for earlier, later in (("train", "val"), ("val", "test")):
        assert max(m.target.kickoff for m in datasets[earlier].metadata) < min(
            m.target.kickoff for m in datasets[later].metadata)
    report = dict(datasets["train"].report)
    assert sum(s["completed_fixtures"] for s in report["splits"].values()) == report["included_completed_fixtures"]
    assert report["included_completed_fixtures"] + report["excluded_season_fixtures"] == report["unique_completed_fixtures"]
    report["source"] = "dataset/matches.csv"
    return {"passed": True, "checks": ["disjoint chronological target splits",
            "ten earlier-date results per history", "no history crosses a missing season",
            "all included source seasons have complete fixture pairings",
            "excluded season absent from targets and histories"],
            "preparation": report, "season_coverage": coverage, "history_ages": ages,
            "evaluation_status": EVALUATION_STATUS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/data_audit.json"))
    args = parser.parse_args()
    result = audit_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": result["passed"], "splits": result["preparation"]["splits"],
                      "history_ages": result["history_ages"]}, indent=2))


if __name__ == "__main__":
    main()
