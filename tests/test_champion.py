from __future__ import annotations

import json

from benchmarks import champion


def _result(n: int, r1: float) -> dict:
    return {"n": n, "R@1": r1, "R@5": 1.0, "R@10": 1.0, "by_type": {}}


def test_experimental_eval_never_rewrites_champion(tmp_path, monkeypatch):
    champ = tmp_path / "CHAMPION.json"
    ledger = tmp_path / "eval_ledger.jsonl"
    champ.write_text(json.dumps({"n": 500, "R@1": 0.958}))
    monkeypatch.setattr(champion, "CHAMPION", champ)
    monkeypatch.setattr(champion, "LEDGER", ledger)

    out = champion.update(
        _result(500, 0.999),
        {"config": "experimental", "deterministic": True},
    )

    assert out["champion"] is False
    assert out["eligible_for_champion"] is False
    assert json.loads(champ.read_text())["R@1"] == 0.958
    assert len(ledger.read_text().splitlines()) == 1


def test_sub500_eval_is_never_eligible(tmp_path, monkeypatch):
    champ = tmp_path / "CHAMPION.json"
    ledger = tmp_path / "eval_ledger.jsonl"
    monkeypatch.setattr(champion, "CHAMPION", champ)
    monkeypatch.setattr(champion, "LEDGER", ledger)

    out = champion.update(
        _result(100, 1.0),
        {
            "eligible_for_champion": True,
            "deterministic": True,
            "config": "candidate",
        },
    )

    assert out["champion"] is False
    assert out["eligible_for_champion"] is False
    assert not champ.exists()


def test_explicit_full_deterministic_candidate_can_promote(tmp_path, monkeypatch):
    champ = tmp_path / "CHAMPION.json"
    ledger = tmp_path / "eval_ledger.jsonl"
    champ.write_text(json.dumps({"n": 500, "R@1": 0.958}))
    monkeypatch.setattr(champion, "CHAMPION", champ)
    monkeypatch.setattr(champion, "LEDGER", ledger)

    out = champion.update(
        _result(500, 0.96),
        {
            "eligible_for_champion": True,
            "deterministic": True,
            "config": "verified-full",
        },
    )

    assert out["champion"] is True
    assert out["eligible_for_champion"] is True
    assert json.loads(champ.read_text())["R@1"] == 0.96


def test_nondeterministic_full_run_cannot_promote(tmp_path, monkeypatch):
    champ = tmp_path / "CHAMPION.json"
    ledger = tmp_path / "eval_ledger.jsonl"
    champ.write_text(json.dumps({"n": 500, "R@1": 0.958}))
    monkeypatch.setattr(champion, "CHAMPION", champ)
    monkeypatch.setattr(champion, "LEDGER", ledger)

    out = champion.update(
        _result(500, 0.99),
        {
            "eligible_for_champion": True,
            "deterministic": False,
            "config": "nondeterministic",
        },
    )

    assert out["champion"] is False
    assert json.loads(champ.read_text())["R@1"] == 0.958
