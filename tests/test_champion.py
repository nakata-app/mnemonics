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


def test_equal_verified_reproduction_can_fill_missing_by_type(tmp_path, monkeypatch):
    champ = tmp_path / "CHAMPION.json"
    ledger = tmp_path / "eval_ledger.jsonl"
    champ.write_text(json.dumps({
        "n": 500,
        "R@1": 0.958,
        "R@5": 1.0,
        "R@10": 1.0,
        "by_type": None,
        "config": "original",
        "date": "2026-06-21T00:00:00+00:00",
    }))
    monkeypatch.setattr(champion, "CHAMPION", champ)
    monkeypatch.setattr(champion, "LEDGER", ledger)

    result = _result(500, 0.958)
    result["by_type"] = {
        "temporal-reasoning": {"n": 10, "R@1": 0.9, "R@5": 1.0, "R@10": 1.0}
    }
    out = champion.update(
        result,
        {
            "eligible_for_champion": True,
            "deterministic": True,
            "config": "verified reproduction",
            "source": "kaggle_champion.py",
            "pinned_sha": "abc123",
        },
    )

    assert out["champion"] is False
    assert out["enriched"] is True
    saved = json.loads(champ.read_text())
    assert saved["R@1"] == 0.958
    assert saved["config"] == "original"
    assert saved["by_type"] == result["by_type"]
    assert saved["reproduction_source"] == "kaggle_champion.py"
    assert saved["reproduction_sha"] == "abc123"


def test_equal_reproduction_cannot_replace_existing_by_type(tmp_path, monkeypatch):
    champ = tmp_path / "CHAMPION.json"
    ledger = tmp_path / "eval_ledger.jsonl"
    existing = {"knowledge-update": {"n": 1, "R@1": 1.0, "R@5": 1.0, "R@10": 1.0}}
    champ.write_text(json.dumps({
        "n": 500,
        "R@1": 0.958,
        "R@5": 1.0,
        "R@10": 1.0,
        "by_type": existing,
    }))
    monkeypatch.setattr(champion, "CHAMPION", champ)
    monkeypatch.setattr(champion, "LEDGER", ledger)

    result = _result(500, 0.958)
    result["by_type"] = {"other": {"n": 1, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0}}
    out = champion.update(
        result,
        {
            "eligible_for_champion": True,
            "deterministic": True,
            "config": "verified reproduction",
        },
    )

    assert out["champion"] is False
    assert out["enriched"] is False
    assert json.loads(champ.read_text())["by_type"] == existing
