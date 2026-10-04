"""_temporal_rerank: refactor parity with the old inline block, and the
label-routing switch.

The legacy implementation is reproduced verbatim as the reference. With
label_routing=True (the default) the extracted function must reorder rows
exactly like it. label_routing=False must switch off ONLY the ordinal branch,
which is the one gated on the dataset's gold question_type.
"""
from __future__ import annotations

import importlib.util
import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "lme_harness", Path(__file__).resolve().parent.parent / "benchmarks" / "longmemeval_eval.py"
)
H = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(H)

_parse_lme_date = H._parse_lme_date
_session_id_of = H._session_id_of
_detect_relative_target = H._detect_relative_target
_detect_ordinal = H._detect_ordinal
_event_date_of = H._event_date_of
datetime = datetime  # used by the verbatim block below


def _legacy_temporal(rows, q, temporal_v2, temporal_v3):
    """The pre-extraction inline block of evaluate_mnemonics, verbatim except
    that `result["results"]` is the local `rows` (mutated in place -> returned)."""
    sid_to_date: dict[str, datetime] = {}
    for sid_x, d_str in zip(
        q.get("haystack_session_ids", []),
        q.get("haystack_dates", []) or [],
    ):
        sdate = _parse_lme_date(d_str)
        if sdate is not None:
            sid_to_date[sid_x] = sdate

    def _rdate(r):
        return sid_to_date.get(_session_id_of(r.get("text")) or "")

    target_info = _detect_relative_target(
        q.get("question", ""), q.get("question_date"),
        require_count=temporal_v2,
        weekdays=temporal_v3,
    )
    if sid_to_date and target_info is not None:
        # "N ago": promote in-window candidates AND, within the window,
        # rank by closeness to the target date so the date-correct chunk
        # wins #1 (the CE frequently leaves the gold stuck at rank 2).
        target_date, tol = target_info

        def _in_win(r):
            d = _rdate(r)
            return d is not None and abs((d - target_date).days) <= tol

        in_window = [r for r in rows if _in_win(r)]
        out_window = [r for r in rows if not _in_win(r)]
        in_window.sort(key=lambda r: abs((_rdate(r) - target_date).days))
        rows = in_window + out_window
    elif sid_to_date and q.get("question_type") == "temporal-reasoning":
        # Ordinal/comparative ("first/earliest/order" vs "last/latest"):
        # sort dated candidates chronologically so the chronological
        # extreme lands at #1; undated keep their CE order behind.
        # Gated to temporal-reasoning questions: "first/last" fire on
        # non-temporal queries ("last name", "first purchase") ~6.5% of
        # the time and would corrupt currently-correct answers. The gate
        # uses the dataset label, so this measures the lever's ceiling;
        # production would route via temporal-intent detection instead.
        direction = _detect_ordinal(q.get("question", ""), v2=temporal_v2)
        if direction is not None:
            rows = rows
            if temporal_v2:
                # Relevance-scoped (v2): the date decides only among
                # the top-5 CE-ranked candidates. Sorting the whole
                # list promoted chronologically-extreme but
                # irrelevant sessions over the gold answer.
                # v3: sort key is the EVENT date parsed from the
                # chunk text (first explicit cue; relative cues
                # resolve against the session date), falling back
                # to the session date.
                def _odate(r):
                    sd = _rdate(r)
                    if temporal_v3:
                        return _event_date_of(r.get("text"), sd)
                    return sd
                head, tail = rows[:5], rows[5:]
                dated = [r for r in head if _odate(r) is not None]
                undated = [r for r in head if _odate(r) is None]
                dated.sort(key=_odate, reverse=(direction == "desc"))
                rows = dated + undated + tail
            else:
                dated = [r for r in rows if _rdate(r) is not None]
                undated = [r for r in rows if _rdate(r) is None]
                dated.sort(key=_rdate, reverse=(direction == "desc"))
                rows = dated + undated
    return rows


BASE = datetime(2023, 5, 1)
QUESTIONS = [
    ("what did we discuss 2 weeks ago", "single-session-user"),
    ("what did I buy three days ago", "temporal-reasoning"),
    ("which trip was first, Rome or Paris", "temporal-reasoning"),
    ("which trip was first, Rome or Paris", "single-session-user"),
    ("what is my most recent purchase", "temporal-reasoning"),
    ("in order from earliest to latest, list my trips", "temporal-reasoning"),
    ("in order from latest to earliest, list my trips", "temporal-reasoning"),
    ("what did I do last Saturday", "temporal-reasoning"),
    ("what did I say yesterday", "temporal-reasoning"),
    ("how many days ago did I start", "temporal-reasoning"),
    ("tell me about my cat", "single-session-user"),
]


def _scenario(rng, question, qtype):
    n_sessions = rng.randint(3, 12)
    sids = [f"s{i}" for i in range(n_sessions)]
    dates = []
    for i in range(n_sessions):
        if rng.random() < 0.15:
            dates.append("")  # undated session
        else:
            d = BASE + timedelta(days=rng.randint(0, 90))
            dates.append(d.strftime("%Y/%m/%d (Mon) 10:00"))
    q = {
        "question": question,
        "question_type": qtype,
        "question_date": (BASE + timedelta(days=rng.randint(60, 120))).strftime("%Y/%m/%d (Sun) 09:00"),
        "haystack_session_ids": sids,
        "haystack_dates": dates if rng.random() > 0.1 else [],
    }
    rows = []
    for k in range(rng.randint(1, 12)):
        sid = rng.choice(sids + ["unknown"])
        rows.append({"id": k, "text": f"SID={sid}|on {rng.choice(['May 3', 'last week', 'yesterday', 'nothing'])} we talked"})
    return rows, q


CASES = [(seed, qi) for seed in range(40) for qi in range(len(QUESTIONS))]


@pytest.mark.parametrize("v2,v3", [(False, False), (True, False), (True, True)])
def test_extraction_matches_the_legacy_inline_block(v2, v3):
    checked = 0
    for seed, qi in CASES:
        rng = random.Random(seed * 100 + qi)
        question, qtype = QUESTIONS[qi]
        rows, q = _scenario(rng, question, qtype)
        legacy = _legacy_temporal([dict(r) for r in rows], q, v2 or v3, v3)
        now = H._temporal_rerank([dict(r) for r in rows], q, temporal_v2=v2 or v3, temporal_v3=v3)
        assert [r["id"] for r in now] == [r["id"] for r in legacy], (seed, qi, v2, v3)
        checked += 1
    assert checked == len(CASES)


def test_the_parity_suite_actually_reorders_something():
    """Guard against a vacuous parity test: many cases must change the order."""
    changed = 0
    for seed, qi in CASES:
        rng = random.Random(seed * 100 + qi)
        question, qtype = QUESTIONS[qi]
        rows, q = _scenario(rng, question, qtype)
        out = H._temporal_rerank([dict(r) for r in rows], q)
        changed += [r["id"] for r in out] != [r["id"] for r in rows]
    assert changed >= 30


def _ordinal_case():
    q = {
        "question": "which trip was first, Rome or Paris",
        "question_type": "temporal-reasoning",
        "question_date": "2023/09/01 (Fri) 09:00",
        "haystack_session_ids": ["a", "b", "c"],
        "haystack_dates": ["2023/05/20 (Sat) 10:00", "2023/03/01 (Wed) 10:00", "2023/07/01 (Sat) 10:00"],
    }
    rows = [{"id": 0, "text": "SID=a|x"}, {"id": 1, "text": "SID=b|x"}, {"id": 2, "text": "SID=c|x"}]
    return rows, q


def test_label_routing_on_sorts_ordinal_questions_by_the_gold_label():
    rows, q = _ordinal_case()
    out = H._temporal_rerank(rows, q, label_routing=True)
    assert [r["id"] for r in out] == [1, 0, 2]  # oldest session first


def test_label_routing_off_leaves_ordinal_questions_alone():
    rows, q = _ordinal_case()
    out = H._temporal_rerank(rows, q, label_routing=False)
    assert [r["id"] for r in out] == [0, 1, 2]


def test_label_routing_off_does_not_need_or_use_the_gold_label():
    rows, q = _ordinal_case()
    for qtype in ("temporal-reasoning", "single-session-user", "multi-session"):
        q2 = dict(q, question_type=qtype)
        out = H._temporal_rerank([dict(r) for r in rows], q2, label_routing=False)
        assert [r["id"] for r in out] == [0, 1, 2]


def test_label_routing_off_keeps_the_label_free_relative_target_branch():
    q = {
        "question": "what did I buy 2 weeks ago",
        "question_type": "single-session-user",  # label irrelevant to this branch
        "question_date": "2023/05/30 (Tue) 09:00",
        "haystack_session_ids": ["a", "b"],
        "haystack_dates": ["2023/03/01 (Wed) 10:00", "2023/05/16 (Tue) 10:00"],
    }
    rows = [{"id": 0, "text": "SID=a|x"}, {"id": 1, "text": "SID=b|x"}]
    on = H._temporal_rerank([dict(r) for r in rows], q, label_routing=True)
    off = H._temporal_rerank([dict(r) for r in rows], q, label_routing=False)
    assert [r["id"] for r in on] == [r["id"] for r in off] == [1, 0]


def test_cli_flag_exists_and_defaults_to_historical_behaviour():
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, str(Path(H.__file__)), "--help"], capture_output=True, text=True
    ).stdout
    assert "--temporal-no-label-routing" in out


# ── offline ablation tool ─────────────────────────────────────────────────────

_ABL = importlib.util.spec_from_file_location(
    "temporal_ablation_offline",
    Path(__file__).resolve().parent.parent / "benchmarks" / "temporal_ablation_offline.py",
)
A = importlib.util.module_from_spec(_ABL)
_ABL.loader.exec_module(A)


def _dump_and_dataset():
    """Two questions: an ordinal one the label rescues, a relative one that is label-free."""
    ordinal_rows = [
        {"id": 1, "sid": "a", "text": "SID=a|x"},
        {"id": 2, "sid": "b", "text": "SID=b|x"},
        {"id": 3, "sid": "c", "text": "SID=c|x"},
    ]
    rel_rows = [
        {"id": 1, "sid": "a", "text": "SID=a|x"},
        {"id": 2, "sid": "b", "text": "SID=b|x"},
    ]

    def cand(qid, qtype, answer, rows):
        return {
            "schema": A.SCHEMA,
            "stage": A.STAGE,
            "qid": qid,
            "qtype": qtype,
            "answer_sids": answer,
            "rows": rows,
            "gate_top_id": None,
            "gate_info": None,
            "trust_gate_pin_margin": 0.5,
            "temporal_v2": True,
            "temporal_v3": False,
        }

    cands = [
        cand("ord", "temporal-reasoning", ["b"], ordinal_rows),
        cand("rel", "single-session-user", ["b"], rel_rows),
        cand("ghost", "x", ["a"], rel_rows),
    ]
    data = {
        "ord": {
            "question_id": "ord", "question_type": "temporal-reasoning",
            "question": "which trip was first, Rome or Paris",
            "question_date": "2023/09/01 (Fri) 09:00",
            "haystack_session_ids": ["a", "b", "c"],
            "haystack_dates": ["2023/05/20 (Sat) 10:00", "2023/03/01 (Wed) 10:00",
                               "2023/07/01 (Sat) 10:00"],
        },
        "rel": {
            "question_id": "rel", "question_type": "single-session-user",
            "question": "what did I buy 2 weeks ago",
            "question_date": "2023/05/30 (Tue) 09:00",
            "haystack_session_ids": ["a", "b"],
            "haystack_dates": ["2023/03/01 (Wed) 10:00", "2023/05/16 (Tue) 10:00"],
        },
    }
    return cands, data


def test_ablation_replays_high_confidence_gate_pin_after_temporal():
    cands, data = _dump_and_dataset()
    c = cands[0]
    # Gate chose row a at #1 before temporal; ordinal temporal would put b first.
    c["gate_top_id"] = 1
    c["gate_info"] = {"fired": True, "ftce_margin": 0.9}
    labeled = A.apply_variant("on_labeled", c, data["ord"])
    no_label = A.apply_variant("on_no_label", c, data["ord"])
    assert labeled[0]["id"] == 1  # temporal move is pinned back by confident gate
    assert no_label[0]["id"] == 1

    c["gate_info"] = {"fired": True, "ftce_margin": 0.2}
    labeled = A.apply_variant("on_labeled", c, data["ord"])
    assert labeled[0]["id"] == 2  # below pin threshold: temporal win survives


def test_ablation_rejects_pre_gate_training_dump_contract():
    cands, _ = _dump_and_dataset()
    cands[0]["stage"] = "post_ce_pre_gate"
    with pytest.raises(ValueError, match="dump-temporal-candidates"):
        A.validate_candidates(cands, 3)

def test_ablation_separates_label_routing_from_the_label_free_branch():
    cands, data = _dump_and_dataset()
    res = A.evaluate_variants(cands, data)
    assert res["missing"] == 1  # 'ghost' is not in the dataset: skipped, counted
    s = A.summarize(res["per_q"], res["qtype"])
    # off: neither question has its gold at #1 -> R@1 0
    assert s["off"]["R@1"] == 0.0
    # on_labeled: the label rescues 'ord' AND the relative cue rescues 'rel'
    assert s["on_labeled"]["R@1"] == 1.0
    # on_no_label: only the label-free 'rel' is rescued
    assert s["on_no_label"]["R@1"] == 0.5
    assert s["on_no_label"]["by_type"]["single-session-user"]["R@1"] == 1.0
    assert s["on_no_label"]["by_type"]["temporal-reasoning"]["R@1"] == 0.0

    cmp_ = A.compare(res["per_q"], k=1, iters=200, seed=1)
    assert cmp_["on_labeled_vs_off"]["helped"] == 2
    assert cmp_["on_no_label_vs_off"]["helped"] == 1
    assert cmp_["on_labeled_vs_on_no_label"]["helped"] == 1  # the label-only gain
    assert cmp_["on_labeled_vs_on_no_label"]["hurt"] == 0


def test_paired_bootstrap_degenerate_and_nondegenerate():
    same = A.paired_bootstrap([1, 0, 1], [1, 0, 1])
    assert same["delta"] == 0.0 and same["ci95"] == [0.0, 0.0]
    assert A.paired_bootstrap([], [])["n"] == 0
    win = A.paired_bootstrap([1] * 40 + [0] * 10, [0] * 40 + [0] * 10, iters=500, seed=3)
    assert win["helped"] == 40 and win["hurt"] == 0
    assert win["ci95"][0] > 0
    mixed = A.paired_bootstrap([1, 0] * 10, [0, 1] * 10, iters=500, seed=3)
    assert mixed["delta"] == 0.0 and mixed["ci95"][0] < 0 < mixed["ci95"][1]


def test_ablation_cli_end_to_end(tmp_path):
    import json
    import subprocess
    import sys
    cands, data = _dump_and_dataset()
    (tmp_path / "c.json").write_text(json.dumps(cands))
    (tmp_path / "d.json").write_text(json.dumps(list(data.values())))
    out = tmp_path / "o.json"
    run = subprocess.run(
        [sys.executable, str(Path(A.__file__)), "--candidates", str(tmp_path / "c.json"),
         "--data", str(tmp_path / "d.json"), "--out", str(out), "--iters", "100",
         "--expected-n", "3"],
        capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    res = json.loads(out.read_text())
    assert res["missing_from_dataset"] == 1
    assert res["variants"]["on_labeled"]["R@1"] == 1.0
    assert "on_labeled_vs_on_no_label" in res["paired_R@1"]
    assert "on_no_label" in run.stdout


def test_ablation_cli_rejects_wrong_candidate_count(tmp_path):
    import json
    import subprocess
    import sys

    cands, data = _dump_and_dataset()
    (tmp_path / "c.json").write_text(json.dumps(cands))
    (tmp_path / "d.json").write_text(json.dumps(list(data.values())))
    run = subprocess.run(
        [
            sys.executable,
            str(Path(A.__file__)),
            "--candidates",
            str(tmp_path / "c.json"),
            "--data",
            str(tmp_path / "d.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert "candidate count mismatch: expected 500, got 3" in run.stderr


def test_query_only_ordinal_routing_works_without_gold_question_type():
    rows, q = _ordinal_case()
    q = dict(q, question="What is the order of my trips from earliest to latest?",
             question_type="single-session-user")
    out = H._temporal_rerank(
        [dict(r) for r in rows], q,
        temporal_v2=True,
        label_routing=False,
        query_ordinal_routing=True,
    )
    assert [r["id"] for r in out] == [1, 0, 2]


def test_query_only_ordinal_router_is_narrow_about_ambiguous_first_last_words():
    assert H._detect_query_ordinal("What is the order of the concerts I attended?") == "asc"
    assert H._detect_query_ordinal("List them from latest to earliest") == "desc"
    assert H._detect_query_ordinal("What was my first purchase at the store?") is None
    assert H._detect_query_ordinal("What is my friend's last name?") is None


def test_temporal_entity_anchor_promotes_unique_named_entity_match():
    q = {"question": "How many days ago did I meet Emma?"}
    rows = [
        {"id": 0, "text": "SID=a|[user] I met Sophia at a networking event."},
        {"id": 1, "text": "SID=b|[user] I caught up with Emma over lunch today."},
        {"id": 2, "text": "SID=c|[user] I went to lunch downtown."},
    ]
    out = H._temporal_entity_anchor_rerank([dict(r) for r in rows], q)
    assert [r["id"] for r in out] == [1, 0, 2]


def test_temporal_entity_anchor_does_not_fire_without_temporal_fact_cue():
    q = {"question": "Can you recommend a restaurant for Emma?"}
    rows = [
        {"id": 0, "text": "SID=a|[user] I like Italian food."},
        {"id": 1, "text": "SID=b|[user] Emma likes sushi."},
    ]
    out = H._temporal_entity_anchor_rerank([dict(r) for r in rows], q)
    assert [r["id"] for r in out] == [0, 1]


def test_query_only_temporal_cli_flags_exist():
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, str(Path(H.__file__)), "--help"], capture_output=True, text=True
    ).stdout
    assert "--temporal-query-ordinal" in out
    assert "--temporal-entity-anchor" in out
