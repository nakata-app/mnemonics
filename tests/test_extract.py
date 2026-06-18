"""FactExtractor birim testleri — sahte istemciyle, API'siz."""
from __future__ import annotations

import json

import pytest

from mnemonics.extract import FactExtractor, _parse_json_array


class FakeClient:
    """OpenAI-compatible sahte istemci: sirayla verilen cevaplari dondurur."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        chat = self

        class _Completions:
            def create(_self, model, messages, max_tokens, temperature):
                chat.prompts.append(messages[0]["content"])
                content = chat._responses.pop(0)

                class _Msg:
                    pass

                class _Choice:
                    message = _Msg()

                _Choice.message.content = content

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


TURNS = [
    {"id": "D1:1", "speaker": "Caroline", "text": "I went to an LGBTQ support group yesterday."},
    {"id": "D1:2", "speaker": "Melanie", "text": "That's great! I painted a sunrise last week."},
]


def _ok_json(facts):
    return json.dumps(facts)


def test_happy_path_with_provenance():
    fake = FakeClient([_ok_json([
        {"fact": "Caroline attended an LGBTQ support group on 7 May 2023.",
         "sources": ["D1:1"]},
        {"fact": "Melanie painted a sunrise in early May 2023.",
         "sources": ["D1:2"]},
    ])])
    ex = FactExtractor(client=fake)
    facts = ex.extract_session(TURNS, session_date="8 May 2023")
    assert len(facts) == 2
    assert facts[0]["source_ids"] == ["D1:1"]
    assert facts[0]["kind"] == "fact"
    assert ex.stats["calls"] == 1 and ex.stats["facts"] == 2
    # prompt turn id'leri ve session tarihini icermeli
    assert "[D1:1]" in fake.prompts[0] and "8 May 2023" in fake.prompts[0]


def test_fenced_json_is_parsed():
    fake = FakeClient(["```json\n[{\"fact\": \"F.\", \"sources\": [\"D1:1\"]}]\n```"])
    facts = FactExtractor(client=fake).extract_session(TURNS)
    assert len(facts) == 1


def test_malformed_json_returns_empty_and_counts():
    fake = FakeClient(["bu json degil"])
    ex = FactExtractor(client=fake)
    assert ex.extract_session(TURNS) == []
    assert ex.stats["parse_failures"] == 1


def test_hallucinated_sources_dropped():
    fake = FakeClient([_ok_json([
        {"fact": "Gercek.", "sources": ["D1:2"]},
        {"fact": "Uydurma kaynak.", "sources": ["D9:99"]},
        {"fact": "Karisik kaynak.", "sources": ["D9:99", "D1:1"]},
    ])])
    ex = FactExtractor(client=fake)
    facts = ex.extract_session(TURNS)
    texts = [f["text"] for f in facts]
    assert "Uydurma kaynak." not in texts          # tamamen uydurma -> dustu
    assert any(f["source_ids"] == ["D1:1"] for f in facts)  # bilinmeyen id suzuldu
    assert ex.stats["dropped_bad_sources"] == 1


def test_call_budget_enforced():
    fake = FakeClient([_ok_json([]), _ok_json([])])
    ex = FactExtractor(client=fake, max_calls=1)
    ex.extract_session(TURNS)
    with pytest.raises(RuntimeError, match="budget"):
        ex.extract_session(TURNS)


def test_max_facts_cap():
    many = [{"fact": f"F{i}.", "sources": ["D1:1"]} for i in range(40)]
    fake = FakeClient([_ok_json(many)])
    ex = FactExtractor(client=fake, max_facts=5)
    assert len(ex.extract_session(TURNS)) == 5


def test_parse_json_array_variants():
    assert _parse_json_array("") == []
    assert _parse_json_array("prose [1, 2] trailing") == [1, 2]
    assert _parse_json_array("{\"a\": 1}") == []
    assert _parse_json_array("[{\"fact\": \"x\"}]") == [{"fact": "x"}]
    # s < 0 (no '[') → returns []
    assert _parse_json_array("no brackets here") == []
    # s >= 0, e <= s (no ']') → returns []
    assert _parse_json_array("[no close bracket") == []
    # Both brackets present but invalid JSON inside → JSONDecodeError path
    assert _parse_json_array("[broken json]") == []
    # Non-list JSON (dict at top level) → returns []
    assert _parse_json_array("{\"k\": \"v\"}") == []
