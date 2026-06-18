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


# ── edge cases ────────────────────────────────────────────────────────────────

class _ErrorThenOkClient:
    """First call raises, second call returns valid JSON."""
    def __init__(self, raises, then_returns):
        self._raises = raises
        self._then = then_returns
        self._calls = 0
        chat = self

        class _Completions:
            def create(_self, **kw):
                chat._calls += 1
                if chat._calls == 1:
                    raise chat._raises
                content = chat._then

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


def test_no_valid_ids_returns_empty():
    fake = FakeClient(["[]"])
    ex = FactExtractor(client=fake)
    assert ex.extract_session([{"speaker": "A", "text": "hello"}]) == []


def test_empty_llm_response_counts_empty_session():
    fake = FakeClient([""])  # empty string → raw.strip() is falsy
    ex = FactExtractor(client=fake)
    result = ex.extract_session(TURNS)
    assert result == []
    assert ex.stats["empty_sessions"] == 1


def test_retry_on_exception_succeeds():
    valid_json = json.dumps([{"fact": "A fact.", "sources": ["D1:1"]}])
    client = _ErrorThenOkClient(raises=OSError("timeout"), then_returns=valid_json)
    ex = FactExtractor(client=client, retries=1)
    facts = ex.extract_session(TURNS)
    assert len(facts) == 1


def test_non_dict_items_in_array_skipped():
    raw = json.dumps(["not a dict", {"fact": "Real.", "sources": ["D1:1"]}])
    fake = FakeClient([raw])
    ex = FactExtractor(client=fake)
    facts = ex.extract_session(TURNS)
    assert len(facts) == 1  # only the dict item survives


def test_item_without_text_skipped():
    raw = json.dumps([{"fact": "", "sources": ["D1:1"]}])
    fake = FakeClient([raw])
    ex = FactExtractor(client=fake)
    facts = ex.extract_session(TURNS)
    assert facts == []
    assert ex.stats["empty_sessions"] == 1


def test_extract_many():
    valid = json.dumps([{"fact": "A fact.", "sources": ["D1:1"]}])
    fake = FakeClient([valid, "[]"])
    ex = FactExtractor(client=fake)
    results = ex.extract_many([(TURNS, "2025-01-01"), (TURNS, "")])
    assert len(results) == 2
    assert len(results[0]) == 1
    assert results[1] == []


# ── _default_client ───────────────────────────────────────────────────────────

def test_default_client_deepseek_key(monkeypatch, tmp_path):
    from unittest.mock import MagicMock as _MM
    import sys
    mock_openai_mod = _MM()
    mock_cls = _MM(return_value=_MM())
    mock_openai_mod.OpenAI = mock_cls
    monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from mnemonics.extract import _default_client
    _default_client()
    _, kwargs = mock_cls.call_args
    assert kwargs["api_key"] == "sk-test-deepseek"
    assert "deepseek" in kwargs["base_url"]


def test_default_client_openai_key(monkeypatch):
    from unittest.mock import MagicMock as _MM
    import sys
    mock_openai_mod = _MM()
    mock_cls = _MM(return_value=_MM())
    mock_openai_mod.OpenAI = mock_cls
    monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    from mnemonics.extract import _default_client
    _default_client()
    _, kwargs = mock_cls.call_args
    assert kwargs["api_key"] == "sk-openai"


def test_default_client_no_key_raises(monkeypatch):
    from unittest.mock import MagicMock as _MM
    import sys
    mock_openai_mod = _MM()
    mock_openai_mod.OpenAI = _MM(return_value=_MM())
    monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from mnemonics.extract import _default_client
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        _default_client()


def test_retry_exhausted_raises():
    """When all retries fail the exception propagates."""
    from unittest.mock import MagicMock as _MM
    client = _MM()
    client.chat.completions.create.side_effect = OSError("network down")
    ex = FactExtractor(client=client, retries=1)
    with pytest.raises(OSError):
        ex.extract_session(TURNS)


def test_client_property_lazy_init(monkeypatch):
    """extract.py line 101: client property calls _default_client() when None."""
    import sys
    from unittest.mock import MagicMock as _MM
    mock_openai_mod = _MM()
    mock_client = _MM()
    mock_openai_mod.OpenAI = _MM(return_value=mock_client)
    monkeypatch.setitem(sys.modules, "openai", mock_openai_mod)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    ex = FactExtractor()   # no client passed → _client = None
    assert ex._client is None
    c = ex.client          # triggers lazy init
    assert c is not None
    assert ex._client is not None
