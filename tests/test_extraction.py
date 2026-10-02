"""Tests for rentml.extraction and rentml.extraction_llm (no network, fake Anthropic client)."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rentml import extraction_llm
from rentml.extraction import (
    ATTRIBUTE_FIELDS,
    ATTRIBUTE_SCHEMA,
    VIEWS,
    ExtractedAttributes,
    ExtractionError,
    evaluate_extraction,
    extract_frame,
    rule_based_extract,
)
from rentml.extraction_llm import (
    FALLBACK_BETA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TOOL_NAME,
    ClaudeExtractor,
    ExtractionCache,
    build_prompt,
)

# --- schema, prompt, dataclass ------------------------------------------------------------------


def test_schema_matches_dataclass_and_is_strict_compatible() -> None:
    names = [f.name for f in dataclasses.fields(ExtractedAttributes)]
    assert names == [*ATTRIBUTE_FIELDS, "evidence", "extractor"]
    props = ATTRIBUTE_SCHEMA["properties"]
    assert set(props) == {*ATTRIBUTE_FIELDS, "evidence"}  # type: ignore[arg-type]
    assert ATTRIBUTE_SCHEMA["additionalProperties"] is False
    assert set(ATTRIBUTE_SCHEMA["required"]) == set(props)  # type: ignore[arg-type]
    assert props["evidence"]["additionalProperties"] is False  # type: ignore[index]
    assert props["view"]["enum"] == list(VIEWS)  # type: ignore[index]
    assert "minimum" not in json.dumps(ATTRIBUTE_SCHEMA)  # unsupported by strict tools
    assert len(ATTRIBUTE_FIELDS) == 17


def test_build_prompt_wraps_text_and_system_prompt_names_tool() -> None:
    prompt = build_prompt("3.5 Zimmer mit Balkon")
    assert "<description>\n3.5 Zimmer mit Balkon\n</description>" in prompt
    assert TOOL_NAME in SYSTEM_PROMPT
    assert build_prompt("").endswith("<description>\n\n</description>")


def test_extracted_attributes_roundtrip_through_dict() -> None:
    attrs = ExtractedAttributes(floor=2, rooms=3.5, view="lake", evidence={"floor": "2. OG"})
    attrs.extractor = "rules"
    assert ExtractedAttributes.from_dict(json.loads(json.dumps(attrs.to_dict()))) == attrs


def test_from_dict_coerces_messy_values_and_drops_unknown_keys() -> None:
    data = {
        "floor": "2",
        "rooms": "3,5",
        "area_sqm": float("nan"),
        "renovation_year": 2019.0,
        "has_lift": "true",
        "is_furnished": 0,
        "has_garden": "maybe",
        "view": "LAKE",
        "parking": "bogus",
        "unknown_field": 1,
        "evidence": {"floor": " 2. OG ", "rooms": None, "unknown": "x"},
    }
    attrs = ExtractedAttributes.from_dict(data, extractor="test")
    assert (attrs.floor, attrs.rooms, attrs.area_sqm, attrs.renovation_year) == (2, 3.5, None, 2019)
    assert (attrs.has_lift, attrs.is_furnished, attrs.has_garden) == (True, False, None)
    assert (attrs.view, attrs.parking, attrs.rent_regime) == ("lake", "none", "market")
    assert attrs.evidence == {"floor": "2. OG"}
    assert attrs.extractor == "test"


def test_from_dict_accepts_numpy_bools_and_label_words() -> None:
    data = {"has_lift": np.True_, "is_furnished": np.False_, "has_garden": "ja", "rooms": np.True_}
    attrs = ExtractedAttributes.from_dict(data)
    assert (attrs.has_lift, attrs.is_furnished, attrs.has_garden) == (True, False, True)
    assert attrs.rooms is None


# --- ExtractionCache ----------------------------------------------------------------------------


def test_cache_roundtrip_persists_and_keys_by_model(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "cache.jsonl"
    cache = ExtractionCache(path)
    assert cache.get("text", "m1") is None and len(cache) == 0
    attrs = ExtractedAttributes(rooms=3.5, evidence={"rooms": "3.5 Zimmer"}, extractor="x")
    cache.put("text", "m1", attrs)
    assert cache.get("text", "m1") == attrs
    assert cache.get("text", "m2") is None
    reopened = ExtractionCache(path)
    assert reopened.get("text", "m1") == attrs and len(reopened) == 1
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["prompt_version"] == PROMPT_VERSION and record["model"] == "m1"


def test_cache_key_changes_with_prompt_version(monkeypatch: pytest.MonkeyPatch) -> None:
    before = ExtractionCache.key("text", "m1")
    monkeypatch.setattr(extraction_llm, "PROMPT_VERSION", "other-version")
    assert ExtractionCache.key("text", "m1") != before


def test_cache_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    good = {"key": ExtractionCache.key("t", "m"), "attributes": {"rooms": 2.0}}
    path.write_text("not json\n\n" + json.dumps({"no": "key"}) + "\n" + json.dumps(good) + "\n")
    cache = ExtractionCache(path)
    assert len(cache) == 1
    assert cache.get("t", "m").rooms == 2.0  # type: ignore[union-attr]


# --- ClaudeExtractor (fake client) --------------------------------------------------------------


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {name: None for name in ATTRIBUTE_FIELDS}
    payload.update(view="none", parking="none", rent_regime="market")
    payload["evidence"] = {name: None for name in ATTRIBUTE_FIELDS}
    payload.update(overrides)
    return payload


def _message(payload: dict[str, object] | None, stop_reason: str = "tool_use") -> SimpleNamespace:
    blocks = [SimpleNamespace(type="thinking", thinking="")]
    if payload is not None:
        blocks.append(SimpleNamespace(type="tool_use", name=TOOL_NAME, input=payload))
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


class _FakeEndpoint:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.response


class _FakeBatches:
    def __init__(self, results: list[SimpleNamespace]) -> None:
        self.created: list[list[dict[str, object]]] = []
        self._results = results

    def create(self, requests: list[dict[str, object]]) -> SimpleNamespace:
        self.created.append(requests)
        return SimpleNamespace(id="msgbatch_1")

    def results(self, batch_id: str) -> list[SimpleNamespace]:
        assert batch_id == "msgbatch_1"
        return self._results


def _client(
    response: SimpleNamespace, results: list[SimpleNamespace] | None = None
) -> SimpleNamespace:
    messages = _FakeEndpoint(response)
    messages.batches = _FakeBatches(results or [])  # type: ignore[attr-defined]
    return SimpleNamespace(
        messages=messages, beta=SimpleNamespace(messages=_FakeEndpoint(response))
    )


def test_claude_extractor_forces_strict_tool_and_anonymises_input(tmp_path: Path) -> None:
    payload = _payload(rooms=3.5, view="lake", evidence={"rooms": "3.5 Zimmer", "view": "Seesicht"})
    client = _client(_message(payload))
    extractor = ClaudeExtractor(cache=ExtractionCache(tmp_path / "c.jsonl"), client=client)
    attrs = extractor.extract("3.5 Zimmer, Seesicht, Miete CHF 2'100.-, Tel. 079 123 45 67")
    assert attrs.rooms == 3.5 and attrs.view == "lake" and attrs.extractor == "claude:claude-opus-5"
    assert attrs.evidence == {"rooms": "3.5 Zimmer", "view": "Seesicht"}
    assert client.messages.calls == []  # Opus 5 goes through the beta endpoint with fallbacks
    call = client.beta.messages.calls[0]
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert call["tools"][0]["strict"] is True  # type: ignore[index]
    assert call["output_config"] == {"effort": "low"}
    sent = call["messages"][0]["content"]  # type: ignore[index]
    assert "[PRICE]" in sent and "[PHONE]" in sent and "2'100" not in sent and "079" not in sent
    extractor.extract("3.5 Zimmer, Seesicht, Miete CHF 2'100.-, Tel. 079 123 45 67")
    assert len(client.beta.messages.calls) == 1  # second call served from the cache


def test_claude_extractor_model_specific_request_options() -> None:
    client = _client(_message(_payload()))
    ClaudeExtractor("claude-opus-5-5", client=client).extract("Balkon")
    call = client.messages.calls[-1]
    assert call["tool_choice"] == {"type": "auto"} and "fallbacks" not in call
    ClaudeExtractor("claude-haiku-4-5", client=client).extract("Balkon")
    assert "output_config" not in client.messages.calls[-1]


@pytest.mark.parametrize(
    "message",
    [_message(None, stop_reason="refusal"), _message(_payload(), "max_tokens"), _message(None)],
)
def test_claude_extractor_raises_without_usable_tool_call(message: SimpleNamespace) -> None:
    extractor = ClaudeExtractor(client=_client(message), use_fallbacks=False)
    with pytest.raises(ExtractionError):
        extractor.extract("3 Zimmer")


def test_claude_extractor_skips_empty_text_without_api_call() -> None:
    client = _client(_message(_payload()))
    attrs = ClaudeExtractor(client=client)(None)  # type: ignore[arg-type]
    assert attrs == ExtractedAttributes(extractor="claude:claude-opus-5")
    assert client.beta.messages.calls == [] and client.messages.calls == []


def test_claude_extractor_creates_default_client_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    sentinel = object()
    monkeypatch.setattr(anthropic, "Anthropic", lambda: sentinel)
    extractor = ClaudeExtractor()
    assert extractor._client is None
    assert extractor.client is sentinel


def test_batch_requests_builds_params_and_skips_cached_or_empty(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "c.jsonl")
    extractor = ClaudeExtractor(cache=cache, client=_client(_message(_payload())))
    cache.put("cached text", extractor.model, ExtractedAttributes())
    texts = pd.Series(["Miete CHF 1850, Balkon", "", "cached text", None], index=[10, 11, 12, 13])
    requests = extractor.batch_requests(texts)
    assert [r["custom_id"] for r in requests] == ["row-10"]
    params = requests[0]["params"]
    assert "fallbacks" not in params and "betas" not in params  # not allowed in batches
    assert "[PRICE]" in params["messages"][0]["content"]  # type: ignore[index]
    assert len(extractor.batch_requests(texts, skip_cached=False)) == 2
    with pytest.raises(ValueError):
        extractor.batch_requests(["a", "b"], custom_ids=["x", "x"])


def test_submit_and_collect_batch_write_cache(tmp_path: Path) -> None:
    ok = SimpleNamespace(type="succeeded", message=_message(_payload(rooms=2.0)))
    results = [
        SimpleNamespace(custom_id="row-1", result=ok),
        SimpleNamespace(custom_id="row-2", result=SimpleNamespace(type="errored")),
        SimpleNamespace(
            custom_id="row-3",
            result=SimpleNamespace(type="succeeded", message=_message(None, "refusal")),
        ),
    ]
    client = _client(_message(_payload()), results)
    cache = ExtractionCache(tmp_path / "c.jsonl")
    extractor = ClaudeExtractor(cache=cache, client=client)
    texts = {"row-1": "2 Zimmer", "row-2": "b", "row-3": "c"}
    requests = extractor.batch_requests(list(texts.values()), custom_ids=list(texts))
    assert extractor.submit_batch(requests) == "msgbatch_1"
    out = extractor.collect_batch("msgbatch_1", texts)
    assert list(out) == ["row-1"] and out["row-1"].rooms == 2.0
    assert cache.get("2 Zimmer", extractor.model) == out["row-1"]


def test_request_params_serialise_through_the_real_sdk() -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    seen: list[dict[str, object]] = []

    def handler(request: object) -> object:
        body = json.loads(request.content)  # type: ignore[attr-defined]
        seen.append(body)
        block = {"type": "tool_use", "id": "toolu_1", "name": TOOL_NAME, "input": _payload()}
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
            "content": [block], "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })  # fmt: skip

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    client = anthropic.Anthropic(api_key="test-key", http_client=http, max_retries=0)
    attrs = ClaudeExtractor(client=client).extract("3 Zimmer")
    assert attrs.extractor == "claude:claude-opus-5"
    assert seen[0]["fallbacks"] == "default" and seen[0]["tools"][0]["name"] == TOOL_NAME


def test_lazy_reexports_from_extraction_module() -> None:
    from rentml import extraction

    assert extraction.ClaudeExtractor is ClaudeExtractor
    assert extraction.PROMPT_VERSION == PROMPT_VERSION
    with pytest.raises(AttributeError):
        _ = extraction.does_not_exist


# --- extract_frame ------------------------------------------------------------------------------


def test_extract_frame_columns_dtypes_and_deduplication() -> None:
    calls: list[str] = []

    def counting(text: str) -> ExtractedAttributes:
        calls.append(text)
        return rule_based_extract(text)

    texts = pd.Series(
        ["3.5 Zimmer, Balkon", "3.5 Zimmer, Balkon", None, "WG-Zimmer"], index=[5, 6, 7, 8]
    )
    frame = extract_frame(texts, counting, include_evidence=True)
    assert sorted(calls) == ["", "3.5 Zimmer, Balkon", "WG-Zimmer"]
    assert list(frame.index) == [5, 6, 7, 8]
    assert {f"attr_{n}" for n in ATTRIBUTE_FIELDS if n != "rent_regime"} <= set(frame.columns)
    assert frame["attr_has_balcony_or_terrace"].dtype == float
    assert frame["attr_view"].dtype == "category"
    assert frame.loc[5, "attr_rooms"] == 3.5 and np.isnan(frame.loc[7, "attr_rooms"])
    assert frame.loc[8, "rent_regime"] == "shared_flat" and frame.loc[5, "extractor"] == "rules"
    assert json.loads(frame.loc[5, "evidence"])["rooms"] == "3.5 Zimmer"


def test_extract_frame_error_handling() -> None:
    def failing(text: str) -> ExtractedAttributes:
        if text == "bad":
            raise ExtractionError("refusal")
        return rule_based_extract(text)

    texts = pd.Series(["2 Zimmer", "bad"])
    frame = extract_frame(texts, failing, errors="ignore")
    assert frame.loc[0, "attr_rooms"] == 2.0 and np.isnan(frame.loc[1, "attr_rooms"])
    with pytest.raises(ExtractionError):
        extract_frame(texts, failing)
    with pytest.raises(ValueError):
        extract_frame(texts, failing, errors="skip")


# --- evaluate_extraction ------------------------------------------------------------------------


def test_evaluate_extraction_counts_and_groups() -> None:
    idx = [1, 2, 3, 4]
    pred = pd.DataFrame(
        {"attr_floor": [2.0, 3.0, np.nan, 0.0], "attr_area_sqm": [85.4, 70.0, np.nan, np.nan],
         "rent_regime": ["market", "cooperative", "shared_flat", "market"]},
        index=idx,
    )  # fmt: skip
    gold = pd.DataFrame(
        {"floor": [2, 1, 4, None], "area_sqm": [85.0, 70.0, 60.0, None],
         "rent_regime": ["market", "cooperative", "market", "market"]},
        index=idx,
    )  # fmt: skip
    table = evaluate_extraction(pred, gold, ["floor", "area_sqm", "rent_regime"]).set_index("field")
    floor = table.loc["floor"]
    assert (floor.tp, floor.fp, floor.fn, floor.support, floor.n_pred) == (1, 2, 2, 3, 3)
    assert floor.precision == pytest.approx(1 / 3) and floor.recall == pytest.approx(1 / 3)
    assert table.loc["area_sqm", "tp"] == 2  # 85.4 vs 85.0 within 1 m²
    regime = table.loc["rent_regime"]
    assert (regime.tp, regime.fp, regime.fn, regime.accuracy) == (1, 1, 0, 0.75)
    lang = pd.Series(["de", "de", "fr", "fr"], index=idx, name="lang")
    grouped = evaluate_extraction(pred, gold, ["floor"], by=lang)
    assert list(grouped["lang"]) == ["de", "fr"] and list(grouped["tp"]) == [1, 0]


def test_evaluate_extraction_rejects_unknown_fields_and_missing_columns() -> None:
    frame = pd.DataFrame({"attr_floor": [1.0]})
    with pytest.raises(KeyError):
        evaluate_extraction(frame, frame, ["nonsense"])
    with pytest.raises(KeyError):
        evaluate_extraction(frame, frame, ["rooms"])
    with pytest.raises(ValueError, match="at least one"):
        evaluate_extraction(frame, frame, [])


def test_evaluate_extraction_reads_multilingual_label_words() -> None:
    pred = pd.DataFrame(
        {"attr_has_lift": [1.0, 0.0, np.nan, 1.0, 0.0], "attr_rooms": [3.5, 2.0, 1.0, np.nan, 4.0]}
    )
    gold = pd.DataFrame(
        {"has_lift": ["ja", "Nein", "", "oui", np.False_], "rooms": ["3,5", 2, "n/a", None, "4"]}
    )
    table = evaluate_extraction(pred, gold, ["has_lift", "rooms"]).set_index("field")
    assert table.loc["has_lift", "support"] == 4 and table.loc["has_lift", "tp"] == 4
    assert table.loc["rooms", "support"] == 3 and table.loc["rooms", "fp"] == 1
    assert table.loc["has_lift", "recall"] == 1.0


@pytest.mark.parametrize(
    ("column", "values"),
    [("has_lift", ["ja", "vielleicht"]), ("rooms", ["3", "drei"]), ("view", ["lake", "Seeblick"])],
)
def test_evaluate_extraction_rejects_unreadable_labels(column: str, values: list[str]) -> None:
    pred = pd.DataFrame({column: values[:1] * 2})
    gold = pd.DataFrame({column: values})
    with pytest.raises(ValueError, match=f"gold column '{column}'"):
        evaluate_extraction(pred, gold, [column])
