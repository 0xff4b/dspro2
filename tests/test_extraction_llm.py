"""Tests for rentml.extraction_llm: prompt version, retry, batch dedup and split (no network)."""

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rentml.extraction import ATTRIBUTE_FIELDS, ExtractedAttributes, ExtractionError
from rentml.extraction_llm import (
    ATTRIBUTE_TOOL,
    BATCH_MAX_REQUESTS,
    PROMPT_LABEL,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TOOL_NAME,
    ClaudeExtractor,
    ExtractionCache,
    MissingToolCallError,
    build_prompt,
    prompt_fingerprint,
)

DUPLICATES = pd.Series(
    [
        "3 Zimmer mit Balkon, Tel. 079 123 45 67",
        "Attika mit Seesicht",
        "3 Zimmer mit Balkon, Tel. 078 765 43 21",  # same text once the phone is anonymised
        "  Attika mit Seesicht  ",
    ],
    index=[1, 2, 3, 4],
)


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {name: None for name in ATTRIBUTE_FIELDS}
    payload.update(view="none", parking="none", rent_regime="market")
    payload["evidence"] = {name: None for name in ATTRIBUTE_FIELDS}
    payload.update(overrides)
    return payload


def _tool_message(**overrides: object) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name=TOOL_NAME, input=_payload(**overrides))
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def _text_message(stop_reason: str = "end_turn") -> SimpleNamespace:
    block = SimpleNamespace(type="text", text="The flat has 3 rooms.")
    return SimpleNamespace(stop_reason=stop_reason, content=[block])


class _Endpoint:
    """Fake ``messages`` endpoint returning the given responses in order (the last one repeats)."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responses[min(len(self.calls), len(self.responses)) - 1]


class _Batches:
    def __init__(
        self, results: dict[str, list[SimpleNamespace]] | None = None, fail_on: int = 0
    ) -> None:
        self.created: list[list[dict[str, object]]] = []
        self.results_by_id = results or {}
        self.fail_on = fail_on

    def create(self, requests: list[dict[str, object]]) -> SimpleNamespace:
        if len(self.created) + 1 == self.fail_on:
            raise RuntimeError("HTTP 500")
        self.created.append(list(requests))
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}")

    def results(self, batch_id: str) -> list[SimpleNamespace]:
        return self.results_by_id[batch_id]


def _client(
    responses: list[SimpleNamespace] | None = None, batches: _Batches | None = None
) -> SimpleNamespace:
    messages = _Endpoint(responses or [_tool_message()])
    messages.batches = batches or _Batches()  # type: ignore[attr-defined]
    beta = SimpleNamespace(messages=_Endpoint(responses or [_tool_message()]))
    return SimpleNamespace(messages=messages, beta=beta)


def _succeeded(custom_id: str, **overrides: object) -> SimpleNamespace:
    result = SimpleNamespace(type="succeeded", message=_tool_message(**overrides))
    return SimpleNamespace(custom_id=custom_id, result=result)


# --- PROMPT_VERSION -----------------------------------------------------------------------------


def test_prompt_version_is_derived_from_current_prompt_tool_and_schema() -> None:
    fingerprint = prompt_fingerprint(SYSTEM_PROMPT, ATTRIBUTE_TOOL, build_prompt("{text}"))
    assert f"{PROMPT_LABEL}+{fingerprint}" == PROMPT_VERSION
    assert len(fingerprint) == 12 and int(fingerprint, 16) >= 0


def test_prompt_fingerprint_changes_with_prompt_schema_or_user_turn() -> None:
    user = build_prompt("{text}")
    base = prompt_fingerprint(SYSTEM_PROMPT, ATTRIBUTE_TOOL, user)
    reordered = dict(reversed(list(ATTRIBUTE_TOOL.items())))
    assert prompt_fingerprint(SYSTEM_PROMPT, reordered, user) == base  # key order is irrelevant
    schema_edit = copy.deepcopy(ATTRIBUTE_TOOL)
    schema_edit["input_schema"]["properties"]["view"]["enum"].append("garden")  # type: ignore[index]
    variants = [
        prompt_fingerprint(SYSTEM_PROMPT + " Be brief.", ATTRIBUTE_TOOL, user),
        prompt_fingerprint(SYSTEM_PROMPT, schema_edit, user),
        prompt_fingerprint(SYSTEM_PROMPT, {**ATTRIBUTE_TOOL, "strict": False}, user),
        prompt_fingerprint(SYSTEM_PROMPT, ATTRIBUTE_TOOL, user.replace("Extract", "List")),
    ]
    assert len({base, *variants}) == 5


def test_cache_key_uses_derived_prompt_version(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "c.jsonl")
    cache.put("text", "m", ExtractedAttributes(rooms=2.0))
    record = json.loads((tmp_path / "c.jsonl").read_text(encoding="utf-8"))
    assert record["prompt_version"] == PROMPT_VERSION


# --- retry on a missing tool call ---------------------------------------------------------------


@pytest.mark.parametrize("model", ["claude-opus-5", "claude-opus-5-5"])
def test_extract_retries_once_when_the_tool_call_is_missing(model: str, tmp_path: Path) -> None:
    client = _client([_text_message(), _tool_message(rooms=3.0)])
    cache = ExtractionCache(tmp_path / "c.jsonl")
    attrs = ClaudeExtractor(model, cache=cache, client=client).extract("3 Zimmer")
    endpoint = client.beta.messages if model == "claude-opus-5" else client.messages
    assert attrs.rooms == 3.0 and len(endpoint.calls) == 2
    assert endpoint.calls[0] == endpoint.calls[1]  # the identical request is sent again
    assert cache.get("3 Zimmer", model) == attrs


def test_extract_gives_up_after_one_retry(caplog: pytest.LogCaptureFixture) -> None:
    client = _client([_text_message()])
    extractor = ClaudeExtractor("claude-opus-5-5", client=client)
    with caplog.at_level(logging.WARNING), pytest.raises(MissingToolCallError):
        extractor.extract("3 Zimmer")
    assert len(client.messages.calls) == 2 and "retrying once" in caplog.text
    assert issubclass(MissingToolCallError, ExtractionError)


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_extract_does_not_retry_refusal_or_truncation(stop_reason: str) -> None:
    client = _client([_text_message(stop_reason)])
    with pytest.raises(ExtractionError) as info:
        ClaudeExtractor(client=client, use_fallbacks=False).extract("3 Zimmer")
    assert not isinstance(info.value, MissingToolCallError)
    assert len(client.messages.calls) == 1


# --- batch deduplication ------------------------------------------------------------------------


def test_batch_requests_sends_each_distinct_text_once(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "c.jsonl")
    extractor = ClaudeExtractor(cache=cache, client=_client())
    requests = extractor.batch_requests(DUPLICATES)
    assert [r["custom_id"] for r in requests] == ["row-1", "row-2"]  # first id of each text
    cache.put("Attika mit Seesicht", extractor.model, ExtractedAttributes())
    assert [r["custom_id"] for r in extractor.batch_requests(DUPLICATES)] == ["row-1"]


def test_collect_batch_maps_results_back_to_duplicate_texts(tmp_path: Path) -> None:
    batches = _Batches({"msgbatch_1": [_succeeded("row-1", rooms=3.0), _succeeded("row-2")]})
    cache = ExtractionCache(tmp_path / "c.jsonl")
    extractor = ClaudeExtractor(cache=cache, client=_client(batches=batches))
    batch_id = extractor.submit_batch(extractor.batch_requests(DUPLICATES))
    texts = {f"row-{i}": text for i, text in DUPLICATES.items()}
    out = extractor.collect_batch(batch_id, texts)
    assert sorted(out) == ["row-1", "row-2", "row-3", "row-4"]
    assert out["row-3"] == out["row-1"] and out["row-3"] is not out["row-1"]
    assert out["row-1"].rooms == 3.0 and out["row-4"].rooms is None
    assert cache.get("3 Zimmer mit Balkon, Tel. [PHONE]", extractor.model) == out["row-1"]


# --- batch splitting ----------------------------------------------------------------------------


def _requests(n: int) -> list[dict[str, object]]:
    return [{"custom_id": f"r{i:03d}", "params": {"x": "y"}} for i in range(n)]


def test_submit_batch_keeps_a_single_id_when_everything_fits() -> None:
    batches = _Batches()
    extractor = ClaudeExtractor(client=_client(batches=batches))
    assert extractor.submit_batch(_requests(3)) == "msgbatch_1"
    assert [len(chunk) for chunk in batches.created] == [3]


def test_submit_batch_splits_by_request_count_and_collects_all() -> None:
    results = {f"msgbatch_{k}": [_succeeded(f"r{k:03d}")] for k in (1, 2, 3)}
    batches = _Batches(results)
    extractor = ClaudeExtractor(client=_client(batches=batches))
    requests = _requests(5)
    ids = extractor.submit_batch(requests, max_requests=2)
    assert ids == ["msgbatch_1", "msgbatch_2", "msgbatch_3"]
    assert [len(chunk) for chunk in batches.created] == [2, 2, 1]
    assert [r for chunk in batches.created for r in chunk] == requests  # order kept, no loss
    assert sorted(extractor.collect_batch(ids, {})) == ["r001", "r002", "r003"]


def test_submit_batch_splits_by_estimated_size() -> None:
    batches = _Batches()
    requests = _requests(5)
    per_request = len(json.dumps(requests[0])) + 2
    extractor = ClaudeExtractor(client=_client(batches=batches))
    ids = extractor.submit_batch(requests, max_bytes=64 + 2 * per_request)
    assert len(ids) == 3 and [len(chunk) for chunk in batches.created] == [2, 2, 1]


def test_submit_batch_default_limit_is_the_api_request_limit() -> None:
    batches = _Batches()
    extractor = ClaudeExtractor(client=_client(batches=batches))
    ids = extractor.submit_batch(_requests(BATCH_MAX_REQUESTS + 1))
    assert BATCH_MAX_REQUESTS == 100_000 and len(ids) == 2
    assert [len(chunk) for chunk in batches.created] == [100_000, 1]


def test_submit_batch_rejects_empty_or_oversized_input() -> None:
    batches = _Batches()
    extractor = ClaudeExtractor(client=_client(batches=batches))
    with pytest.raises(ValueError, match="No batch requests"):
        extractor.submit_batch([])
    with pytest.raises(ValueError, match="r000"):
        extractor.submit_batch(_requests(2), max_bytes=80)
    with pytest.raises(ValueError, match="max_requests"):
        extractor.submit_batch(_requests(2), max_requests=0)
    assert batches.created == []  # validated before anything is sent


def test_submit_batch_logs_batches_already_running_on_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    extractor = ClaudeExtractor(client=_client(batches=_Batches(fail_on=2)))
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="HTTP 500"):
        extractor.submit_batch(_requests(3), max_requests=1)
    assert "Only 1 of 3 batches submitted" in caplog.text and "msgbatch_1" in caplog.text
