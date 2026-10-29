"""Tests for rentml.llm with a fake Anthropic client (no network)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rentml.llm import ClaudeExtractor
from rentml.text import ExtractionError, extract_frame

ANSWER = {
    "has_lift": True,
    "rooms": 3.5,
    "view": "none",
    "evidence": {"has_lift": "Lift", "rooms": None},
}


def _message(stop_reason: str = "end_turn") -> SimpleNamespace:
    text = SimpleNamespace(type="text", text=json.dumps(ANSWER))
    return SimpleNamespace(stop_reason=stop_reason, content=[text])


class FakeClient:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.sent: list[dict] = []
        self.batches: list[list[dict]] = []
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create, batches=self)

    def _create(self, **params: object) -> SimpleNamespace:
        self.sent.append(params)
        return _message(self.stop_reason)

    def create(self, requests: list[dict]) -> SimpleNamespace:
        self.batches.append(requests)
        return SimpleNamespace(id=f"batch-{len(self.batches)}")

    def results(self, batch_id: str) -> list[SimpleNamespace]:
        items = [r["custom_id"] for b in self.batches for r in b]
        ok = SimpleNamespace(type="succeeded", message=_message())
        return [SimpleNamespace(custom_id=items[0], result=ok)] + [
            SimpleNamespace(custom_id=i, result=SimpleNamespace(type="errored")) for i in items[1:]
        ]


def test_extractor_anonymises_parses_and_caches(tmp_path: Path) -> None:
    cache = tmp_path / "extraction.jsonl"
    client = FakeClient()
    extractor = ClaudeExtractor(cache_path=cache, client=client)
    attrs = extractor("3.5 Zimmer mit Lift, Miete CHF 1850, Tel. 079 123 45 67")
    assert attrs["has_lift"] is True and attrs["evidence"] == {"has_lift": "Lift"}
    prompt = client.sent[0]["messages"][0]["content"]
    assert "1850" not in prompt and "[PHONE]" in prompt
    assert client.sent[0]["output_config"]["format"]["type"] == "json_schema"
    again = ClaudeExtractor(cache_path=cache, client=FakeClient())
    frame = extract_frame(
        pd.Series(["3.5 Zimmer mit Lift, Miete CHF 1850, Tel. 079 123 45 67"]), again
    )
    assert again.client.sent == []  # served from the JSONL cache
    assert frame.loc[0, "attr_rooms"] == 3.5 and frame.loc[0, "extractor"] == "claude:claude-opus-5"
    assert extractor(None) == {"evidence": {}}


def test_refusal_raises(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError):
        ClaudeExtractor(client=FakeClient(stop_reason="refusal"))("Wohnung mit Lift")


def test_batch_roundtrip(tmp_path: Path) -> None:
    client = FakeClient()
    extractor = ClaudeExtractor(cache_path=tmp_path / "c.jsonl", client=client)
    texts = pd.Series(["Wohnung A", "Wohnung A", "Wohnung B", None, " "])
    requests = extractor.batch_requests(texts)
    assert len(requests) == 2  # duplicates, NA and blank texts are skipped
    assert all(len(r["custom_id"]) == 64 for r in requests)
    ids = extractor.submit_batch(requests, chunk_size=1)
    assert ids == ["batch-1", "batch-2"]
    assert extractor.collect_batch(ids[:1]) == 1  # one succeeded, one errored
    assert len(extractor.batch_requests(texts)) == 1
