"""Tests for rentml.text."""

import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rentml.text import (
    ATTRIBUTE_FIELDS,
    ExtractionError,
    anonymize,
    anonymize_series,
    detect_language,
    embed_texts,
    evaluate_extraction,
    extract_frame,
    keyword_flags,
    make_tfidf_svd,
    price_leak_mask,
    price_leak_rate,
    reduce_embeddings,
    rule_based_extract,
    tfidf_features,
)

DEMO = pd.Series(
    {
        1: "Schöne 3.5-Zimmer-Wohnung mit Balkon und Seesicht in 8044 Zürich, Baujahr 1995, "
        "renoviert 2021. Mietzins CHF 2'150.– plus Nebenkosten 250.–. Bezug ab 1. Juli 2026. "
        "Kontakt: Frau Anna Muster, Tel. 079 123 45 67, anna.muster@example.ch",
        2: "Bel appartement de 4 pièces (95 m²) à Lausanne avec ascenseur et balcon, vue sur le "
        "lac. Loyer: 1'980 CHF + charges 180 CHF. Renseignements: M. Jean Dupont, "
        "+41 21 555 12 34, www.regie-exemple.ch",
        3: "Appartamento di 2.5 locali a Lugano con terrazza e vista lago, 3° piano, ascensore. "
        "Affitto mensile Fr. 1'450.- spese escluse. Contatto: Sig.ra Maria Rossi, "
        "091 987 65 43, maria.rossi@esempio.ch",
    }
)
RENT = pd.Series({1: 2150.0, 2: 1980.0, 3: 1450.0})


def test_anonymize_removes_contacts_and_rent_but_keeps_facts() -> None:
    anon = anonymize_series(DEMO)
    for text in anon:
        assert "[PRICE]" in text and "[PHONE]" in text and "[NAME]" in text
        assert "@" not in text and "Muster" not in text and "Rossi" not in text
    assert "[URL]" in anon[2]
    assert all(kept in anon[1] for kept in ("3.5-Zimmer", "8044 Zürich", "1995", "2021", "2026"))
    assert "95 m²" in anon[2]
    assert anonymize_series(anon).equals(anon)  # idempotent


@pytest.mark.parametrize(
    ("raw", "kept"),
    [
        ("Miete: 1850", None),
        ("Fr. 2000 pro Monat", None),
        ("2 100 CHF", None),
        ("Miete ab Juli 2026: 1850", "2026"),
        ("Parkplatz an der Bahnhofstrasse, 8001 Zürich", "8001"),
        ("Besichtigung Fr. 10-12 Uhr", "10-12"),
    ],
)
def test_anonymize_prices(raw: str, kept: str | None) -> None:
    out = anonymize(raw)
    if kept is None:
        assert "[PRICE]" in out and not re.search(r"\d", out)
    else:
        assert kept in out and "1850" not in out


def test_anonymize_series_blank_and_missing_become_na() -> None:
    out = anonymize_series(pd.Series(["  ", None, "Hallo"]))
    assert out.isna().tolist() == [True, True, False]
    assert anonymize(None) == ""


def test_price_leak_rate_drops_after_anonymisation() -> None:
    assert price_leak_rate(DEMO.astype("string"), RENT) == 1.0
    assert price_leak_rate(anonymize_series(DEMO), RENT) == 0.0
    mask = price_leak_mask(pd.Series(["Miete 1'990.-", "3 Zimmer", None]), pd.Series([2000.0] * 3))
    assert mask.tolist() == [True, False, False]
    assert np.isnan(price_leak_rate(pd.Series([None]), pd.Series([1000.0])))


def test_detect_language() -> None:
    assert [detect_language(t) for t in DEMO] == ["de", "fr", "it"]
    assert detect_language("") == "unknown"
    assert detect_language(None) == "unknown"


def test_keyword_flags() -> None:
    flags = keyword_flags(pd.Series(["Balkon und Lift", "unmöbliert, Kindergarten", None]))
    assert flags.loc[0, ["kw_balcony", "kw_lift"]].tolist() == [1, 1]
    assert flags.loc[1, ["kw_furnished", "kw_garden"]].tolist() == [0, 0]
    assert flags.loc[2].sum() == 0 and flags.dtypes.eq("int8").all()


def test_rule_based_extract_values_and_evidence() -> None:
    text = anonymize(DEMO[3])
    attrs = rule_based_extract(text)
    assert attrs["rooms"] == 2.5 and attrs["floor"] == 3 and attrs["view"] == "lake"
    assert attrs["has_lift"] is True and attrs["has_balcony_or_terrace"] is True
    assert all(span in text for span in attrs["evidence"].values())
    assert set(attrs["evidence"]) == set(attrs) - {"evidence"}


def test_rule_based_extract_negations_and_regime() -> None:
    text = (
        "WG-Zimmer im 2. OG, 85 m2 Wohnfläche, Balkon 10 m2. Ohne Lift, unmöbliert, "
        "keine Haustiere, renoviert 2019."
    )
    attrs = rule_based_extract(text)
    assert attrs["has_lift"] is False and attrs["is_furnished"] is False
    assert attrs["pets_allowed"] is False and attrs["rent_regime"] == "shared_flat"
    assert attrs["floor"] == 2 and attrs["area_sqm"] == 85.0 and attrs["renovation_year"] == 2019
    assert rule_based_extract("Ideal für eine WG").get("rent_regime") is None
    assert rule_based_extract(None) == {"evidence": {}}


def test_extract_frame_columns_and_dedup() -> None:
    calls: list[str] = []

    def extractor(text: str) -> dict:
        calls.append(text)
        if text == "boom":
            raise ExtractionError("refused")
        return rule_based_extract(text)

    texts = pd.Series(["Balkon, Genossenschaft", "Balkon, Genossenschaft", None, "boom"])
    with pytest.raises(ExtractionError):
        extract_frame(texts, extractor)
    calls.clear()
    frame = extract_frame(texts, extractor, include_evidence=True, errors="ignore")
    assert len(calls) == 3  # each distinct text once
    assert frame["attr_has_balcony_or_terrace"].tolist()[:2] == [1.0, 1.0]
    assert frame["rent_regime"].tolist()[:3] == ["cooperative", "cooperative", "market"]
    assert frame["attr_view"].dtype == "category" and frame.loc[2, "attr_view"] == "none"
    assert frame.loc[3].isna().all()  # failed row stays NA
    assert json.loads(frame.loc[0, "evidence"])["has_balcony_or_terrace"] == "Balkon"
    assert {f"attr_{f}" for f in ATTRIBUTE_FIELDS if f != "rent_regime"} <= set(frame.columns)


def test_evaluate_extraction_counts() -> None:
    pred = pd.DataFrame(
        {"attr_has_lift": [1.0, 0.0, np.nan, 1.0], "attr_area_sqm": [80.5, 60, 0, 1]}
    )
    gold = pd.DataFrame({"has_lift": ["ja", "true", "", None], "area_sqm": [80, 60, None, None]})
    lang = pd.Series(["de", "de", "fr", "fr"], name="lang")
    table = evaluate_extraction(pred, gold, ["has_lift", "area_sqm"]).set_index("field")
    # has_lift: 1 TP, 1 wrong value (FP + FN), 1 FP without gold -> P = 1/3, R = 1/2.
    assert table.loc["has_lift", ["support", "n_pred"]].tolist() == [2, 3]
    assert table.loc["has_lift", "precision"] == pytest.approx(1 / 3)
    assert table.loc["has_lift", "recall"] == pytest.approx(0.5)
    assert table.loc["area_sqm", "recall"] == 1.0  # 80.5 vs 80 is within 1 m²
    by_lang = evaluate_extraction(pred, gold, ["has_lift"], by=lang)
    assert by_lang["lang"].tolist() == ["de", "fr"]
    assert np.isnan(by_lang.loc[1, "recall"])  # no gold label in that group


def test_tfidf_svd_pipeline() -> None:
    texts = pd.Series([f"Wohnung {w} mit Balkon Nummer {i}" for i, w in enumerate("abcdefgh")])
    pipe = make_tfidf_svd(n_components=3, min_df=1).fit(texts.tolist())
    feats = tfidf_features(pipe, pd.Series(["Balkon", None], index=[10, 11]))
    assert feats.columns.tolist() == ["tfidf_0", "tfidf_1", "tfidf_2"]
    assert feats.index.tolist() == [10, 11] and np.isfinite(feats.to_numpy()).all()


def test_embed_texts_uses_disk_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    encoded: list[str] = []

    class FakeModel:
        def __init__(self, name: str, device: str | None = None) -> None:
            pass

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            encoded.extend(texts)
            return np.array([[len(t), 1.0] for t in texts])

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    first = embed_texts(["ab", "abc", "ab"], cache_dir=tmp_path)
    second = embed_texts(["abc", None], cache_dir=tmp_path)
    assert first.shape == (3, 2) and first.dtype == np.float32
    assert encoded == ["ab", "abc", ""]  # "abc" came from the cache the second time
    np.testing.assert_array_equal(second[0], first[1])


def test_reduce_embeddings_fits_on_train_only() -> None:
    rng = np.random.default_rng(42)
    train, test = rng.normal(size=(20, 6)), rng.normal(size=(5, 6))
    red_train, (red_test,) = reduce_embeddings(train, [test], n_components=4)
    assert red_train.shape == (20, 4) and red_test.shape == (5, 4)
    assert reduce_embeddings(train[:3], n_components=10)[0].shape == (3, 3)  # capped
