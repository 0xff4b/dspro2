"""Tests for rentml.text (anonymisation, language, keywords, TF-IDF, embeddings)."""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from rentml import text as text_mod
from rentml.text import (
    KEYWORD_PATTERNS,
    SentenceEmbedder,
    TfidfSvd,
    anonymize,
    anonymize_series,
    detect_language,
    keyword_flags,
    price_leak_mask,
    price_leak_rate,
    reduce_embeddings,
)

# --- anonymize ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "Tel. 044 123 45 67",
        "Natel +41 79 123 45 67",
        "Anrufe an 0041 79 123 45 67",
        "Büro 0441234567 oder 044/123 45 67",
        "Téléphone: +41 (0)21 123 45 67",
        "Telefono 091 123 45 67",
        "Gratisnummer 0800 123 456",
        "Tel. 044 - 123 45 67",
        "Appelez le +33 1 23 45 67 89",
        "Tel. (044) 123 45 67",
    ],
)
def test_anonymize_removes_phone_numbers(raw: str) -> None:
    out = anonymize(raw)
    assert "[PHONE]" in out
    assert not re.search(r"\d", out)


@pytest.mark.parametrize(
    "raw",
    [
        "CHF 1'850.–",
        "Fr. 2'000.-",
        "1850.-",
        "2 100 CHF",
        "Miete CHF 1850 exkl. NK",
        "loyer: 1'450.- + charges 150.-",
        "affitto CHF 1200",
        "Miete: 1850",
        "Mietzins Fr. 2'000.- inkl. Nebenkosten Fr. 250.-",
        "Loyer mensuel CHF 1'450.00, charges CHF 180.00",
        "Affitto mensile: 1'350.- spese incluse",
        "Mietpreis CHF 1'800 - 2'000 je nach Etage",
        "Mietzinsdepot 5'550 CHF",
        "Kaltmiete 1.850 €",
        "Nettomiete 2150 / Nebenkosten 220",
        "Pigione mensile fr. 1'650.–",
        "Einstellhallenplatz CHF 120.-/Mt., Aussenparkplatz 60.-",
        "Loyer 1450 francs, place de parc 100.-",
        # No word boundary after CHF, comma thousands, compound cues, unit after the amount:
        "Miete CHF1850 exkl. NK", "CHF1'850.- inkl.", "Mietzins CHF 1,850.00 pro Monat",
        "Rent: CHF 2,150 per month", "Gesamtmiete 2150", "Nettomietzins 1850",
        "Bruttomietzins: 2'150", "Mietkosten 1850", "Netto 1850, NK 250", "Mietzinsdepot: 5550",
        "Miete 1850 pro Monat", "Loyer net 1'450/mois", "Affitto 1200 al mese",
        "Mietzins inkl. Heizung 2150", "Mietpreis 1850 Balkon", "Mietzins, 2150 Parkplatz 120",
    ],
)  # fmt: skip
def test_anonymize_removes_prices(raw: str) -> None:
    out = anonymize(raw)
    assert "[PRICE]" in out
    assert not re.search(r"\d", out), out


@pytest.mark.parametrize(
    "raw",
    [
        "Schöne 3.5 Zimmer Wohnung, 85 m2, Baujahr 1995, renoviert 2019, 2. OG",
        "Bel appartement de 3½ pièces, 85 m², rénové en 2019, 3e étage",
        "Appartamento di 3,5 locali, 85 mq, costruito nel 1995",
        "4.5-Zimmer-Wohnung im 3. Stock, Besichtigung 12.-15. Mai",
        "Wohnfläche 120m2, Parkplatz 100 m entfernt, Bezug per 01.05.2026",
        "Lage: 8044 Zürich, Rütistrasse 1, 5 Min. zum Bahnhof",
        "3 1/2 Zimmer, 4 ½ pièces, 70-80 m2",
        "Mo.–Fr. 10–12 Uhr erreichbar",
        "Miete ab 1. Mai",
        "Herrliche Aussicht, Frau oder Herr willkommen",
        "Mietpreis auf Anfrage. Baujahr 1995",
        "Parkplatz in Tiefgarage. Baujahr 2010",
        "Garage vorhanden, renoviert 2019",
        "Loyer attractif, rénové en 2019",
        "Affitto modico, ristrutturato nel 2018",
        "Place de parc disponible, construit en 1998",
        "Miete ab Juli 2026",
        "Parkplatz an der Bahnhofstrasse, 8001 Zürich",
        "Garage in 8400 Winterthur",
        "Nebenkosten gemäss Abrechnung 2025",
        "Besichtigung mit der Frau der Verwaltung",
    ],
)
def test_anonymize_keeps_rooms_areas_years_and_dates(raw: str) -> None:
    assert anonymize(raw) == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Miete für 3.5 Zi.-Whg: 2150", "Miete für 3.5 Zi.-Whg: [PRICE]"),
        ("Mietzins 4.5-Zimmer-Wohnung: 2'650", "Mietzins 4.5-Zimmer-Wohnung: [PRICE]"),
        ("Loyer (3 pièces): 1450", "Loyer (3 pièces): [PRICE]"),
        ("Miete ab Juli 2026: 1850", "Miete ab Juli 2026: [PRICE]"),
        ("Garage, renoviert 2019, Miete 1850", "Garage, renoviert 2019, Miete [PRICE]"),
        ("Miete ab 2000", "Miete ab [PRICE]"),
        ("Mietpreis 1850 Tel. 044", "Mietpreis [PRICE] Tel. 044"),
    ],
)
def test_anonymize_removes_price_next_to_rooms_years_and_postcodes(raw: str, expected: str) -> None:
    assert anonymize(raw) == expected


@pytest.mark.parametrize("run", ["a.", "1.2.", "abc.", "a-", "x@", "Miete "])
def test_anonymize_runs_in_linear_time_on_long_runs(run: str) -> None:
    start = time.perf_counter()
    anonymize(run * 20_000)
    assert time.perf_counter() - start < 1.0


def test_anonymize_removes_email_url_and_iban() -> None:
    raw = (
        "Mail an peter.muster@immo-xyz.ch oder info(at)verwaltung.ch, "
        "Infos auf www.immo-xyz.ch/objekt/123 und https://example.com/a?b=1. "
        "IBAN CH93 0076 2011 6238 5295 7"
    )
    out = anonymize(raw)
    assert out.count("[EMAIL]") == 2
    assert out.count("[URL]") == 2
    assert "[IBAN]" in out
    assert "@" not in out and "immo-xyz" not in out and "0076" not in out
    assert anonymize("iban ch93 0076 2011 6238 5295 7") == "iban [IBAN]"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Kontakt: Peter Muster, Verwaltung", "Kontakt: [NAME], Verwaltung"),
        ("Besichtigung mit Frau Anna Müller.", "Besichtigung mit Frau [NAME]."),
        ("Contact: M. Jean Dupont", "Contact: M. [NAME]"),
        ("Visite avec Mme Dubois", "Visite avec Mme [NAME]"),
        ("Contatto: Sig.ra Maria Rossi", "Contatto: Sig.ra [NAME]"),
        ("Ansprechpartner Herr Dr. Keller", "Ansprechpartner Herr [NAME]"),
        ("Kontakt: P. Muster", "Kontakt: [NAME]"),
        ("Visite avec Mme de la Tour", "Visite avec Mme [NAME]"),
        ("Kontakt: Herr van der Berg", "Kontakt: Herr [NAME]"),
    ],
)
def test_anonymize_replaces_names_after_salutations_and_contact_cues(
    raw: str, expected: str
) -> None:
    assert anonymize(raw) == expected


def test_anonymize_is_idempotent() -> None:
    raw = (
        "Contact: M. Jean Dupont, jean.dupont@gmail.com, +41 79 123 45 67. "
        "Kontakt: Herr Hans Meier. Loyer CHF 1'450.- + charges 150.-"
    )
    once = anonymize(raw)
    assert anonymize(once) == once
    assert "Dupont" not in once and "Meier" not in once


@pytest.mark.parametrize("raw", [None, "", "   ", float("nan"), 42])
def test_anonymize_returns_empty_string_for_missing_input(raw: object) -> None:
    assert anonymize(raw) == ""  # type: ignore[arg-type]


def test_anonymize_placeholders_are_not_price_cues() -> None:
    once = anonymize("Mietpreis 1850, Garage 120. Kontakt: 044 123")
    assert once == "Mietpreis [PRICE], Garage [PRICE]. Kontakt: 044 123"
    assert anonymize(once) == once


def test_anonymize_series_keeps_index_and_missing_values() -> None:
    s = pd.Series(["Miete CHF 1850", None, "  ", "3.5 Zimmer"], index=[10, 11, 12, 13])
    out = anonymize_series(s)
    assert list(out.index) == [10, 11, 12, 13]
    assert out.dtype == "string"
    assert out.loc[10] == "Miete [PRICE]"
    assert out.loc[13] == "3.5 Zimmer"
    assert out.loc[[11, 12]].isna().all()


# --- price leak check ---------------------------------------------------------------------------

_RENT_TEMPLATES = [
    "Schöne 3.5 Zimmer Wohnung, 85 m2, Baujahr 1965. Miete CHF {q}.– exkl. NK",
    "Gesamtmiete {p}, 2. OG, renoviert 2019",
    "Bel appartement de 3½ pièces, loyer: {q}.- + charges 150.-",
    "Mietzins CHF {c}.00 pro Monat, Tel. (044) 123 45 67",
    "Appartamento di 3,5 locali, affitto mensile {p}.– spese incluse",
    "Rent: CHF {c} per month, 4.5 rooms, 110 m2",
    "Miete für 4.5 Zi.-Whg: {p}, Einstellplatz CHF 120.-",
    "Nettomietzins {p} / Nebenkosten 220, Bezug per 01.05.2026",
]


def _listings(n: int, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    rents = pd.Series(rng.integers(2100, 4500, size=n) // 10 * 10, name="price")
    texts = [
        _RENT_TEMPLATES[i % len(_RENT_TEMPLATES)].format(
            p=r, q=f"{r:,}".replace(",", "'"), c=f"{r:,}"
        )
        for i, r in enumerate(rents)
    ]
    return pd.Series(texts, index=rents.index), rents


def test_price_leak_rate_is_zero_after_anonymisation() -> None:
    raw, rents = _listings(160)
    assert price_leak_rate(raw, rents) == 1.0
    anon = anonymize_series(raw)
    assert price_leak_rate(anon, rents) == 0.0
    assert anon.str.contains(r"\[PRICE\]").all()


def test_price_leak_mask_tolerance_alignment_and_missing_values() -> None:
    anon = pd.Series(["Preis [PRICE]", "Baujahr 1995", None, "3.5 Zimmer"], index=[4, 3, 2, 1])
    price = pd.Series({1: 2000.0, 2: 1800.0, 3: 1950.0, 4: np.nan})  # aligned by index
    assert price_leak_mask(anon, price).tolist() == [False, True, False, False]
    assert price_leak_mask(anon, price, rel_tol=0.01).tolist() == [False, False, False, False]
    assert price_leak_rate(anon, price) == pytest.approx(1 / 2)  # rows 4 (no rent), 2 (no text) out
    with pytest.raises(ValueError):
        price_leak_mask(anon, price, rel_tol=-0.1)


@pytest.mark.parametrize("text", ["Nr. 1'900", "Total 1,900.00", "1900.-", "1 900 CHF", "1.900"])
def test_price_leak_mask_parses_thousands_separators(text: str) -> None:
    assert price_leak_mask(pd.Series([text]), pd.Series([1900.0])).tolist() == [True]


def test_price_leak_rate_is_nan_without_text_or_rent() -> None:
    assert np.isnan(price_leak_rate(pd.Series([None, " "]), pd.Series([1500.0, 1600.0])))
    assert np.isnan(price_leak_rate(pd.Series(["Miete 1500"]), pd.Series([0.0])))
    empty = price_leak_mask(pd.Series(["keine Zahl"], index=[7]), pd.Series([1500.0], index=[7]))
    assert empty.tolist() == [False] and list(empty.index) == [7]


# --- detect_language ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "lang"),
    [
        ("Schöne Wohnung mit Balkon und Seesicht, ruhige Lage", "de"),
        ("Bel appartement lumineux avec balcon et vue sur le lac", "fr"),
        ("Appartamento luminoso con balcone e vista lago, vicino alla stazione", "it"),
        ("Bright flat with a balcony and a view of the lake", "en"),
    ],
)
def test_detect_language_recognises_de_fr_it_en(raw: str, lang: str) -> None:
    assert detect_language(raw) == lang


def test_detect_language_ignores_single_letter_e() -> None:
    assert detect_language("3.5 Zimmer, 2. OG, Balkon, E-Mail an Verwaltung") == "de"
    assert detect_language("Appartement au 3e étage, balcon, e-mail") == "fr"


@pytest.mark.parametrize("raw", [None, "", "12345 [PRICE]", "Minergie"])
def test_detect_language_returns_unknown_without_evidence(raw: str | None) -> None:
    assert detect_language(raw) == "unknown"


# --- keyword_flags ------------------------------------------------------------------------------


def test_keyword_patterns_compile_without_capture_groups() -> None:
    for name, pattern in KEYWORD_PATTERNS.items():
        assert re.compile(pattern, re.IGNORECASE).groups == 0, name


def test_keyword_patterns_are_lower_case_for_lowered_matching() -> None:
    for name, pattern in KEYWORD_PATTERNS.items():
        literal = re.sub(r"\\.", "", pattern)  # drop escapes such as \\W and \\b
        assert literal == literal.lower(), name


def test_keyword_flags_detects_multilingual_mentions() -> None:
    texts = pd.Series(
        [
            "Wohnung mit Balkon, Lift und Seesicht",
            "Appartement avec balcon, ascenseur, vue sur le lac",
            "Appartamento con balcone, ascensore e vista sul lago",
            None,
        ],
        index=["a", "b", "c", "d"],
    )
    flags = keyword_flags(texts)
    assert list(flags.columns) == [f"kw_{name}" for name in KEYWORD_PATTERNS]
    assert (flags.dtypes == "int8").all()
    assert flags.loc[["a", "b", "c"], ["kw_balcony", "kw_lift", "kw_lake_view"]].eq(1).all().all()
    assert flags.loc["d"].sum() == 0


@pytest.mark.parametrize(
    ("raw", "column", "expected"),
    [
        ("unmöbliert", "kw_furnished", 0),
        ("möbliert", "kw_furnished", 1),
        ("non meublé", "kw_furnished", 0),
        ("unbefristet", "kw_temporary", 0),
        ("befristet bis Juni", "kw_temporary", 1),
        ("Kindergarten in der Nähe", "kw_garden", 0),
        ("Gartensitzplatz", "kw_garden", 1),
        ("WG-Zimmer frei", "kw_shared_flat", 1),
        ("WG-taugliche Wohnung", "kw_shared_flat", 0),
        ("Coop und Migros in der Nähe", "kw_cooperative", 0),
        ("Skilift in der Nähe", "kw_lift", 0),
        ("Personenlift im Haus", "kw_lift", 1),
        ("Wohnbaugenossenschaft", "kw_cooperative", 1),
        ("nicht möbliert", "kw_furnished", 0),
        ("non-meublé", "kw_furnished", 0),
        ("nicht befristet", "kw_temporary", 0),
        ("Gartenstrasse 12", "kw_garden", 0),
        ("Gartenweg 3", "kw_garden", 0),
        ("Garten mit Sitzplatz", "kw_garden", 1),
    ],
)
def test_keyword_flags_negation_and_false_friends(raw: str, column: str, expected: int) -> None:
    assert keyword_flags(pd.Series([raw]))[column].iloc[0] == expected


# --- TfidfSvd -----------------------------------------------------------------------------------

_VOCAB = [
    "wohnung", "balkon", "seesicht", "lift", "ruhig", "hell", "küche", "bad", "parkett", "garten",
    "zimmer", "appartement", "balcon", "ascenseur", "cuisine", "jardin", "appartamento",
    "balcone", "cucina", "giardino",
]  # fmt: skip


def _corpus(n: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    docs = [" ".join(rng.choice(_VOCAB, size=12)) for _ in range(n)]
    return pd.Series(docs, index=pd.RangeIndex(1000, 1000 + n, name="listing_id"))


def test_tfidf_svd_returns_frame_with_index_and_is_deterministic() -> None:
    corpus = _corpus(60)
    first = TfidfSvd(n_components=10).fit_transform(corpus)
    second = TfidfSvd(n_components=10).fit(corpus).transform(corpus)
    assert first.shape == (60, 10)
    assert list(first.columns) == [f"tfidf_{i}" for i in range(10)]
    assert first.index.equals(corpus.index)
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy(), atol=1e-10)


def test_tfidf_svd_transforms_unseen_text_and_na() -> None:
    model = TfidfSvd(n_components=5).fit(_corpus(40))
    out = model.transform(pd.Series(["balkon seesicht", None, "völlig neue wörter"]))
    assert out.shape == (3, 5)
    assert np.isfinite(out.to_numpy()).all()
    np.testing.assert_allclose(out.iloc[1].to_numpy(), 0.0)


def test_tfidf_svd_caps_components_and_clones() -> None:
    model = TfidfSvd(n_components=500, min_df=1).fit(_corpus(8))
    assert model.n_components_ <= 7
    assert clone(model).get_params()["n_components"] == 500


def test_tfidf_svd_rejects_empty_corpus_and_single_string() -> None:
    with pytest.raises(ValueError):
        TfidfSvd().fit(pd.Series([None, "", None]))
    with pytest.raises(TypeError):
        TfidfSvd().fit("just one string")


# --- SentenceEmbedder ---------------------------------------------------------------------------


class _FakeModel:
    """Deterministic stand-in for SentenceTransformer (no download)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(list(texts))
        return np.array([[len(t), sum(map(ord, t)) % 97, 1.0] for t in texts], dtype=np.float64)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeModel:
    model = _FakeModel()
    monkeypatch.setattr(SentenceEmbedder, "_load_model", lambda self: model)
    return model


def test_sentence_embedder_encodes_each_unique_text_once(fake_model: _FakeModel) -> None:
    embedder = SentenceEmbedder(cache_dir=None)
    out = embedder.encode(["a", "bb", "a", None])  # type: ignore[list-item]
    assert out.shape == (4, 3) and out.dtype == np.float32
    np.testing.assert_array_equal(out[0], out[2])
    assert fake_model.calls == [["a", "bb", ""]]
    embedder.encode(["bb", "a"])
    assert len(fake_model.calls) == 1


def test_sentence_embedder_disk_cache_survives_new_instance(
    fake_model: _FakeModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = SentenceEmbedder(model_name="org/model", cache_dir=tmp_path).encode(["x", "yy"])
    files = list((tmp_path / "org__model").glob("*.npy"))
    assert len(files) == 2
    assert {f.stem for f in files} == {SentenceEmbedder.text_key(t) for t in ("x", "yy")}

    def _fail(self: SentenceEmbedder) -> None:
        raise AssertionError("model must not be loaded on a full cache hit")

    monkeypatch.setattr(SentenceEmbedder, "_load_model", _fail)
    second = SentenceEmbedder(model_name="org/model", cache_dir=tmp_path).encode(["yy", "x"])
    np.testing.assert_array_equal(second, first[::-1])


def test_sentence_embedder_ignores_corrupt_cache_file(
    fake_model: _FakeModel, tmp_path: Path
) -> None:
    model_dir = tmp_path / "org__model"
    model_dir.mkdir()
    (model_dir / f"{SentenceEmbedder.text_key('x')}.npy").write_bytes(b"not numpy")
    out = SentenceEmbedder(model_name="org/model", cache_dir=tmp_path).encode(["x"])
    assert out.shape == (1, 3)
    assert fake_model.calls == [["x"]]


def test_sentence_embedder_cache_is_separate_per_normalisation(
    fake_model: _FakeModel, tmp_path: Path
) -> None:
    SentenceEmbedder(model_name="org/model", cache_dir=tmp_path).encode(["x"])
    raw = SentenceEmbedder(model_name="org/model", cache_dir=tmp_path, normalize=False)
    raw.encode(["x"])
    assert len(fake_model.calls) == 2  # the normalised vector is not reused
    assert (tmp_path / "org__model__unnormalized").is_dir()
    assert {p.name for p in tmp_path.iterdir()} == {"org__model", "org__model__unnormalized"}


def test_sentence_embedder_empty_input_does_not_load_model(fake_model: _FakeModel) -> None:
    assert SentenceEmbedder().encode([]).shape == (0, 0)
    assert fake_model.calls == []


def test_sentence_embedder_is_available_checks_importability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(SentenceEmbedder.is_available(), bool)
    monkeypatch.setattr(text_mod.importlib.util, "find_spec", lambda name: None)
    assert SentenceEmbedder.is_available() is False


# --- reduce_embeddings --------------------------------------------------------------------------


def test_reduce_embeddings_fits_on_train_only() -> None:
    rng = np.random.default_rng(42)
    train, test = rng.normal(size=(50, 20)), rng.normal(loc=5.0, size=(10, 20))
    train_red, (test_red,) = reduce_embeddings(train, [test], n_components=8, seed=42)
    assert train_red.shape == (50, 8) and test_red.shape == (10, 8)
    np.testing.assert_allclose(train_red.mean(axis=0), 0.0, atol=1e-10)
    assert np.abs(test_red.mean(axis=0)).max() > 0.1


def test_reduce_embeddings_caps_components_and_validates_shapes() -> None:
    rng = np.random.default_rng(42)
    train_red, others = reduce_embeddings(rng.normal(size=(5, 4)), n_components=32)
    assert train_red.shape == (5, 4) and others == []
    with pytest.raises(ValueError):
        reduce_embeddings(rng.normal(size=(5, 4)), [rng.normal(size=(3, 6))])
    with pytest.raises(ValueError):
        reduce_embeddings(np.zeros(4))
