"""Tests for rentml.extraction.rule_based_extract (regex rules in rentml._extraction_rules)."""

import pytest

from rentml.extraction import ExtractedAttributes, rule_based_extract

# --- rule_based_extract -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Schöne 3.5-Zimmer-Wohnung im 2. OG mit Balkon und Seesicht, Wohnfläche 85 m2, "
            "Lift vorhanden. Totalsaniert 2019. Einstellhallenplatz. Haustiere erlaubt.",
            {"rooms": 3.5, "floor": 2, "area_sqm": 85.0, "has_lift": True, "view": "lake",
             "renovation_year": 2019, "parking": "garage", "pets_allowed": True,
             "has_balcony_or_terrace": True},
        ),
        (
            "Bel appartement de 3½ pièces au 3e étage, 72 m², rénové en 2018, sans ascenseur, "
            "vue sur les Alpes, place de parc. Pas d'animaux. Meublé.",
            {"rooms": 3.5, "floor": 3, "area_sqm": 72.0, "renovation_year": 2018,
             "has_lift": False, "view": "mountain", "parking": "outdoor", "pets_allowed": False,
             "is_furnished": True},
        ),
        (
            "Appartamento di 3,5 locali al piano terra, 90 mq, giardino, cooperativa.",
            {"rooms": 3.5, "floor": 0, "area_sqm": 90.0, "has_garden": True,
             "rent_regime": "cooperative"},
        ),
    ],
)  # fmt: skip
def test_rule_based_extract_de_fr_it(raw: str, expected: dict[str, object]) -> None:
    attrs = rule_based_extract(raw)
    for name, value in expected.items():
        assert getattr(attrs, name) == value, name
    assert set(attrs.evidence) >= set(expected)
    assert all(span in raw for span in attrs.evidence.values())
    assert attrs.extractor == "rules"


@pytest.mark.parametrize(
    ("raw", "rooms"),
    [
        ("3.5-Zimmer-Wohnung", 3.5),
        ("3½ pièces", 3.5),
        ("3,5 locali", 3.5),
        ("4 1/2 Zimmer", 4.5),
        ("5.5 Zi.-Whg", 5.5),
        ("2 Zimmer", 2.0),
        ("Studio meublé", 1.0),
        ("mit 2 Schlafzimmern", None),
    ],
)
def test_rule_based_extract_room_formats(raw: str, rooms: float | None) -> None:
    assert rule_based_extract(raw).rooms == rooms


@pytest.mark.parametrize(
    ("raw", "floor"),
    [
        ("Wohnung im 2. OG", 2),
        ("im 3. Stock", 3),
        ("au 3e étage", 3),
        ("1er étage", 1),
        ("2° piano", 2),
        ("secondo piano", 2),
        ("im zweiten Stock", 2),
        ("Erdgeschoss", 0),
        ("rez-de-chaussée", 0),
        ("piano terra", 0),
        ("ruhige Lage", None),
        ("Wohnung im obersten Stock", None),
        ("Wohnung im untersten Stock", None),
        ("Waschküche im EG, Wohnung im 3. OG", 3),
        ("Keller im Erdgeschoss; Wohnung 2. Stock", 2),
        ("Wohnung im EG, Estrich im 3. Stock", 0),
        ("Erdgeschosswohnung mit Garten", 0),
    ],
)
def test_rule_based_extract_floor_formats(raw: str, floor: int | None) -> None:
    assert rule_based_extract(raw).floor == floor


@pytest.mark.parametrize(
    ("raw", "regime"),
    [
        ("WG-Zimmer in 3er WG", "shared_flat"),
        ("Chambre en colocation", "shared_flat"),
        ("Zimmer in Untermiete", "shared_flat"),
        ("Genossenschaftswohnung, Anteilschein erforderlich", "cooperative"),
        ("Logement de la coopérative", "cooperative"),
        ("Subventionierte Wohnung", "cost_based_or_subsidised"),
        ("WG-taugliche 4.5-Zimmer-Wohnung", "market"),
        ("Coop in der Nähe", "market"),
        ("Möbliertes Zimmer in einer 3er-WG", "shared_flat"),
        ("Wir suchen eine neue Mitbewohnerin", "shared_flat"),
        ("Cerchiamo una coinquilina", "shared_flat"),
        ("4.5-Zimmer-Wohnung, ideal für eine WG", "market"),
        ("Auch als WG möglich", "market"),
        ("Keine WG", "market"),
        ("Idéal pour une colocation", "market"),
        ("Appartement de 4 pièces, colocation possible", "market"),
        ("Wohnung in Genossenschaftssiedlung? Nein, private Verwaltung", "market"),
        ("Keine Genossenschaftswohnung", "market"),
        ("Genossenschaftswohnung, auch als WG möglich", "cooperative"),
    ],
)
def test_rule_based_extract_rent_regime(raw: str, regime: str) -> None:
    assert rule_based_extract(raw).rent_regime == regime


def test_rule_based_extract_negations_and_context_rules() -> None:
    attrs = rule_based_extract(
        "Unmöbliert, ohne Lift, keine Haustiere, unbefristet. Gemeinsame Waschküche. "
        "Balkon 10 m2, Wohnfläche 95 m2. Baujahr 1978, renoviert 2012."
    )
    assert (attrs.is_furnished, attrs.has_lift, attrs.pets_allowed) == (False, False, False)
    assert (attrs.is_temporary, attrs.has_own_washer) == (False, False)
    assert attrs.area_sqm == 95.0
    assert attrs.renovation_year == 2012
    assert rule_based_extract("renoviert, Baujahr 1978").renovation_year is None


@pytest.mark.parametrize(
    ("raw", "name", "value", "evidence"),
    [
        ("Pas d'ascenseur", "has_lift", False, "Pas d'ascenseur"),
        ("Kein Personenlift", "has_lift", False, "Kein Personenlift"),
        ("Lift: nein", "has_lift", False, "Lift: nein"),
        ("Non c'è ascensore", "has_lift", False, "Non c'è ascensore"),
        ("Ohne Lift aber mit Balkon", "has_lift", False, "Ohne Lift"),
        ("nicht möbliert", "is_furnished", False, "nicht möbliert"),
        ("non-meublé", "is_furnished", False, "non-meublé"),
        ("nicht befristet", "is_temporary", False, "nicht befristet"),
        ("Kein Balkon", "has_balcony_or_terrace", False, "Kein Balkon"),
        ("Sans balcon", "has_balcony_or_terrace", False, "Sans balcon"),
        ("Senza balcone", "has_balcony_or_terrace", False, "Senza balcone"),
        ("Kein Balkon, aber grosse Terrasse", "has_balcony_or_terrace", True, "Terrasse"),
        ("Ohne Lift aber mit Balkon", "has_balcony_or_terrace", True, "Balkon"),
        ("nicht nur Balkon sondern auch Terrasse", "has_balcony_or_terrace", True, "Balkon"),
        ("Keine Haustiere, Balkon vorhanden", "has_balcony_or_terrace", True, "Balkon"),
        ("Kein Garten", "has_garden", False, "Kein Garten"),
        ("Gartenstrasse 12", "has_garden", None, None),
        ("Keine Seesicht", "view", "none", None),
        ("Vue sur le lac non loin de la gare", "view", "lake", "Vue sur le lac"),
        ("Keine Garage, Parkplatz vorhanden", "parking", "outdoor", "Parkplatz"),
        ("3.5-Zimmer-Wohnung mit Balkon 85 m2", "area_sqm", 85.0, "85 m2"),
        ("Wohnung mit Balkon 10 m2", "area_sqm", None, None),
    ],
)
def test_rule_based_extract_negations_amenities_and_street_names(
    raw: str, name: str, value: object, evidence: str | None
) -> None:
    attrs = rule_based_extract(raw)
    assert getattr(attrs, name) == value
    assert attrs.evidence.get(name) == evidence


def test_rule_based_extract_is_case_insensitive_with_original_evidence() -> None:
    raw = "İstanbul-Stil: 3.5 ZIMMER im 2. OG, BALKON, WG-ZIMMER"
    attrs = rule_based_extract(raw)
    assert (attrs.rooms, attrs.floor, attrs.has_balcony_or_terrace) == (3.5, 2, True)
    assert attrs.rent_regime == "shared_flat"
    assert attrs.evidence["rooms"] == "3.5 ZIMMER" and attrs.evidence["floor"] == "2. OG"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_rule_based_extract_empty_input_gives_defaults(raw: str | None) -> None:
    attrs = rule_based_extract(raw, lang="de")
    assert attrs == ExtractedAttributes(extractor="rules")
