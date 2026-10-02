"""Structured attribute extraction from (anonymised) listing descriptions.

Two extractors share one schema (:data:`ATTRIBUTE_SCHEMA`) and one result type
(:class:`ExtractedAttributes`): a transparent regex baseline (:func:`rule_based_extract`, rules in
:mod:`rentml._extraction_rules`) and an LLM extractor (:class:`ClaudeExtractor`, structured output
via forced tool use, Message Batches for bulk runs, JSONL cache; implemented in
:mod:`rentml.extraction_llm` and re-exported here). Every non-null field carries an evidence span
quoted from the text.
"""

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from rentml._extraction_rules import evidence_span, find_attributes, lower_keep_length

logger = logging.getLogger(__name__)

# Defined in rentml.extraction_llm (prompt, cache, Claude client); resolved by __getattr__ below.
_LLM_EXPORTS = frozenset(
    {"ClaudeExtractor", "ExtractionCache", "DEFAULT_CLAUDE_MODEL", "PROMPT_VERSION", "build_prompt"}
)

VIEWS = ("none", "lake", "mountain", "city", "other")
PARKING = ("none", "outdoor", "garage")
RENT_REGIMES = ("market", "cooperative", "cost_based_or_subsidised", "shared_flat")
_ENUMS: dict[str, tuple[str, ...]] = dict(view=VIEWS, parking=PARKING, rent_regime=RENT_REGIMES)
# Label words accepted in hand-labelled gold files and messy model output (DE/FR/IT/EN).
_MISSING_WORDS = frozenset({"", "nan", "none", "null", "na", "n/a", "<na>", "-"})
_NULL_LIKE = _MISSING_WORDS | {"market"}
_BOOL_WORDS: dict[str, bool] = {
    **dict.fromkeys(("true", "yes", "ja", "oui", "si", "sì", "wahr", "vrai", "vero"), True),
    **dict.fromkeys(("false", "no", "nein", "non", "falsch", "faux", "falso"), False),
}

# field -> (JSON kind, description); kinds: integer | number | boolean | enum.
_FIELD_SPECS: dict[str, tuple[str, str]] = {
    "floor": ("integer", "Floor; ground floor (EG, rez, piano terra) = 0, '2. OG' = 2."),
    "has_lift": ("boolean", "Building has a lift/elevator."),
    "has_balcony_or_terrace": ("boolean", "Flat has a balcony, loggia, terrace or patio."),
    "view": ("enum", "Notable view: lake, mountain, city, other; none if not mentioned."),
    "renovation_year": ("integer", "Year of the most recent renovation (not construction)."),
    "is_furnished": ("boolean", "Flat is let furnished."),
    "is_temporary": ("boolean", "Fixed-term or temporary lease."),
    "parking": ("enum", "garage = indoor/underground space, outdoor = outdoor space, else none."),
    "is_minergie": ("boolean", "Building is Minergie certified."),
    "pets_allowed": ("boolean", "Pets explicitly allowed (true) or forbidden (false)."),
    "has_garden": ("boolean", "Private or shared garden."),
    "has_own_washer": ("boolean", "Own washer/tumbler in the flat (false: shared laundry)."),
    "is_new_build": ("boolean", "New building or first occupancy."),
    "is_attic": ("boolean", "Attic, penthouse or top-floor loft flat."),
    "area_sqm": ("number", "Living area in m² (not garden, terrace or cellar)."),
    "rooms": ("number", "Swiss room count, e.g. 3.5 for '3½ Zimmer'."),
    "rent_regime": ("enum", "market, cooperative, cost_based_or_subsidised or shared_flat."),
}
ATTRIBUTE_FIELDS: tuple[str, ...] = tuple(_FIELD_SPECS)


def _property_schema(name: str, kind: str, description: str) -> dict[str, object]:
    if kind == "enum":
        return {"type": "string", "enum": list(_ENUMS[name]), "description": description}
    return {"anyOf": [{"type": kind}, {"type": "null"}], "description": description}


ATTRIBUTE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        **{name: _property_schema(name, *spec) for name, spec in _FIELD_SPECS.items()},
        "evidence": {
            "type": "object",
            "description": "Verbatim quote from the description supporting each set field.",
            "properties": {
                name: {"anyOf": [{"type": "string"}, {"type": "null"}]} for name in ATTRIBUTE_FIELDS
            },
            "required": list(ATTRIBUTE_FIELDS),
            "additionalProperties": False,
        },
    },
    "required": [*ATTRIBUTE_FIELDS, "evidence"],
    "additionalProperties": False,
}


class ExtractionError(RuntimeError):
    """The model returned no usable result (refusal, truncation or missing tool call)."""


@dataclass
class ExtractedAttributes:
    """Attributes of one description; ``None`` means "not mentioned"."""

    floor: int | None = None
    has_lift: bool | None = None
    has_balcony_or_terrace: bool | None = None
    view: str = "none"
    renovation_year: int | None = None
    is_furnished: bool | None = None
    is_temporary: bool | None = None
    parking: str = "none"
    is_minergie: bool | None = None
    pets_allowed: bool | None = None
    has_garden: bool | None = None
    has_own_washer: bool | None = None
    is_new_build: bool | None = None
    is_attic: bool | None = None
    area_sqm: float | None = None
    rooms: float | None = None
    rent_regime: str = "market"
    evidence: dict[str, str] = field(default_factory=dict)
    extractor: str = "none"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict (fields, ``evidence``, ``extractor``)."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, object], *, extractor: str | None = None
    ) -> "ExtractedAttributes":
        """Build from a (possibly messy) mapping, coercing types and dropping unknown keys.

        Args:
            data: Mapping with schema fields and optional ``evidence``/``extractor``.
            extractor: Overrides ``data["extractor"]``.

        Returns:
            The validated attributes.
        """
        values = {
            name: _coerce(name, kind, data.get(name)) for name, (kind, _) in _FIELD_SPECS.items()
        }
        raw_evidence = data.get("evidence")
        evidence = {
            key: value.strip()
            for key, value in (raw_evidence.items() if isinstance(raw_evidence, Mapping) else [])
            if key in _FIELD_SPECS and isinstance(value, str) and value.strip()
        }
        name = extractor if extractor is not None else str(data.get("extractor") or "none")
        return cls(**values, evidence=evidence, extractor=name)


def _coerce(name: str, kind: str, value: object) -> object:
    if kind == "enum":
        text = str(value).strip().lower() if value is not None else ""
        return text if text in _ENUMS[name] else _ENUMS[name][0]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if kind == "boolean":
        if isinstance(value, bool | np.bool_):
            return bool(value)
        if isinstance(value, str):
            return _BOOL_WORDS.get(value.strip().lower())
        numeric = isinstance(value, int | float | np.integer | np.floating)
        return bool(value) if numeric and value in (0, 1) else None
    try:
        is_bool = isinstance(value, bool | np.bool_)
        number = None if is_bool else float(str(value).replace(",", "."))
    except ValueError:
        return None
    if number is None or not np.isfinite(number):
        return None
    return round(number) if kind == "integer" else number


# --- Rule-based baseline (rules in rentml._extraction_rules) ------------------------------------
def rule_based_extract(text: str | None, lang: str | None = None) -> ExtractedAttributes:
    """Extract attributes with multilingual regexes (DE/FR/IT) as a transparent baseline.

    Negated mentions ("kein Balkon", "pas d'ascenseur", "nicht möbliert") give ``False``; only
    offers of a room ("WG-Zimmer", "chambre en colocation") give ``rent_regime="shared_flat"``,
    not flats that are merely suitable for a shared flat.

    Args:
        text: Description (anonymised or raw).
        lang: Detected language; the patterns are multilingual, so it is only logged.

    Returns:
        Attributes with an evidence span (substring of ``text``) for every set field.
    """
    result = ExtractedAttributes(extractor="rules")
    if not isinstance(text, str) or not text.strip():
        return result
    logger.debug("Rule-based extraction (lang=%s, %d chars)", lang, len(text))
    for name, (value, span) in find_attributes(lower_keep_length(text)).items():
        setattr(result, name, value)
        result.evidence[name] = evidence_span(text, span)
    return result


# --- Frames and evaluation ----------------------------------------------------------------------
def _column_name(name: str) -> str:
    return name if name == "rent_regime" else f"attr_{name}"


def extract_frame(
    texts: pd.Series,
    extractor: Callable[[str], ExtractedAttributes],
    *,
    errors: str = "raise",
    include_evidence: bool = False,
) -> pd.DataFrame:
    """Apply an extractor to a Series of descriptions (each distinct text is extracted once).

    Args:
        texts: Descriptions (NA treated as "").
        extractor: E.g. :func:`rule_based_extract` or a ``ClaudeExtractor``.
        errors: "raise" or "ignore" (rows with an :class:`ExtractionError` become NA).
        include_evidence: Add an ``evidence`` column with the spans as JSON.

    Returns:
        DataFrame indexed like ``texts``: ``attr_<field>`` (booleans as 1.0/0.0/NaN, numbers as
        float, view/parking as categoricals), ``rent_regime`` and ``extractor``.

    Raises:
        ValueError: If ``errors`` is invalid.
    """
    if errors not in ("raise", "ignore"):
        raise ValueError("errors must be 'raise' or 'ignore'")
    clean = texts.astype("string").fillna("").astype(str)
    results: dict[str, dict[str, object]] = {}
    for text in pd.unique(clean):
        try:
            results[text] = extractor(text).to_dict()
        except ExtractionError as err:
            if errors == "raise":
                raise
            logger.warning("Extraction failed, row left empty: %s", err)
            results[text] = {}
    records = pd.DataFrame([results[t] for t in clean], index=texts.index)
    records = records.reindex(columns=[*ATTRIBUTE_FIELDS, "evidence", "extractor"])
    out = pd.DataFrame(index=texts.index)
    for name, (kind, _) in _FIELD_SPECS.items():
        values = records[name]
        if kind == "enum" and name != "rent_regime":
            out[_column_name(name)] = pd.Categorical(values, categories=_ENUMS[name])
        elif kind == "enum":
            out[_column_name(name)] = values
        else:
            out[_column_name(name)] = pd.to_numeric(values.astype(object), errors="coerce")
            out[_column_name(name)] = out[_column_name(name)].astype(float)
    out["extractor"] = records["extractor"]
    if include_evidence:
        out["evidence"] = records["evidence"].map(
            lambda ev: json.dumps(ev, ensure_ascii=False), na_action="ignore"
        )
    return out


def _parse_label(value: object, kind: str) -> object:
    if isinstance(value, bool | np.bool_):
        return float(value)
    if not isinstance(value, str):
        return value
    text = value.strip().lower()
    if text in _MISSING_WORDS:
        return None
    if kind == "boolean" and text in _BOOL_WORDS:
        return float(_BOOL_WORDS[text])
    return text.replace(",", ".")  # "3,5" -> 3.5


def _field_values(frame: pd.DataFrame, name: str, kind: str, label: str) -> pd.Series:
    column = next((c for c in (f"attr_{name}", name) if c in frame.columns), None)
    if column is None:
        raise KeyError(f"Column 'attr_{name}' or '{name}' not found in {label}")
    values = frame[column].astype(object)
    if kind == "enum":
        text = values.map(lambda v: str(v).strip().lower(), na_action="ignore")
        out = text.where(~text.isin(list(_NULL_LIKE)))
        bad = out.notna() & ~out.isin(list(_ENUMS[name]))
    else:
        parsed = values.map(lambda v: _parse_label(v, kind))
        out = pd.to_numeric(parsed, errors="coerce").astype(float)
        bad = out.isna() & parsed.notna()
    if bad.any():
        examples = sorted({str(v) for v in values[bad]})[:5]
        raise ValueError(f"{label} column {column!r} has values that are not {kind}: {examples}")
    return out


def evaluate_extraction(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    fields: list[str],
    by: pd.Series | None = None,
) -> pd.DataFrame:
    """Precision, recall and F1 per field (optionally per group, e.g. language).

    A prediction counts when it is set (not null, not "none"/"market"); it is correct when the
    gold value is set and equal (``area_sqm`` within 1 m²). A wrong value is both FP and FN.
    ``accuracy`` also counts agreeing nulls.

    Args:
        pred: Extracted frame (``attr_<field>`` or plain field columns).
        gold: Hand labels with the same column convention; rows aligned on the shared index.
            Booleans may be written as true/false, 1/0, ja/nein, oui/non or sì/no; blank,
            "none" or "n/a" mean not mentioned.
        fields: Fields to evaluate (names from :data:`ATTRIBUTE_FIELDS`).
        by: Optional grouping Series aligned by index.

    Returns:
        One row per field (and group) with n, support, n_pred, tp, fp, fn, precision, recall,
        f1, accuracy.

    Raises:
        KeyError: For unknown fields or missing columns.
        ValueError: If ``fields`` is empty or a column holds a value that is not a valid label
            (the message names the column).
    """
    if not fields:
        raise ValueError("fields must name at least one attribute")
    index = pred.index.intersection(gold.index)
    groups = pd.Series("all", index=index) if by is None else by.reindex(index)
    group_name = "group" if by is None or by.name is None else str(by.name)
    parts = [
        _field_counts(pred.loc[index], gold.loc[index], name, groups, group_name) for name in fields
    ]
    table = pd.concat(parts, ignore_index=True)
    table["precision"] = table["tp"] / table["tp"].add(table["fp"]).replace(0, np.nan)
    table["recall"] = table["tp"] / table["tp"].add(table["fn"]).replace(0, np.nan)
    pr_sum = (table["precision"] + table["recall"]).replace(0, np.nan)
    table["f1"] = 2 * table["precision"] * table["recall"] / pr_sum
    table["accuracy"] = table.pop("agree") / table["n"]
    return table if by is not None else table.drop(columns=group_name)


def _field_counts(
    pred: pd.DataFrame, gold: pd.DataFrame, name: str, groups: pd.Series, group_name: str
) -> pd.DataFrame:
    if name not in _FIELD_SPECS:
        raise KeyError(f"Unknown field {name!r}")
    kind = _FIELD_SPECS[name][0]
    p, g = _field_values(pred, name, kind, "pred"), _field_values(gold, name, kind, "gold")
    p_set, g_set = p.notna(), g.notna()
    if kind == "enum":
        equal = (p == g).astype(bool)
    else:
        tol = 1.0 if name == "area_sqm" else 1e-9
        equal = pd.Series(np.isclose(p.to_numpy(float), g.to_numpy(float), atol=tol), index=p.index)
    tp = p_set & g_set & equal
    flags = pd.DataFrame({"n": 1, "support": g_set, "n_pred": p_set, "tp": tp})
    flags["fp"], flags["fn"] = p_set & ~tp, g_set & ~tp
    flags["agree"], flags[group_name] = tp | (~p_set & ~g_set), groups
    counts = flags.groupby(group_name, dropna=False).sum(numeric_only=True).reset_index()
    counts.insert(0, "field", name)
    return counts


def __getattr__(name: str) -> object:
    """Resolve the LLM extractor names lazily from :mod:`rentml.extraction_llm`.

    Args:
        name: Attribute name.

    Returns:
        The object of that name from :mod:`rentml.extraction_llm`.

    Raises:
        AttributeError: For any other name.
    """
    if name in _LLM_EXPORTS:
        # Imported here to avoid a circular import (extraction_llm builds on this module).
        from rentml import extraction_llm

        return getattr(extraction_llm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
