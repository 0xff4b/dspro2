"""Parsing of Swiss address strings into postcode, locality, street and house number.

Split out of :mod:`rentml.geo`, which re-exports :func:`parse_address` so
``rentml.geo.parse_address`` keeps working.
"""

import re
import unicodedata

import pandas as pd

_CH = r"(?:CH\s*-\s*)?"
_POSTCODE_ONLY_RE = re.compile(rf"^{_CH}([1-9]\d{{3}})$", re.IGNORECASE)
_POSTCODE_LEAD_RE = re.compile(rf"^{_CH}([1-9]\d{{3}})\s*-?\s+(\D.*)$", re.IGNORECASE)
_POSTCODE_INLINE_RE = re.compile(rf"^(.*?\S)\s+{_CH}([1-9]\d{{3}})\s*-?\s+(\D.*)$", re.I)
_POSTCODE_TRAIL_RE = re.compile(rf"^(\D.*?)\s+{_CH}([1-9]\d{{3}})$", re.IGNORECASE)
# "12", "12a", "9-13a", "20/22/24", "25 bis"
_NUMBER = r"\d{1,4}\s?[a-z]?(?:\s?[-/]\s?\d{1,4}\s?[a-z]?)*(?:\s?(?:bis|ter))?"
_HOUSE_NUMBER_RE = re.compile(rf"^{_NUMBER}$", re.IGNORECASE)
_STREET_NO_RE = re.compile(rf"^(.*\D)\s+({_NUMBER})$", re.IGNORECASE)
_NO_STREET_RE = re.compile(rf"^({_NUMBER})[\s,]+(\D.*)$", re.IGNORECASE)
_NUMBER_PATTERNS = ((_STREET_NO_RE, 2, 1), (_NO_STREET_RE, 1, 2))
_DIGIT_RE = re.compile(r"\d")
_COUNTRY_TOKENS = frozenset({"schweiz", "suisse", "svizzera", "switzerland", "ch"})


def parse_address(address: pd.Series) -> pd.DataFrame:
    """Split address strings into postcode, locality, street and house number.

    Handles the DSPRO1 format ``"8044, Gockhausen-Zürich, Rütistrasse, 1"`` and portal variants
    (``"Street 3, 1012 Lausanne"``, ``"Street **, Muttenz 4132, Schweiz"``, ``"Street 1 4058
    Basel"``, ``"Street **, Street 5, Zurich"``, bare localities; ``**`` = masked number).

    Args:
        address: Raw address strings (may contain missing values or non-strings).

    Returns:
        ``postcode`` (``Int64``), ``locality``, ``street``, ``house_number`` (str or None).
    """
    parsed = [_parse_one(value) for value in address.to_numpy(dtype=object)]
    cols = ["postcode", "locality", "street", "house_number"]
    out = pd.DataFrame(parsed, index=address.index, columns=cols)
    out["postcode"] = pd.to_numeric(out["postcode"]).astype("Int64")
    return out


def _parse_one(text: object) -> tuple[int | None, str | None, str | None, str | None]:
    parts = _clean_parts(text) if isinstance(text, str) else []
    hit = _locate_postcode(parts)
    postcode, locality, street_parts = None, None, parts
    if hit is not None:
        idx, postcode, locality, prefix = hit
        street_parts = [*parts[:idx], *([prefix] if prefix else [])]
        if locality is None and idx + 1 < len(parts):
            locality = parts[idx + 1]
            street_parts = parts[2:] if idx == 0 else street_parts
        elif idx == 0 and not prefix:
            street_parts = parts[1:]
    if locality is None:
        locality, street_parts = _pop_locality(street_parts, allow_single=hit is None)
    return postcode, locality, *_split_street(street_parts)


def _clean_parts(text: str) -> list[str]:
    parts = [" ".join(raw.replace("*", " ").split()) for raw in text.split(",")]
    parts = [p for p in parts if p and p.casefold() not in _COUNTRY_TOKENS]
    folded = [_fold(p) for p in parts]
    bases = [_fold(m.group(1)) if (m := _STREET_NO_RE.match(p)) else "" for p in parts]
    # Portals repeat parts ("Street **, Street 5, Zurich"): drop exact repeats and a bare
    # street name that reappears later with its house number.
    return [p for i, p in enumerate(parts) if folded[i] not in folded[:i] + bases[i + 1 :]]


def _locate_postcode(parts: list[str]) -> tuple[int, int, str | None, str | None] | None:
    for idx, part in enumerate(parts):
        if match := _POSTCODE_ONLY_RE.match(part):
            return idx, int(match.group(1)), None, None
        if match := _POSTCODE_LEAD_RE.match(part):
            return idx, int(match.group(1)), match.group(2).strip(), None
        if match := _POSTCODE_INLINE_RE.match(part):
            return idx, int(match.group(2)), match.group(3).strip(), match.group(1).strip()
    # "Muttenz 4132" only as a last resort: "Street 1234" looks the same.
    for idx, part in enumerate(parts):
        if match := _POSTCODE_TRAIL_RE.match(part):
            return idx, int(match.group(2)), match.group(1).strip(), None
    return None


def _pop_locality(parts: list[str], *, allow_single: bool) -> tuple[str | None, list[str]]:
    if len(parts) == 1 and not allow_single:
        return None, parts
    for i in (-1, 0):  # locality is the last or the first part without digits
        if parts and not _DIGIT_RE.search(parts[i]):
            return parts[i], parts[:-1] if i else parts[1:]
    return None, parts


def _split_street(parts: list[str]) -> tuple[str | None, str | None]:
    while len(parts) > 1 and not _DIGIT_RE.search(parts[-1]) and _STREET_NO_RE.match(parts[-2]):
        parts = parts[:-1]  # trailing locality repetition after "Street 12"
    if not parts:
        return None, None
    number = None
    if len(parts) > 1 and _HOUSE_NUMBER_RE.match(parts[-1]):
        number, parts = parts[-1], parts[:-1]
    street = ", ".join(parts)
    if _HOUSE_NUMBER_RE.match(street):
        return None, number or street
    for pattern, num_group, street_group in _NUMBER_PATTERNS:
        # The separate number part wins over one embedded in the street ("Rue X 14, 2A"),
        # so the street name matches other listings of the same street.
        if match := pattern.match(street):
            return match.group(street_group).strip(), number or match.group(num_group)
    return street, number


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if ch.isalnum()).casefold()
