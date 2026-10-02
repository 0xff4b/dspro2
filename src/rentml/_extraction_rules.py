"""Regex rules of the rule-based extractor for :mod:`rentml.extraction` (private helpers).

All patterns are lower case and run on lower-cased text (much faster than ``re.IGNORECASE``);
:func:`lower_keep_length` keeps the length, so match spans index the original text for the
evidence. A mention preceded by a negation in the last three words of its clause ("kein Balkon",
"pas d'ascenseur", "nicht befristet") or followed by ": nein" counts as false (booleans) or is
skipped (view, parking, rent regime). Use :func:`rentml.extraction.rule_based_extract`.
"""

import re
from collections.abc import Callable
from functools import partial

from rentml.config import REFERENCE_YEAR
from rentml.text import KEYWORD_PATTERNS

Span = tuple[int, int]
Found = tuple[object, Span] | None  # (value, evidence span in the text)

_ROOMS_RE = re.compile(
    r"(?<![\d.,'’/])(?P<num>\d{1,2}(?:[.,][05])?)(?P<half>\s?(?:½|1/2))?\s*-?\s*"
    r"(?:zimmer|zi\b\.?|pi[èe]ces?|pces?\b|locali|locale|vani|rooms?)"
)
_STUDIO_RE = re.compile(r"\b(?:studio|monolocale|einzimmerwohnung)\b")
_AREA_RE = re.compile(
    r"(?<![\d.,'’])(?P<num>\d{2,3}(?:[.,]\d{1,2})?)\s*"
    r"(?:m2|m²|m\s2\b|qm|sqm|mq|quadratmeter|mètres carrés|metri quadrati)"
)
_AREA_CUE_RE = re.compile(r"wohnfl|fl[äa]che|surface|superficie|gr[öo](?:ss|ß)e|living")
_AREA_SKIP_RE = re.compile(
    r"garten|terrass|balkon|sitzplatz|grundst|keller|estrich|hobby|bastel|lager|büro|atelier|"
    r"jardin|balcon|cave|terrain|parcelle|giardin|terrazz|cantina"
)
_FLAT_WORD_RE = re.compile(
    r"wohnung|whg|zimmer|appartement|pi[èe]ces|logement|appartamento|locali|alloggio|apartment"
)
_ORDINALS = {
    "erste": 1, "zweite": 2, "dritte": 3, "vierte": 4, "fünfte": 5, "premier": 1,
    "première": 1, "deuxième": 2, "troisième": 3, "quatrième": 4, "cinquième": 5,
    "primo": 1, "secondo": 2, "terzo": 3, "quarto": 4, "quinto": 5,
}  # fmt: skip
_FLOOR_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![\d.])(?P<n>\d{1,2})\.\s?(?:og|obergescho(?:ss|ß)|stock(?:werk)?\b|etage)"),
    re.compile(r"(?<![\d.])(?P<n>\d{1,2})\s?(?:e|er|ère|ème|eme)\s+[ée]tage"),
    re.compile(r"(?<![\d.])(?P<n>\d{1,2})\s?[°º]?\s?piano\b"),
    # Look-behind: "erste" inside "obersten"/"untersten" is no ordinal.
    re.compile(
        rf"(?<![a-zäöüéè])(?P<w>{'|'.join(_ORDINALS)})[nmr]?\s+(?:stock|obergescho|[ée]tage|piano)"
    ),
    re.compile(
        r"(?P<zero>erdgescho(?:ss|ß)|\beg\b|parterre|rez-de-chauss[ée]e|piano terr[ae]|"
        r"pianterreno)"
    ),
)
_CLAUSE_END_RE = re.compile(r"[,;:\n!?()]|\.\s")
_RENO = r"(?:r[ée]nov\w*|\w*saniert|sanierung|moderni[sz]iert|ristruttur\w*|rinnov\w*|umgebaut)"
_NOT_BUILT = r"(?!baujahr|erbaut|gebaut|construit|construction|costruit|built)"
_RENO_AFTER_RE = re.compile(rf"{_RENO}(?:{_NOT_BUILT}[^.\n\d]){{0,30}}?(?P<y>(?:19|20)\d{{2}})\b")
_RENO_BEFORE_RE = re.compile(rf"\b(?P<y>(?:19|20)\d{{2}})[^.,;\n\d]{{0,15}}?{_RENO}")
_NEGATION_RE = re.compile(
    r"\b(?:kein\w*|nicht|ohne|pas|sans|non|aucun\w*|nessun\w*|senza|niente|no|not|without)\b"
)
# Punctuation and contrast words end the scope of a negation ("ohne Lift, aber mit Balkon").
_NEGATION_RESET_RE = re.compile(
    r"[,;.:!?()\n]|\b(?:aber|jedoch|sondern|nur|mit|dafür|mais|avec|seulement|ma|con|solo|but|"
    r"with|only)\b"
)
# "Lift: nein", "Genossenschaft? Nein", "Lift nicht vorhanden" (not "vue sur le lac non loin").
_POST_NEGATION_RE = re.compile(
    r"\s*(?:[:?]\s*(?:nein|non|no|keine?)\b|nicht vorhanden|pas disponible|non disponibile)"
)
_WORD_RE = re.compile(r"\S+")
# Explicit negative phrases, checked before any positive mention.
_NEGATIVE: dict[str, str] = {
    "has_lift": r"(?:ohne|kein(?:en)?)\s+(?:lift|aufzug)|sans ascenseur|senza ascensore",
    "is_furnished": r"unm(?:ö|oe|o)bliert|nicht\s+m(?:ö|oe|o)bliert|non[- ]meubl|pas meubl"
    r"|non[- ]arredat|unfurnished",
    "is_temporary": r"unbefristet|nicht\s+befristet|dur[ée]e ind[ée]termin|indeterminat",
    "pets_allowed": r"keine (?:haus)?tiere|tiere\s+(?:sind\s+)?nicht\s+(?:erlaubt|gestattet)|"
    r"pas d'animaux|animaux (?:non|pas|interdit)|(?:niente|nessun|no)\s+animali|"
    r"animali non|no pets",
    "has_own_washer": r"gemeinsame\w*\s+waschk|waschk[üu]che zur mitben|buanderie commune|"
    r"lavanderia comune|shared laundry",
}
_POSITIVE: dict[str, str] = {
    "has_lift": KEYWORD_PATTERNS["lift"],
    "has_balcony_or_terrace": f"{KEYWORD_PATTERNS['balcony']}|{KEYWORD_PATTERNS['terrace']}",
    "is_furnished": KEYWORD_PATTERNS["furnished"],
    "is_temporary": KEYWORD_PATTERNS["temporary"],
    "is_minergie": KEYWORD_PATTERNS["minergie"],
    "pets_allowed": r"tiere\s+(?:sind\s+)?(?:erlaubt|willkommen|gestattet)|katzen erlaubt|"
    r"animaux (?:admis|accept|bienvenu|autoris)|animali (?:ammessi|benvenuti|consentiti)|"
    r"pets? (?:allowed|welcome)",
    "has_garden": KEYWORD_PATTERNS["garden"],
    "has_own_washer": KEYWORD_PATTERNS["own_washer"],
    "is_new_build": KEYWORD_PATTERNS["new_build"],
    "is_attic": KEYWORD_PATTERNS["attic"],
}
_FLAG_RULES: dict[str, tuple[re.Pattern[str] | None, re.Pattern[str]]] = {
    name: (re.compile(_NEGATIVE[name]) if name in _NEGATIVE else None, re.compile(pattern))
    for name, pattern in _POSITIVE.items()
}
# Only offers of a room count as shared_flat; "ideal für eine WG", "WG möglich", "keine WG" or
# "colocation possible" describe a whole flat on the market.
_SHARED_ROOM = (
    r"\bwg[- ]?zimmer|zimmer in (?:einer |unserer |der )?(?:\d+er[- ]?)?(?:wg|wohngemeinschaft)\b"
    r"|zimmer (?:in|zur) untermiete|untermiete (?:eines|für ein|von einem) zimmer"
    r"|mitbewohner(?:in)? gesucht|such\w* (?:eine[nr]? )?(?:neue[nr]? )?mitbewohner"
    r"|chambre (?:en|dans une?) colocation|colocataire recherch|cherch\w* (?:une? )?colocataire"
    r"|(?:camera|stanza) in (?:appartamento condiviso|condivisione)|cerc\w* (?:una? )?coinquilin"
    r"|room in (?:a )?(?:shared flat|flatshare)|flatmate wanted"
)
# field -> ordered (value, pattern) rules; the first non-negated match wins.
_ENUM_RULES: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    name: tuple((value, re.compile(pattern)) for value, pattern in rules)
    for name, rules in {
        "view": (
            ("lake", KEYWORD_PATTERNS["lake_view"]),
            ("mountain", KEYWORD_PATTERNS["mountain_view"]),
            ("city", r"stadtblick|sicht auf die stadt|vue sur la ville|vista sulla citt|city view"),
            ("other", r"aussicht|weitsicht|fernsicht|panorama|vue d[ée]gag|vista aperta"),
        ),
        "parking": (
            ("garage", KEYWORD_PATTERNS["garage"]),
            ("outdoor", KEYWORD_PATTERNS["parking"]),
        ),
        "rent_regime": (
            ("shared_flat", _SHARED_ROOM),
            ("cooperative", KEYWORD_PATTERNS["cooperative"]),
            ("cost_based_or_subsidised", KEYWORD_PATTERNS["subsidised"]),
        ),
    }.items()
}


def lower_keep_length(text: str) -> str:
    """Lower-case ``text`` without changing its length (so spans stay valid)."""
    lowered = text.lower()
    if len(lowered) == len(text):
        return lowered
    return "".join(c if len(c.lower()) != 1 else c.lower() for c in text)  # e.g. "İ" -> 2 chars


def evidence_span(text: str, span: Span) -> str:
    """Return ``text[span]`` widened to whole words ("giardin" -> "giardino")."""
    start, end = span
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return text[start:end].strip()


def _negated_span(text: str, match: re.Match[str]) -> Span | None:
    lo = max(0, match.start() - 60)
    clause_start = max(
        (m.end() for m in _NEGATION_RESET_RE.finditer(text, lo, match.start())), default=lo
    )
    words = [m.start() for m in _WORD_RE.finditer(text, clause_start, match.start())]
    first = words[-3] if len(words) >= 3 else clause_start
    if (negation := _NEGATION_RE.search(text, first, match.start())) is not None:
        return negation.start(), match.end()
    end = match.end()
    while end < len(text) and text[end].isalnum():
        end += 1
    post = _POST_NEGATION_RE.match(text, end)
    return (match.start(), post.end()) if post else None


def _find_flag(text: str, name: str) -> Found:
    negative, positive = _FLAG_RULES[name]
    if negative is not None and (match := negative.search(text)) is not None:
        return False, match.span()
    negated: Span | None = None
    for match in positive.finditer(text):
        span = _negated_span(text, match)
        if span is None:
            return True, match.span()
        negated = negated or span
    return None if negated is None else (False, negated)


def _find_enum(text: str, name: str) -> Found:
    for value, pattern in _ENUM_RULES[name]:
        for match in pattern.finditer(text):
            if _negated_span(text, match) is None:
                return value, match.span()
    return None


def _find_rooms(text: str) -> Found:
    for match in _ROOMS_RE.finditer(text):
        rooms = float(match["num"].replace(",", ".")) + (0.5 if match["half"] else 0.0)
        if 1.0 <= rooms <= 15.0:
            return rooms, match.span()
    studio = _STUDIO_RE.search(text)
    return (1.0, studio.span()) if studio else None


def _find_area(text: str) -> Found:
    fallback: Found = None
    amenity: Found = None
    for match in _AREA_RE.finditer(text):
        area = float(match["num"].replace(",", "."))
        if not 10.0 <= area <= 500.0:
            continue
        # Context = the clause before the number, so "Balkon 10 m2, Wohnfläche 85 m2" works.
        context = re.split(r"[,;\n(]", text[max(0, match.start() - 30) : match.start()])[-1]
        skip = _AREA_SKIP_RE.search(context)
        if skip is None and _AREA_CUE_RE.search(context):
            return area, match.span()
        if skip is None:
            fallback = fallback or (area, match.span())
        elif area >= 40.0 and _FLAT_WORD_RE.search(context, 0, skip.start()):
            # "3.5-Zimmer-Wohnung mit Balkon 85 m2": too large for a balcony, a flat word first.
            amenity = amenity or (area, match.span())
    return fallback or amenity


def _clause(text: str, match: re.Match[str]) -> str:
    lo = max(0, match.start() - 80)
    start = max((m.end() for m in _CLAUSE_END_RE.finditer(text, lo, match.start())), default=lo)
    end = _CLAUSE_END_RE.search(text, match.end())
    return text[start : end.start() if end else len(text)]


def _find_floor(text: str) -> Found:
    matches = [m for pattern in _FLOOR_RES for m in pattern.finditer(text)]
    if not matches:
        return None

    def rank(match: re.Match[str]) -> tuple[bool, bool, int]:
        # The clause that names the flat first, then numbered floors before the ground floor
        # ("Waschküche im EG, Wohnung im 3. OG" -> 3), then the first mention.
        has_flat = _FLAT_WORD_RE.search(_clause(text, match)) is not None
        return not has_flat, "zero" in match.groupdict(), match.start()

    best = min(matches, key=rank)
    groups = best.groupdict()
    if groups.get("n"):
        return int(groups["n"]), best.span()
    return _ORDINALS.get(groups.get("w") or "", 0), best.span()


def _find_renovation_year(text: str) -> Found:
    for pattern in (_RENO_AFTER_RE, _RENO_BEFORE_RE):
        for match in pattern.finditer(text):
            year = int(match["y"])
            if 1950 <= year <= REFERENCE_YEAR + 1:
                return year, match.span()
    return None


_FINDERS: dict[str, Callable[[str], Found]] = {
    "rooms": _find_rooms,
    "area_sqm": _find_area,
    "floor": _find_floor,
    "renovation_year": _find_renovation_year,
    **{name: partial(_find_flag, name=name) for name in _FLAG_RULES},
    **{name: partial(_find_enum, name=name) for name in _ENUM_RULES},
}


def find_attributes(lowered: str) -> dict[str, tuple[object, Span]]:
    """Run every rule on lower-cased text.

    Args:
        lowered: Output of :func:`lower_keep_length`.

    Returns:
        Field name -> (value, evidence span) for every field a rule found.
    """
    found = {name: finder(lowered) for name, finder in _FINDERS.items()}
    return {name: value for name, value in found.items() if value is not None}
