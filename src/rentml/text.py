"""Listing descriptions: anonymisation, language, keywords, attribute extraction and text vectors.

:func:`anonymize` removes contact data (revDSG) and every rent figure, so no model can read the
target from the text. Placeholders: [EMAIL] [URL] [PHONE] [PRICE] [NAME].
"""

import hashlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline

from rentml.config import AREA_RANGE, RANDOM_STATE, REFERENCE_YEAR, ROOMS_RANGE

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_I = re.IGNORECASE

# fmt: off
# Amounts: 1'850 / 2 100 / 1.850 (exactly three digits after the separator) or 2-6 plain digits,
# optional decimals and the Swiss "1850.-" notation.
_NUM = r"(?:\d{1,3}(?:['’ .,]\d{3})+(?!\d)|\d{2,6})"
_AMOUNT = rf"{_NUM}(?:[.,]\d{{1,2}})?(?![\d'’])(?:\s*[.,]\s*[-–—]+)?"
_CUR = r"(?:CHF|SFr\.?|Fr\.|Fr\b|Franken|francs?|franchi|EUR|Euro|€)"
_START = r"(?<![\w'’.,])"
_PRICE_RES = [re.compile(p, _I) for p in (
    # Atomic group: "Fr. 10-12 Uhr" is Friday, not francs.
    rf"(?>{_CUR}\s*:?\s*{_AMOUNT}(?:\s*(?:-|–|bis|à)\s*{_AMOUNT})?)(?!\s*(?:uhr|h\b|heures?))",
    rf"{_START}{_AMOUNT}\s*{_CUR}",
    rf"{_START}{_NUM}\s*[.,]\s*[-–—]+(?!\w)",
    rf"{_START}{_AMOUNT}(?=\s*(?:/\s*|(?:pro|par|per|al|im|a)\s+)(?:mte?\b|monat|mois|mese|month)"
    r"|\s+(?:monatlich|mensuel|mensil|netto|brutto|inkl|exkl|zzgl))",
)]
_CUE_PRICE_RE = re.compile(
    r"(?P<cue>(?<!\[)\b(?:\w*(?:miete|mietzins|preis|kosten|depot|kaution)|nk|hk|akonto|pauschale"
    r"|garantie|netto|brutto|total|\w*parkplatz|einstellplatz|garage|loyer|prix|charges|frais"
    r"|acompte|caution|dépôt|place de parc|parking|affitto|pigione|canone|prezzo|spese|deposito"
    r"|cauzione|posteggio|rent|price|deposit)\b(?P<gap>[^\d\n\[]{0,30}?))"
    rf"(?P<amount>{_START}{_AMOUNT})(?!\s*(?:m2|m²|qm|%|zimmer|zi\b|pi[èe]ces?|locali|jahre?|ans?\b"
    r"|anni))", _I)
_YEAR_CUE_RE = re.compile(
    r"\b(?:baujahr|renov|rénov|sanier|erbaut|gebaut|bezug|constru|ristrutt|costru|seit|depuis|"
    r"since|anno|dal|jan|feb|mär|apr|mai|jun|jul|aug|sep|okt|oct|nov|dez|déc|dic|janv|févr|mars|"
    r"avr|juin|juil|août|genn|febb|marz|magg|giu|lug|ago|sett|ott)", _I)
_EMAIL_RE = re.compile(r"[\w.+-]{1,64}\s?(?:@|\(at\)|\[at\])\s?[\w-]{1,63}(?:\.[\w-]{1,63}){1,10}")
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"]*[^\s<>\".,;:!?)\]]|(?<![\w-])[\w-]{1,63}(?:\.[\w-]{1,63}){0,5}"
    r"\.(?:ch|com|net|org|swiss|info|li)\b(?:/[^\s<>\"]*[^\s<>\".,;:!?)\]])?", _I)
_PHONE_RE = re.compile(
    r"(?<![\w+'’.,])(?:\+|00)\s?[1-9]\d{0,2}(?:\s?\(0\))?(?:[\s./-]{0,2}\d){6,12}(?!\d)"
    r"|(?<![\w+'’.,])\(?0\d{2}\)?(?:[\s./-]{0,3}\d){7}(?!\d)")
_SALUTATION = (r"\b(?:Herrn?|Frau|Hr\.|Fr\.|Monsieur|Madame|Mme\.?|Mlle\.?|M\.|Signor[ae]?"
               r"|Sig\.(?:ra)?|Mrs?\.?|Ms\.?)")
_CONTACT = r"\b(?i:kontakt(?:person)?|ansprechp\w+|contact|personne de contact|contatto|referente)"
_WORD = r"(?:(?:von|van|de|di|da|del|della|du|le|la)[ \t]+)?[A-ZÄÖÜÀÂÇÈÉÊÎÔÙÛ][\w'’-]+"
_NAME_RE = re.compile(rf"(?P<lead>(?:{_SALUTATION}|{_CONTACT}\s*:?)[ \t]+(?:{_SALUTATION}[ \t]+)?"
                      rf"(?:Dr\.[ \t]*)?)(?!{_SALUTATION}(?!\w)){_WORD}(?:[ \t]+{_WORD}){{0,2}}")
_ANY_NUMBER_RE = r"(?<![\d'’.,])(\d{1,3}(?:['’ .,]\d{3})+(?!\d)|\d+)(?:[.,](\d{1,2}))?(?!\d)"
# fmt: on
# Kept years and postcodes are hidden as private-use digits until the cue pass is done, so a
# price behind them is still found ("Miete ab Juli 2026: 1850").
_HIDE = str.maketrans({str(d): chr(0xE000 + d) for d in range(10)})
_SHOW = str.maketrans({chr(0xE000 + d): str(d) for d in range(10)})


def _cue_price(match: re.Match[str]) -> str:
    amount, gap = match["amount"], match["gap"]
    is_year = re.fullmatch(r"(?:19|20)\d\d", amount) and _YEAR_CUE_RE.search(gap)
    # "Parkplatz an der Bahnhofstrasse, 8001 Zürich": a postcode, but "Miete 1850 Nebenkosten".
    is_postcode = re.fullmatch(r"\d{4}", amount) and re.search(r"(?:,|\bin|\bà|\bCH-)\s*$", gap)
    is_postcode = is_postcode and re.match(
        r"[ \t]+[A-ZÄÖÜ][a-zäöüéèà]", match.string[match.end() :]
    )
    return match["cue"] + (amount.translate(_HIDE) if is_year or is_postcode else "[PRICE]")


def anonymize(text: str | None) -> str:
    """Replace contact data and amounts by placeholders; keep rooms, area, years and postcodes."""
    if not isinstance(text, str):
        return ""
    # Order matters: e-mails and URLs contain digits and dots that look like phones or prices.
    out = _PHONE_RE.sub("[PHONE]", _URL_RE.sub("[URL]", _EMAIL_RE.sub("[EMAIL]", text)))
    for pattern in _PRICE_RES:
        out = pattern.sub("[PRICE]", out)
    while (step := _CUE_PRICE_RE.sub(_cue_price, out)) != out:
        out = step
    return _NAME_RE.sub(r"\g<lead>[NAME]", out.translate(_SHOW)).strip()


def anonymize_series(texts: pd.Series) -> pd.Series:
    """Anonymise descriptions; missing or blank texts become ``<NA>`` (``string`` dtype)."""
    out = texts.map(anonymize, na_action="ignore").astype("string")
    return out.mask(out.fillna("").str.len() == 0)


def price_leak_mask(anon: pd.Series, price: pd.Series, rel_tol: float = 0.03) -> pd.Series:
    """Flag texts that still contain a number within ``±rel_tol`` of the listing's own rent."""
    parts = anon.astype("string").fillna("").str.extractall(_ANY_NUMBER_RE)
    if parts.empty:
        return pd.Series(False, index=anon.index)
    number = pd.to_numeric(parts[0].str.replace(r"\D", "", regex=True) + "." + parts[1].fillna("0"))
    rent = pd.to_numeric(price, errors="coerce").reindex(parts.index.get_level_values(0)).to_numpy()
    hits = pd.Series(np.abs(number.to_numpy() - rent) <= rel_tol * rent, index=parts.index)
    return hits.groupby(level=0).any().reindex(anon.index, fill_value=False).astype(bool)


def price_leak_rate(anon: pd.Series, price: pd.Series, rel_tol: float = 0.03) -> float:
    """Share of listings with text and a positive rent that :func:`price_leak_mask` flags."""
    rent = pd.to_numeric(price, errors="coerce").reindex(anon.index)
    valid = anon.astype("string").fillna("").str.strip().ne("") & rent.gt(0)
    return float(price_leak_mask(anon, price, rel_tol)[valid].mean()) if valid.any() else np.nan


# Only discriminative stop words; "la", "le", "in", "per", "da", "un" occur in several languages.
_STOPWORDS = {
    lang: set(words.split())
    for lang, words in {
        "de": "der die das und ist mit von zu im den dem ein eine einer nicht auf für sich auch "
        "wir bei nach aus wird sind oder zum zur sehr wohnung zimmer küche miete",
        "fr": "et est avec du une pour dans au aux sur par pas vous nous sont très appartement "
        "pièces cuisine loyer ce cette les à quartier proche",
        "it": "lo gli è con di del della dei una nel nella sono molto appartamento locali cucina "
        "affitto che al alla delle vicino",
    }.items()
}


def detect_language(text: str | None) -> str:
    """Return "de", "fr", "it" or "unknown" (no stop word, or a tie) by a stop-word vote."""
    tokens = re.findall(r"[^\W\d_]+", text.lower()) if isinstance(text, str) else []
    scores = sorted((sum(t in words for t in tokens), lang) for lang, words in _STOPWORDS.items())
    (second, _), (best, lang) = scores[-2], scores[-1]
    return lang if best > second else "unknown"


# fmt: off
# Lower-case regexes (DE/FR/IT/EN), applied case-insensitively.
KEYWORD_PATTERNS: dict[str, str] = {
    "balcony": r"balkon|balcon|loggia", "terrace": r"terrass|terrazz|sitzplatz|patio",
    "lift": r"(?<!ski)(?<!sessel)lift\b|aufzug|fahrstuhl|ascenseur|ascensore|elevator",
    "lake_view": r"see-?\s?(?:sicht|blick)|sicht auf den \w*see|vue\W+(?:\w+\W+){0,3}lac\b"
                 r"|vista\W+(?:\w+\W+){0,3}lago\b|lake view",
    "mountain_view": r"berg(?:sicht|blick)|alpen(?:sicht|blick|panorama)|sicht auf die (?:berge"
                     r"|alpen)|vue\W+(?:\w+\W+){0,3}(?:montagnes?|alpes)\b|vista\W+(?:\w+\W+)"
                     r"{0,3}(?:montagn[ae]|monti|alpi)\b|mountain view",
    "renovated": r"r[ée]nov|saniert|sanierung|moderni[sz]iert|ristruttur|rinnov",
    "new_build": r"neubau|erstbezug|nouvelle construction|construction neuve|premi[èe]re location"
                 r"|immeuble neuf|appartement neuf|nuova costruzione|prima locazione|new build",
    "furnished": r"(?<!un)(?<!nicht )m(?:ö|oe|o)bliert|(?<!non )(?<!non-)(?<!pas )meubl[ée]"
                 r"|(?<!non )(?<!non-)arredat|(?<!un)furnished",
    "temporary": r"(?<!un)(?<!nicht )befristet|tempor[äa]r|zwischenmiete|auf zeit\b|dur[ée]e "
                 r"d[ée]termin[ée]e|temporaire|courte dur[ée]e|temporane|tempo determinato"
                 r"|breve periodo|temporary",
    "parking": r"parkpl[äa]tz|abstellplatz|places? de parc|parking|posteggi|parcheggi|posto auto",
    "garage": r"garage|einstellhalle|einstellplatz|tiefgarage|autorimessa|box auto",
    "minergie": r"minergie", "loft": r"\bloft", "quiet": r"ruhig|calme|tranquill|silenzios|quiet",
    "pets": r"haustier|\bkatzen?\b|\bhunde?\b|animaux|animali|\bchats?\b|\bchiens?\b|\bpets?\b",
    "garden": r"(?<!kinder)garten(?!\w*(?:strasse|str\b|weg\b|gasse))|jardin|giardin|garden",
    "dishwasher": r"geschirrsp[üu]e?l|abwaschmaschine|sp[üu]lmaschine|lave-vaisselle"
                  r"|lavastoviglie|dishwasher",
    "own_washer": r"waschturm|eigene\w*\s+(?:waschmaschine|wm\b|waschk[üu]che)|wm\s*/\s*(?:tumbler"
                  r"|tr)\b|waschmaschine\s*(?:und|/|&)\s*tumbler|colonne de lavage|(?:lave-linge"
                  r"|machine [àa] laver)\s+(?:priv|individ|dans l)|colonna (?:di )?lavaggio"
                  r"|lavatrice\s+(?:propri|privat|in appartamento)|own washer",
    "fireplace": r"kamin|schwedenofen|chemin[ée]e|caminett|\bcamino\b|fireplace",
    "attic": r"dachwohnung|dachgescho|dachstock|mansard|attika|combles|sous les toits|attique"
             r"|sottotetto|attico|attic|penthouse",
    "cooperative": r"genossenschaft|coop[ée]rative|cooperativa|anteilschein|parts? sociales?"
                   r"|quot[ae] social[ei]",
    "subsidised": r"subventio|verbillig|kostenmiete|gemeinn(?:ü|ue)tzig|loyer mod[ée]r|\blup\b"
                  r"|sussidi|pigione moderata|canone moderato|alloggi popolari",
    "shared_flat": r"\bwg\b(?![-\s]?(?:tauglich|geeignet))|wohngemeinschaft|mitbewohner"
                   r"|untermiet|colocat|\bcoloc\b|coinquilin|appartamento condiviso|flatshare",
    "wheelchair": r"rollstuhl|hindernisfrei|behindertengerecht|barrierefrei|fauteuil roulant"
                  r"|sans obstacle|sedia a rotelle|senza barriere|wheelchair",
}
ENUM_VALUES = {"view": ("none", "lake", "mountain", "city", "other"),
          "parking": ("none", "outdoor", "garage"),
          "rent_regime": ("market", "cooperative", "cost_based_or_subsidised", "shared_flat")}
ATTRIBUTE_FIELDS: dict[str, str] = {  # field -> JSON type; enums default to their first value
    "floor": "integer", "has_lift": "boolean", "has_balcony_or_terrace": "boolean", "view": "enum",
    "renovation_year": "integer", "is_furnished": "boolean", "is_temporary": "boolean",
    "parking": "enum", "is_minergie": "boolean", "pets_allowed": "boolean", "has_garden": "boolean",
    "has_own_washer": "boolean", "is_new_build": "boolean", "is_attic": "boolean",
    "area_sqm": "number", "rooms": "number", "rent_regime": "enum",
}
# A negation in the same clause before a mention ("kein Balkon", "pas d'ascenseur") or a "nein"
# right after it negates it; contrast words end the scope ("ohne Lift aber mit Balkon").
_NEG_BEFORE_RE = re.compile(
    r"\b(?:kein\w*|nicht|ohne|pas|sans|non|aucun\w*|nessun\w*|senza|no|not|without)\b"
    r"(?:(?!\b(?:aber|jedoch|mit|mais|avec|ma|con|but|with)\b)[^,;.:!?()\n]){0,25}$", _I)
_NEG_AFTER_RE = re.compile(
    r"\w*\s*(?:[:?]\s*(?:nein|non|no|keine?)\b|nicht vorhanden|pas disponible|non disponibile)", _I)
_KW = KEYWORD_PATTERNS
# Boolean field -> (pattern for true, explicit wording for false that no negation word marks).
_FLAGS: dict[str, tuple[str, str | None]] = {
    "has_lift": (_KW["lift"], None), "is_minergie": (_KW["minergie"], None),
    "has_balcony_or_terrace": (f"{_KW['balcony']}|{_KW['terrace']}", None),
    "is_furnished": (_KW["furnished"], r"unm(?:ö|oe|o)bliert|unfurnished"),
    "is_temporary": (_KW["temporary"], r"unbefristet|dur[ée]e ind[ée]termin|indeterminat"),
    "pets_allowed": (
        r"tiere\s+(?:sind\s+)?(?:erlaubt|willkommen|gestattet)|katzen erlaubt|animaux (?:admis"
        r"|accept|bienvenu|autoris)|animali (?:ammessi|benvenuti|consentiti)|pets? allowed",
        r"keine (?:haus)?tiere|tiere\s+(?:sind\s+)?nicht\s+(?:erlaubt|gestattet)|pas d'animaux"
        r"|animaux (?:non|pas|interdit)|(?:niente|nessun|no)\s+animali|animali non|no pets"),
    "has_garden": (_KW["garden"], None), "is_new_build": (_KW["new_build"], None),
    "has_own_washer": (_KW["own_washer"], r"gemeinsame\w*\s+waschk|waschk[üu]che zur mitben"
                       r"|buanderie commune|lavanderia comune|shared laundry"),
    "is_attic": (_KW["attic"], None),
}
_ENUM_RULES = {  # field -> ordered (value, pattern); the first non-negated match wins
    "view": [("lake", _KW["lake_view"]), ("mountain", _KW["mountain_view"]),
             ("city", r"stadtblick|sicht auf die stadt|vue sur la ville|vista sulla citt"),
             ("other", r"aussicht|weitsicht|fernsicht|panorama|vue d[ée]gag|vista aperta")],
    "parking": [("garage", _KW["garage"]), ("outdoor", _KW["parking"])],
    # Only offers of a room are shared_flat; "ideal für eine WG" describes a whole flat.
    "rent_regime": [
        ("shared_flat", r"\bwg[- ]?zimmer|zimmer in (?:einer |der )?(?:\d+er[- ]?)?(?:wg|wohngem"
         r"einschaft)\b|zimmer (?:in|zur) untermiete|mitbewohner(?:in)? gesucht|such\w* (?:eine[nr]"
         r"? )?(?:neue[nr]? )?mitbewohner|chambre (?:en|dans une?) colocation|colocataire recherch"
         r"|(?:camera|stanza) in (?:appartamento condiviso|condivisione)|cerc\w* (?:una? )?"
         r"coinquilin|room in (?:a )?(?:shared flat|flatshare)|flatmate wanted"),
        ("cooperative", _KW["cooperative"]), ("cost_based_or_subsidised", _KW["subsidised"])],
}
_ROOMS_RE = re.compile(r"(?<![\d.,'’/])(?P<num>\d{1,2}(?:[.,][05])?)(?P<half>\s?(?:½|1/2))?\s*-?\s*"
                       r"(?:zimmer|zi\b\.?|pi[èe]ces?|pces?\b|locali|locale|vani|rooms?)", _I)
_STUDIO_RE = re.compile(r"\b(?:studio|monolocale|einzimmerwohnung)\b", _I)
_AREA_RE = re.compile(r"(?<![\d.,'’])(?P<num>\d{2,3}(?:[.,]\d{1,2})?)\s*"
                      r"(?:m2|m²|qm|sqm|mq|quadratmeter|mètres carrés|metri quadrati)", _I)
_AREA_SKIP_RE = re.compile(r"garten|terrass|balkon|sitzplatz|grundst|keller|estrich|hobby|bastel"
                           r"|lager|büro|atelier|jardin|balcon|cave|terrain|parcelle|giardin"
                           r"|terrazz|cantina", _I)
_FLOOR_RE = re.compile(
    r"(?<![\d.])(?P<n>\d{1,2})(?:\.\s?(?:og\b|obergescho|stock|etage)|\s?(?:e|er|ère|ème|eme)\s+"
    r"[ée]tage|\s?[°º]?\s?piano\b)|erdgescho|\beg\b|parterre|rez-de-chauss|piano terr|pianterreno",
    _I)
_RENO = r"(?:r[ée]nov\w*|\w*saniert|sanierung|moderni[sz]iert|ristruttur\w*|rinnov\w*|umgebaut)"
_RENO_RE = re.compile(rf"{_RENO}(?:(?!baujahr|erbaut|gebaut|constru|costruit|built)[^.\n\d])"
                      rf"{{0,30}}?(?P<a>(?:19|20)\d\d)\b|\b(?P<b>(?:19|20)\d\d)[^.,;\n\d]{{0,15}}?"
                      rf"{_RENO}", _I)
# fmt: on
Found = tuple[object, tuple[int, int]] | None  # (value, evidence span)


def keyword_flags(texts: pd.Series) -> pd.DataFrame:
    """One ``int8`` column ``kw_<name>`` per :data:`KEYWORD_PATTERNS` entry (missing text = 0)."""
    lowered = texts.astype("string").fillna("").str.lower()
    flags = {f"kw_{k}": lowered.str.contains(p).astype("int8") for k, p in KEYWORD_PATTERNS.items()}
    return pd.DataFrame(flags, index=texts.index)


def _negation(text: str, match: re.Match[str]) -> tuple[int, int] | None:
    if before := _NEG_BEFORE_RE.search(text, max(0, match.start() - 40), match.start()):
        return before.start(), match.end()
    if after := _NEG_AFTER_RE.match(text, match.end()):
        return match.start(), after.end()
    return None


def _find_flag(text: str, field: str) -> Found:
    positive, negative = _FLAGS[field]
    if negative and (match := re.search(negative, text, _I)):
        return False, match.span()
    negated = None
    for match in re.finditer(positive, text, _I):
        if (span := _negation(text, match)) is None:
            return True, match.span()
        negated = negated or span
    return None if negated is None else (False, negated)


def _find_enum(text: str, field: str) -> Found:
    for value, pattern in _ENUM_RULES[field]:
        for match in re.finditer(pattern, text, _I):
            if _negation(text, match) is None:
                return value, match.span()
    return None


def _find_number(text: str, field: str) -> Found:
    if field == "rooms":
        for m in _ROOMS_RE.finditer(text):
            rooms = float(m["num"].replace(",", ".")) + 0.5 * bool(m["half"])
            if ROOMS_RANGE[0] <= rooms <= ROOMS_RANGE[1]:
                return rooms, m.span()
        return (1.0, studio.span()) if (studio := _STUDIO_RE.search(text)) else None
    if field == "area_sqm":
        for m in _AREA_RE.finditer(text):
            clause = re.split(r"[,;\n(]", text[max(0, m.start() - 30) : m.start()])[-1]
            area = float(m["num"].replace(",", "."))
            if AREA_RANGE[0] <= area <= AREA_RANGE[1] and not _AREA_SKIP_RE.search(clause):
                return area, m.span()
        return None
    if field == "floor":  # a numbered floor beats "EG" ("Waschküche im EG, Wohnung im 3. OG")
        matches = list(_FLOOR_RE.finditer(text))
        best = next((m for m in matches if m["n"]), matches[0] if matches else None)
        return None if best is None else (int(best["n"] or 0), best.span())
    for m in _RENO_RE.finditer(text):
        if 1950 <= (year := int(m["a"] or m["b"])) <= REFERENCE_YEAR + 1:
            return year, m.span()
    return None


def rule_based_extract(text: str | None) -> dict[str, Any]:
    """Regex baseline (DE/FR/IT): found fields plus ``evidence`` (field -> verbatim text span).

    Negated mentions ("kein Balkon", "pas d'ascenseur") give ``False``.
    """
    found: dict[str, Any] = {"evidence": {}}
    for field in ATTRIBUTE_FIELDS if isinstance(text, str) else ():
        finder = (
            _find_flag if field in _FLAGS else _find_enum if field in ENUM_VALUES else _find_number
        )
        if (hit := finder(text, field)) is None:
            continue
        found[field], (start, end) = hit
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        while end < len(text) and text[end].isalnum():
            end += 1
        found["evidence"][field] = text[start:end].strip()
    return found


class ExtractionError(RuntimeError):
    """The model returned no usable result (refusal or truncation)."""


def extract_frame(
    texts: pd.Series,
    extractor: Callable[[str], dict[str, Any]],
    include_evidence: bool = False,
    errors: str = "raise",
) -> pd.DataFrame:
    """Apply an extractor once per distinct text: ``attr_<field>``, ``rent_regime``, ``extractor``.

    Booleans become 1.0/0.0/NaN and view/parking categoricals. With ``errors="ignore"``, rows
    whose extraction raised :class:`ExtractionError` stay NA (``extractor`` included).
    """
    name = getattr(extractor, "name", None) or extractor.__name__
    clean = texts.astype("string").fillna("")
    results: dict[str, dict[str, Any]] = {}
    for text in clean.unique():
        try:
            results[text] = {**extractor(text), "extractor": name}
        except ExtractionError as err:
            if errors == "raise":
                raise
            logger.warning("Extraction failed, row left empty: %s", err)
            results[text] = {}
    records = pd.DataFrame([results[t] for t in clean], index=texts.index)
    records = records.reindex(columns=[*ATTRIBUTE_FIELDS, "evidence", "extractor"])
    out = pd.DataFrame(index=texts.index)
    for field, kind in ATTRIBUTE_FIELDS.items():
        values, dtype = records[field], float
        if kind == "enum":
            values = values.where(
                values.notna() | records["extractor"].isna(), ENUM_VALUES[field][0]
            )
            dtype = object if field == "rent_regime" else pd.CategoricalDtype(ENUM_VALUES[field])
        out[field if field == "rent_regime" else f"attr_{field}"] = values.astype(dtype)
    out["extractor"] = records["extractor"]
    if include_evidence:
        out["evidence"] = records["evidence"].map(json.dumps, na_action="ignore")
    return out


def _labels(frame: pd.DataFrame, field: str) -> pd.Series:
    column = f"attr_{field}" if f"attr_{field}" in frame.columns else field
    text = frame[column].astype("string").str.strip().str.lower()
    # Blank, "none" and "market" mean "not mentioned" in predictions and hand labels alike.
    text = text.mask(text.isin(["", "nan", "none", "null", "n/a", "-", "market"]))
    if ATTRIBUTE_FIELDS[field] == "enum":
        return text.astype(object)
    text = text.replace(dict(true="1", false="0", yes="1", no="0", ja="1", nein="0", oui="1"))
    return pd.to_numeric(text.str.replace(",", ".")).astype(float)


def evaluate_extraction(
    pred: pd.DataFrame, gold: pd.DataFrame, fields: Sequence[str], by: pd.Series | None = None
) -> pd.DataFrame:
    """Precision, recall and F1 per field (and group) on the rows shared by ``pred`` and ``gold``.

    A set prediction is a true positive only if it equals the set gold value (``area_sqm``
    within 1 m²), so a wrong value is FP and FN; ``accuracy`` also counts agreeing nulls.
    """
    index = pred.index.intersection(gold.index)
    groups = pd.Series("all", index=index) if by is None else by.reindex(index).fillna("unknown")
    rows = []
    for field in fields:
        p, g = _labels(pred.loc[index], field), _labels(gold.loc[index], field)
        if field == "area_sqm":
            p = p.where(~np.isclose(p, g, atol=1.0), g)
        y_pred, y_true = (s.astype(object).where(s.notna(), "").astype(str) for s in (p, g))
        for group, idx in groups.groupby(groups).groups.items():
            yt, yp = y_true[idx], y_pred[idx]
            labels = sorted((set(yt) | set(yp)) - {""})  # "" = not set, the negative class
            prf = precision_recall_fscore_support(
                yt, yp, labels=labels, average="micro", zero_division=np.nan
            )[:3] if labels else (np.nan,) * 3  # fmt: skip
            rows.append({
                "field": field, "group": group, "n": len(yt), "support": int(yt.ne("").sum()),
                "n_pred": int(yp.ne("").sum()), "precision": prf[0], "recall": prf[1],
                "f1": prf[2], "accuracy": float((yt == yp).mean()),
            })  # fmt: skip
    table = pd.DataFrame(rows)
    return table.drop(columns="group") if by is None else table.rename(columns={"group": by.name})


def make_tfidf_svd(n_components: int = 50, min_df: int = 3) -> Pipeline:
    """Word 1-2-gram + character 3-5-gram TF-IDF, reduced by truncated SVD (LSA)."""
    common = {"min_df": min_df, "sublinear_tf": True, "max_features": 100_000}
    word = TfidfVectorizer(ngram_range=(1, 2), **common)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common)
    svd = TruncatedSVD(n_components, random_state=RANDOM_STATE)
    return Pipeline([("tfidf", FeatureUnion([("word", word), ("char", char)])), ("svd", svd)])


def tfidf_features(pipe: Pipeline, texts: pd.Series) -> pd.DataFrame:
    """Transform texts (NA = "") with a fitted :func:`make_tfidf_svd` to ``tfidf_<i>`` columns."""
    values = pipe.transform(texts.astype("string").fillna("").tolist())
    columns = [f"tfidf_{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=texts.index, columns=columns)


def embed_texts(
    texts: Sequence[str], cache_dir: Path | None = None, device: str | None = None
) -> np.ndarray:
    """L2-normalised float32 sentence embeddings; cached as ``<cache_dir>/<model>/<sha256>.npy``."""
    clean = [t if isinstance(t, str) else "" for t in texts]
    keys = [hashlib.sha256(t.encode()).hexdigest() for t in clean]
    folder = cache_dir / DEFAULT_EMBEDDING_MODEL.replace("/", "__") if cache_dir else None
    paths = {k: folder / f"{k}.npy" for k in keys} if folder else {}
    vectors = {k: np.load(p) for k, p in paths.items() if p.is_file()}
    missing = {k: t for k, t in zip(keys, clean, strict=True) if k not in vectors}
    if missing:
        # Lazy import: sentence-transformers pulls in torch (optional "text" extra).
        from sentence_transformers import SentenceTransformer

        logger.info("Encoding %d new texts (%d cached)", len(missing), len(vectors))
        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, device=device)
        encoded = model.encode(list(missing.values()), batch_size=64, normalize_embeddings=True)
        vectors |= dict(zip(missing, np.asarray(encoded, dtype=np.float32), strict=True))
        for key in missing.keys() & paths.keys():
            paths[key].parent.mkdir(parents=True, exist_ok=True)
            np.save(paths[key], vectors[key])
    return np.vstack([vectors[k] for k in keys]).astype(np.float32)


def reduce_embeddings(
    train: np.ndarray, others: Sequence[np.ndarray] = (), n_components: int = 32
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Reduce embeddings with a PCA fitted on the training rows only (no leakage)."""
    pca = PCA(n_components=min(n_components, *train.shape), random_state=RANDOM_STATE).fit(train)
    return pca.transform(train), [pca.transform(other) for other in others]
