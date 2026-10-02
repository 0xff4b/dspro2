"""Anonymisation of listing descriptions for :mod:`rentml.text` (private helpers).

Contact data (e-mail, URL, IBAN, phone, person names) is removed for revDSG and every amount is
removed so that no model can read the target from the text. Placeholders: [EMAIL] [URL] [PHONE]
[IBAN] [PRICE] [NAME]. :func:`price_leak_rate` measures what slips through on real data. Import the
public functions from :mod:`rentml.text`.
"""

import re

import numpy as np
import pandas as pd

# --- Anonymisation patterns ---------------------------------------------------------------------
# Amounts: thousands separators (1'850, 1’850, 2 100, 1.850, 2,150; exactly 3 digits after the
# separator) or plain 2-6 digit numbers, optional decimals (1,850.00).
_THOUSANDS = r"\d{1,3}(?:['’ .,]\d{3})+(?!\d)"
_NUM = rf"(?:{_THOUSANDS}|\d{{2,6}})(?:[.,]\d{{1,2}})?(?![\d'’])"
_NUM3 = rf"(?:{_THOUSANDS}|\d{{3,6}})(?:[.,]\d{{1,2}})?(?![\d'’]|[.,]\d)"
_DASH = r"(?:\s*[.,]\s*[-–—]+)"  # Swiss "1850.-" / "1'850.–" notation
_CUR = r"(?:(?:CHF|SFr|Frs|Fr|Franken|francs?|franchi|EUR|Euro)(?:\.|(?=\d)|\b)|€)"  # "CHF1850"
_START = r"(?<![\w'’.,])"
# Compound stems catch "Gesamtmiete", "Nettomietzins", "Mietkosten", "Mietzinsdepot".
_PRICE_CUES = (
    r"\w*(?:miete|mietzins|mietpreis|preis|kosten|depot|kaution)|NK|HK|akonto|pauschale|"
    r"garantie|netto|brutto|mtl|monatlich|pro monat|total|einstellplatz|einstellhallenplatz|"
    r"(?:auto|aussen)?parkplatz|abstellplatz|garagenplatz|garage|loyer(?: net| brut)?|prix|"
    r"charges|frais|acompte|caution|dépôt|place de parc|parking|par mois|affitto|pigione|canone|"
    r"prezzo|spese|acconto|deposito|cauzione|garanzia|posteggio|parcheggio|al mese|rent|price|"
    r"deposit"
)
_NOT_AMOUNT_UNIT = r"(?!\s*(?:m2|m²|m\b|qm|%|zimmer|zi\b|pi[èe]ces?|locali|jahre?|ans?\b|anni))"
# Gap between cue and amount: no digits except room/area tokens ("für 3.5 Zi.-Whg:"), and it
# stops at a sentence end (period after a word of 5+ letters, so "inkl. NK" does not stop it).
_GAP = (
    r"(?:(?!(?<=[^\W\d_]{5})\.\s+(?-i:[A-ZÄÖÜ]))[^\d\n\[]"
    r"|\d+(?:[.,]\d)?\s*-?\s*(?:zimmer|zi\b|pi[èe]ces?|locali|m2|m²)){0,30}?"
)
# A year after a year cue or a month ("renoviert 2019", "ab Juli 2026") is not a price.
_YEAR_CONTEXT_RE = re.compile(
    r"baujahr|jahrgang|renov|r[ée]nov|sanier|erbaut|gebaut|bezug|constru|ristrutt|costru|ann[ée]e"
    r"|abrechnung|d[ée]compte"
    r"|\b(?:seit|depuis|since|built|anno|nel|dal|jan(?:uar)?|feb(?:ruar)?|m[äa]rz|apr(?:il)?|mai"
    r"|juni?|juli?|aug(?:ust)?|sept?(?:ember)?|okt(?:ober)?|nov(?:ember)?|dez(?:ember)?|janvier"
    r"|f[ée]vrier|mars|avril|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre|gennaio"
    r"|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|dicembre|frühling"
    r"|sommer|herbst|ende|anfang|mitte|printemps|été|automne|hiver|fin|début|primavera|autunno"
    r"|inverno|fine|inizio)\b",
    re.IGNORECASE,
)
# A 4-digit amount after ", ", "in", "à" or "CH-" and before a capitalised non-price word is a
# postcode ("Parkplatz an der Bahnhofstrasse, 8001 Zürich"); "Miete 1850 Balkon" is a price.
_POSTCODE_LEAD_RE = re.compile(r"(?:,|\bin|\bà|\bCH-)\s*$", re.IGNORECASE)
_POSTCODE_TAIL_RE = re.compile(
    rf"[ \t]+(?!(?i:{_PRICE_CUES}|franken|fr|euro|monat|mte?|plus|inkl|exkl|zzgl)\b)"
    r"[A-ZÄÖÜ][a-zäöüéèà]"
)

_EMAIL_RE = re.compile(  # bounded quantifiers keep long dotted runs linear
    r"(?<![\w.+-])[\w.+-]{1,64}\s?(?:@|\(at\)|\[at\])\s?[\w-]{1,63}"
    r"(?:\s?(?:\.|\(dot\)|\[dot\])\s?[\w-]{1,63}){0,10}\.[a-z]{2,24}",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"]{0,2000}[^\s<>\".,;:!?)\]]|(?<![\w-])[\w-]{1,63}"
    r"(?:\.[\w-]{1,63}){0,10}\.(?:ch|com|net|org|swiss|info|li)\b"
    r"(?:/[^\s<>\"]{0,2000}[^\s<>\".,;:!?)\]])?",
    re.IGNORECASE,
)
_IBAN_RE = re.compile(
    r"(?<!\w)[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b", re.IGNORECASE
)
_PHONE_RE = re.compile(
    r"(?<![\w+'’.,])(?:\+|00)\s?[1-9]\d{0,2}(?:\s?\(0\))?(?:[\s./-]{0,2}\d){6,12}(?!\d)"
    r"|(?<![\w+'’.,])\(?0\d{2}\)?(?:[\s./-]{0,3}\d){7}(?!\d)"  # "044 123 45 67", "(044) …"
)
_PRICE_RES: tuple[re.Pattern[str], ...] = (
    # Currency first, optional range: "CHF 1'850.–", "Fr. 2000", "CHF 1'800 - 2'000".
    re.compile(
        # Atomic group + look-ahead: "Fr. 10-12 Uhr" is Friday, not francs.
        rf"(?>{_CUR}\s*:?\s*{_NUM}{_DASH}?(?:\s*(?:-|–|bis|à)\s*{_NUM}{_DASH}?)?)"
        r"(?!\s*(?:uhr|h\b|heures?|ore\b))",
        re.IGNORECASE,
    ),
    # Amount first: "2 100 CHF", "1850.- Fr.", "1200 francs".
    re.compile(rf"{_START}{_NUM}{_DASH}?\s*{_CUR}", re.IGNORECASE),
    # Bare Swiss notation: "1850.-", "150.–" (not "12.-15. Mai").
    re.compile(rf"{_START}{_NUM}{_DASH}(?!\w)", re.IGNORECASE),
    # Amount then a monthly or net/gross unit: "1850 pro Monat", "1450/mois", "2150 netto".
    re.compile(
        rf"{_START}{_NUM3}{_DASH}?(?=\s*(?:/\s*|(?:pro|par|per|al|im|a)\s+)(?:mte?\b|monat|mois"
        r"|mese|month)|\s+(?:monatlich|mensuel|mensil|netto|brutto|(?:inkl|exkl|zzgl)\.?\s*nk))",
        re.IGNORECASE,
    ),
)
# Cue words: "Miete: 1850", "loyer 1450" (the cue is kept, see _cue_price_sub; "[PRICE]" is no cue).
_CUE_PRICE_RE = re.compile(
    rf"(?P<cue>(?<!\[)\b(?:{_PRICE_CUES})\b(?P<gap>{_GAP}))(?P<amount>{_START}{_NUM3}{_DASH}?)"
    rf"{_NOT_AMOUNT_UNIT}",
    re.IGNORECASE,
)
_UPPER = "A-ZÄÖÜÀÂÇÈÉÊÎÔÙÛ"
_SALUTATION = (
    r"\b(?:Herrn?|Frau|Hr\.|Fr\.|Monsieur|Madame|Mademoiselle|MM?\.|Mmes?\.?|Mlle\.?|"
    r"Signor(?:a|e|ina)?|Sig\.(?:ra|na)?|Dott\.(?:ssa)?|Mrs?\.?|Ms\.?)"
)
_TITLE = r"(?:(?:Dr|Prof|Ing|Dipl|lic|Dott)\.[ \t]*)*"
# "von", "de la", "van der", "von den" (a lone "der" is an article: "Frau der Verwaltung").
_PARTICLE = (
    r"(?:(?:von|van|de|di|da|del|della|du|le|la)[ \t]+(?:(?:der|den|dem|des|la|le)[ \t]+)?)?"
)
# A name never starts with a salutation (keeps the anonymiser idempotent); an initial ("P.")
# must be followed by a capitalised word.
_NAME = (
    rf"(?!{_SALUTATION}(?!\w)){_TITLE}{_PARTICLE}[{_UPPER}](?:[\w'’-]+|\.(?=[ \t]+[{_UPPER}]))"
    rf"(?:[ \t]+{_PARTICLE}[{_UPPER}][\w'’-]+){{0,2}}"
)
_CONTACT_NAME_RE = re.compile(
    r"(?P<cue>\b(?i:kontakt(?:person)?|ansprechperson|ansprechpartner(?:in)?|contact|"
    r"personne de contact|contatto|persona di contatto|referente)\s*:?\s*)"
    rf"(?P<sal>{_SALUTATION}\s+)?(?P<name>{_NAME})"
)
_SALUTATION_NAME_RE = re.compile(rf"(?P<sal>{_SALUTATION}\s+)(?P<name>{_NAME})")
# Kept years/postcodes are hidden as private-use characters so that a later pass can still find
# a price behind them ("Miete ab Juli 2026: 1850"); they are restored at the end.
_HIDE_DIGITS = str.maketrans({str(d): chr(0xE000 + d) for d in range(10)})
_SHOW_DIGITS = str.maketrans({chr(0xE000 + d): str(d) for d in range(10)})
# Any number left in a text, for the leak check: "1'850", "2,150.00", "1850", "3.5".
_ANY_NUMBER_RE = r"(?<![\d'’.,])(\d{1,3}(?:['’ .,]\d{3})+(?!\d)|\d+)(?:[.,](\d{1,2}))?(?!\d)"


def _cue_price_sub(match: re.Match[str]) -> str:
    amount = match["amount"]
    is_year = re.fullmatch(r"(?:19|20)\d\d", amount) and _YEAR_CONTEXT_RE.search(match["gap"])
    is_postcode = (
        len(amount) == 4
        and amount.isdigit()
        and _POSTCODE_LEAD_RE.search(match["gap"])
        and _POSTCODE_TAIL_RE.match(match.string, match.end())
    )
    if is_year or is_postcode:
        return match["cue"] + amount.translate(_HIDE_DIGITS)
    return f"{match['cue']}[PRICE]"


def anonymize(text: str | None) -> str:
    """Remove contact data and amounts from a listing description.

    Order matters: e-mails and URLs first (they contain digits and dots), then IBANs before phone
    numbers (an IBAN contains phone-like digit groups), then amounts, then person names.
    Room counts ("3.5 Zimmer"), areas ("85 m²"), dates, years after a year cue ("Baujahr 1995",
    "renoviert 2019") and postcodes ("8001 Zürich") are kept. Check the result on real data
    with :func:`price_leak_rate`.

    Args:
        text: Raw description; ``None`` or non-string values give an empty string.

    Returns:
        The anonymised text with placeholders [EMAIL] [URL] [IBAN] [PHONE] [PRICE] [NAME].
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    out = _EMAIL_RE.sub("[EMAIL]", text)
    out = _URL_RE.sub("[URL]", out)
    out = _IBAN_RE.sub("[IBAN]", out)
    out = _PHONE_RE.sub("[PHONE]", out)
    for pattern in _PRICE_RES:
        out = pattern.sub("[PRICE]", out)
    while (hidden := _CUE_PRICE_RE.sub(_cue_price_sub, out)) != out:  # each pass removes digits
        out = hidden
    out = out.translate(_SHOW_DIGITS)
    out = _CONTACT_NAME_RE.sub(lambda m: f"{m.group('cue')}{m.group('sal') or ''}[NAME]", out)
    out = _SALUTATION_NAME_RE.sub(lambda m: f"{m.group('sal')}[NAME]", out)
    return out.strip()


def anonymize_series(s: pd.Series) -> pd.Series:
    """Anonymise a Series of descriptions.

    Args:
        s: Raw descriptions (may contain NA).

    Returns:
        A ``string``-dtype Series with the same index; missing or blank inputs stay ``<NA>``.
    """
    out = s.map(anonymize, na_action="ignore").astype("string")
    return out.mask(out.fillna("").str.len() == 0)


def anonymize_descriptions(df: pd.DataFrame, col: str = "description") -> pd.DataFrame:
    """Return a copy of ``df`` whose text column is anonymised (idempotent, NA stays ``None``).

    Used before a database export is cached, so raw contact data never reaches the disk.

    Args:
        df: Listings with the text column ``col``.
        col: Description column.

    Returns:
        The copy with an object column of anonymised texts (``None`` for missing/blank texts).
    """
    anon = anonymize_series(df[col]).astype(object)
    return df.assign(**{col: anon.where(anon.notna(), None)})


def price_leak_mask(anon: pd.Series, price: pd.Series, rel_tol: float = 0.03) -> pd.Series:
    """Flag texts that still contain a number within ``±rel_tol`` of the listing's own rent.

    Years or postcodes that happen to lie near the rent are flagged as well, so inspect the
    flagged rows before blaming the anonymiser.

    Args:
        anon: Anonymised descriptions (NA allowed) with a unique index.
        price: Monthly rent in CHF, aligned to ``anon`` by index.
        rel_tol: Relative tolerance (0.03 = ±3 %).

    Returns:
        Boolean Series indexed like ``anon``.

    Raises:
        ValueError: If ``rel_tol`` is negative.
    """
    if rel_tol < 0:
        raise ValueError(f"rel_tol must be >= 0, got {rel_tol}")
    parts = anon.astype("string").fillna("").str.extractall(_ANY_NUMBER_RE)
    if parts.empty:
        return pd.Series(False, index=anon.index)
    number = parts[0].str.replace(r"['’ .,]", "", regex=True) + "." + parts[1].fillna("0")
    values = pd.to_numeric(number).to_numpy(float)
    rent = pd.to_numeric(price, errors="coerce").reindex(parts.index.get_level_values(0))
    rent_chf = rent.to_numpy(float)  # NaN rent never matches
    hits = pd.Series(np.abs(values - rent_chf) <= rel_tol * rent_chf, index=parts.index)
    return hits.groupby(level=0).any().reindex(anon.index, fill_value=False).astype(bool)


def price_leak_rate(anon: pd.Series, price: pd.Series, rel_tol: float = 0.03) -> float:
    """Share of listings (non-empty text, positive rent) flagged by :func:`price_leak_mask`.

    Args:
        anon: Anonymised descriptions with a unique index.
        price: Monthly rent in CHF, aligned by index.
        rel_tol: Relative tolerance (0.03 = ±3 %).

    Returns:
        The leak rate in [0, 1]; NaN if no listing has both a text and a positive rent.
    """
    rent = pd.to_numeric(price, errors="coerce").reindex(anon.index)
    has_text = anon.astype("string").fillna("").str.strip().str.len().gt(0).to_numpy(bool)
    valid = has_text & rent.gt(0).to_numpy(bool)
    if not valid.any():
        return float("nan")
    return float(price_leak_mask(anon, price, rel_tol).to_numpy(bool)[valid].mean())
