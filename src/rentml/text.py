"""Text preparation for listing descriptions.

Anonymisation (revDSG: no contact data, no prices so the model cannot read the target; implemented
in :mod:`rentml._anonymize` and re-exported here together with the leak check), a stop-word
language detector, multilingual keyword flags, a TF-IDF + SVD transformer and a cached
multilingual sentence embedder. Placeholders: [EMAIL] [URL] [PHONE] [IBAN] [PRICE] [NAME].
"""

import hashlib
import importlib.util
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from rentml._anonymize import anonymize as anonymize
from rentml._anonymize import anonymize_series as anonymize_series
from rentml._anonymize import price_leak_mask as price_leak_mask
from rentml._anonymize import price_leak_rate as price_leak_rate
from rentml.config import RANDOM_STATE

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Language detection -------------------------------------------------------------------------
# Only discriminative stop words: "la", "le", "in", "per", "da", "un", "a" are shared; Italian "e"
# is left out because "E-Mail" and "3e étage" tokenise to it.
_STOPWORDS: dict[str, frozenset[str]] = {
    "de": frozenset({
        "der", "die", "das", "und", "ist", "mit", "von", "zu", "im", "den", "dem", "ein", "eine",
        "einer", "nicht", "auf", "für", "sich", "auch", "wir", "sie", "bei", "nach", "aus", "wird",
        "sind", "oder", "zum", "zur", "sehr", "ihr", "wohnung", "zimmer", "küche", "miete",
        "ruhige", "grosse", "schöne", "neu", "hell",
    }),
    "fr": frozenset({
        "et", "est", "avec", "du", "une", "pour", "dans", "au", "aux", "sur", "par", "pas", "vous",
        "nous", "sont", "très", "appartement", "pièces", "cuisine", "loyer", "ce", "cette", "les",
        "à", "grand", "belle", "lumineux", "quartier", "proche",
    }),
    "it": frozenset({
        "lo", "gli", "è", "con", "di", "del", "della", "dei", "una", "nel", "nella", "sono",
        "molto", "appartamento", "locali", "cucina", "affitto", "che", "al", "alla", "delle",
        "luminoso", "bello", "vicino",
    }),
    "en": frozenset({
        "the", "and", "is", "with", "of", "to", "for", "this", "are", "be", "apartment", "flat",
        "room", "kitchen", "rent", "bedroom", "located", "near", "bright", "large",
    }),
}  # fmt: skip
_TOKEN_RE = re.compile(r"[a-zäöüàâçéèêëîïôûùìíòóú]+")


def detect_language(text: str | None, *, min_hits: int = 1) -> str:
    """Detect the language of a description with a stop-word vote.

    Args:
        text: Description (raw or anonymised).
        min_hits: Minimum number of stop-word hits of the winning language.

    Returns:
        ``"de"``, ``"fr"``, ``"it"``, ``"en"`` or ``"unknown"`` (empty text, too few hits or a
        tie between the two best languages).
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    tokens = _TOKEN_RE.findall(text.lower())
    scores = {lang: sum(tok in words for tok in tokens) for lang, words in _STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score < min_hits or best_score == second_score:
        return "unknown"
    return best


# --- Keyword flags ------------------------------------------------------------------------------
# Non-capturing regexes (DE/FR/IT/EN) written in lower case and applied to lower-cased text
# (about 5x faster than re.IGNORECASE). Flags mark mentions; furnished and temporary skip simple
# negations ("unmöbliert", "nicht befristet"), garden skips "Kindergarten" and "Gartenstrasse".
KEYWORD_PATTERNS: dict[str, str] = {
    "balcony": r"balkon|balcon|loggia",
    "terrace": r"terrass|terrazz|sitzplatz|patio",
    "lift": r"(?<!ski)(?<!sessel)lift\b|aufzug|fahrstuhl|ascenseur|ascensore|elevator",
    "lake_view": r"see-?\s?(?:sicht|blick)|sicht auf den \w*see|vue\W+(?:\w+\W+){0,3}lac\b"
    r"|vista\W+(?:\w+\W+){0,3}lago\b|lake view",
    "mountain_view": r"berg(?:sicht|blick)|alpen(?:sicht|blick|panorama)"
    r"|sicht auf die (?:berge|alpen)|vue\W+(?:\w+\W+){0,3}(?:montagnes?|alpes)\b"
    r"|vista\W+(?:\w+\W+){0,3}(?:montagn[ae]|monti|alpi)\b|mountain view",
    "renovated": r"r[ée]nov|saniert|sanierung|moderni[sz]iert|ristruttur|rinnov",
    "new_build": r"neubau|erstbezug|nouvelle construction|construction neuve|premi[èe]re location"
    r"|immeuble neuf|appartement neuf|nuova costruzione|prima locazione|new build|newly built",
    "furnished": r"(?<!un)(?<!nicht )m(?:ö|oe|o)bliert|(?<!non )(?<!non-)(?<!pas )meubl[ée]"
    r"|(?<!non )(?<!non-)arredat|(?<!un)furnished",
    "temporary": r"(?<!un)(?<!nicht )befristet|tempor[äa]r|zwischenmiete|auf zeit\b"
    r"|dur[ée]e d[ée]termin[ée]e|temporaire|courte dur[ée]e|temporane|tempo determinato"
    r"|breve periodo|temporary",
    "parking": r"parkpl[äa]tz|abstellplatz|place de parc|places de parc|parking|posteggi"
    r"|parcheggi|posto auto",
    "garage": r"garage|einstellhalle|einstellplatz|tiefgarage|autorimessa|box auto",
    "minergie": r"minergie",
    "pets": r"haustier|\bkatzen?\b|\bhunde?\b|animaux|animali|\bchats?\b|\bchiens?\b|\bpets?\b",
    "garden": r"(?<!kinder)garten(?!\w*(?:strasse|str\b|weg\b|gasse))|jardin|giardin|garden",
    "dishwasher": r"geschirrsp[üu]e?l|abwaschmaschine|sp[üu]lmaschine|lave-vaisselle|lavastoviglie"
    r"|dishwasher",
    "own_washer": r"waschturm|eigene\w*\s+(?:waschmaschine|wm\b|waschk[üu]che)"
    r"|wm\s*/\s*(?:tumbler|tr)\b|waschmaschine\s*(?:und|/|&)\s*tumbler|colonne de lavage"
    r"|(?:lave-linge|machine [àa] laver)\s+(?:priv|individ|dans l)|colonna (?:di )?lavaggio"
    r"|lavatrice\s+(?:propri|privat|in appartamento)|own washer",
    "fireplace": r"kamin|schwedenofen|chemin[ée]e|caminett|\bcamino\b|fireplace",
    "attic": r"dachwohnung|dachgescho|dachstock|mansard|attika|combles|sous les toits|attique"
    r"|sottotetto|attico|attic|penthouse",
    "loft": r"\bloft",
    "quiet": r"ruhig|calme|tranquill|silenzios|quiet",
    "cooperative": r"genossenschaft|coop[ée]rative|cooperativa|anteilschein|parts? sociales?"
    r"|quot[ae] social[ei]",
    "subsidised": r"subventio|verbillig|kostenmiete|gemeinn(?:ü|ue)tzig|loyer mod[ée]r|\blup\b"
    r"|sussidi|pigione moderata|canone moderato|alloggi popolari",
    "shared_flat": r"\bwg\b(?![-\s]?(?:tauglich|geeignet))|wohngemeinschaft|mitbewohner|untermiet"
    r"|colocat|\bcoloc\b|coinquilin|appartamento condiviso|shared flat|flatshare",
    "wheelchair": r"rollstuhl|hindernisfrei|behindertengerecht|barrierefrei|fauteuil roulant"
    r"|sans obstacle|sedia a rotelle|senza barriere|wheelchair",
}


def keyword_flags(texts: pd.Series) -> pd.DataFrame:
    """Flag keyword mentions per description.

    Args:
        texts: Descriptions (NA treated as empty text).

    Returns:
        DataFrame with the same index and one ``int8`` column ``kw_<name>`` per
        :data:`KEYWORD_PATTERNS` entry.
    """
    lowered = texts.astype("string").fillna("").astype(str).str.lower()
    flags = {
        f"kw_{name}": lowered.str.contains(pattern, regex=True).astype("int8")
        for name, pattern in KEYWORD_PATTERNS.items()
    }
    return pd.DataFrame(flags, index=texts.index)


# --- TF-IDF + SVD -------------------------------------------------------------------------------
def _as_text_list(x: object) -> tuple[list[str], pd.Index | None]:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected one text column, got {x.shape[1]}")
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return x.astype("string").fillna("").astype(str).tolist(), x.index
    if isinstance(x, str) or not isinstance(x, Iterable):
        raise TypeError("Expected an iterable of texts, not a single string")
    return [v if isinstance(v, str) else "" for v in x], None


class TfidfSvd(TransformerMixin, BaseEstimator):
    """Word (1-2 gram) + character (3-5 gram, ``char_wb``) TF-IDF compressed by truncated SVD.

    Both TF-IDF blocks use sublinear term frequency and ``min_df``; they are stacked horizontally
    and reduced with a seeded :class:`~sklearn.decomposition.TruncatedSVD` (LSA). The character
    block makes the representation robust to Swiss spelling variants and compounds.

    Args:
        n_components: Target number of SVD components; capped at ``min(n_features, n_docs) - 1``.
        min_df: Minimum document frequency for both vectorisers.
        max_features: Vocabulary cap per vectoriser (``None`` = unlimited).
        random_state: Seed of the randomised SVD.
    """

    def __init__(
        self,
        n_components: int = 50,
        min_df: int = 3,
        max_features: int | None = 100_000,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_components = n_components
        self.min_df = min_df
        self.max_features = max_features
        self.random_state = random_state

    def _stack(self, texts: list[str]) -> sparse.csr_matrix:
        return sparse.hstack(
            [self.word_vectorizer_.transform(texts), self.char_vectorizer_.transform(texts)]
        ).tocsr()

    def fit(self, X: object, y: object = None) -> Self:
        """Fit both vectorisers and the SVD.

        Args:
            X: Texts (Series, one-column DataFrame or list); NA is treated as "".
            y: Ignored.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If no term survives ``min_df`` (e.g. all texts empty).
        """
        texts, _ = _as_text_list(X)
        common = {"min_df": self.min_df, "sublinear_tf": True, "max_features": self.max_features}
        self.word_vectorizer_ = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), **common)
        self.char_vectorizer_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common)
        self.word_vectorizer_.fit(texts)
        self.char_vectorizer_.fit(texts)
        matrix = self._stack(texts)
        self.n_components_ = max(1, min(self.n_components, matrix.shape[1] - 1, len(texts) - 1))
        if self.n_components_ < self.n_components:
            logger.warning("TfidfSvd: n_components capped to %d", self.n_components_)
        self.svd_ = TruncatedSVD(n_components=self.n_components_, random_state=self.random_state)
        self.svd_.fit(matrix)
        return self

    def transform(self, X: object) -> pd.DataFrame:
        """Project texts onto the fitted SVD components.

        Args:
            X: Texts (Series keeps its index in the output).

        Returns:
            DataFrame with columns ``tfidf_0`` … ``tfidf_{k-1}``.
        """
        texts, index = _as_text_list(X)
        values = self.svd_.transform(self._stack(texts))
        return pd.DataFrame(values, index=index, columns=self.get_feature_names_out())

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the output column names.

        Args:
            input_features: Ignored (sklearn API).

        Returns:
            Array of ``tfidf_<i>`` names.
        """
        return np.array([f"tfidf_{i}" for i in range(self.n_components_)], dtype=object)


# --- Sentence embeddings ------------------------------------------------------------------------
class SentenceEmbedder:
    """Multilingual sentence embeddings with an on-disk ``.npy`` cache per text.

    Each text is stored once as ``<cache_dir>/<model>[__unnormalized]/<sha256(text)>.npy``, so
    re-running the notebook never re-encodes a description. Without ``cache_dir`` an in-memory
    cache is used.

    Args:
        model_name: Hugging Face model id.
        cache_dir: Cache root directory (``None`` = memory only).
        device: Torch device ("cuda", "cpu"); ``None`` lets sentence-transformers choose.
        batch_size: Encoding batch size.
        normalize: L2-normalise embeddings.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path | None = None,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: Any = None  # SentenceTransformer: untyped optional dependency
        self._memory: dict[str, np.ndarray] = {}

    @staticmethod
    def is_available() -> bool:
        """Return whether ``sentence_transformers`` is importable (without importing it)."""
        return importlib.util.find_spec("sentence_transformers") is not None

    @staticmethod
    def text_key(text: str) -> str:
        """Return the cache key (sha256 hex digest) of a text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _model_dir(self) -> Path | None:
        if self.cache_dir is None:
            return None
        suffix = "" if self.normalize else "__unnormalized"  # never mix the two vector kinds
        return self.cache_dir / f"{self.model_name.replace('/', '__')}{suffix}"

    def _load_model(self) -> Any:  # SentenceTransformer is an untyped optional dependency
        # Lazy import: sentence_transformers pulls in torch (slow, large, optional extra "text").
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer %s", self.model_name)
        return SentenceTransformer(self.model_name, device=self.device)

    def _read_cached(self, key: str) -> np.ndarray | None:
        if key in self._memory:
            return self._memory[key]
        model_dir = self._model_dir()
        path = None if model_dir is None else model_dir / f"{key}.npy"
        if path is None or not path.is_file():
            return None
        try:
            vector = np.load(path)
        except (OSError, ValueError) as err:
            logger.warning("Ignoring unreadable embedding cache file %s: %s", path, err)
            return None
        self._memory[key] = vector
        return vector

    def _write_cached(self, key: str, vector: np.ndarray) -> None:
        self._memory[key] = vector
        model_dir = self._model_dir()
        if model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
            np.save(model_dir / f"{key}.npy", vector)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts, encoding only those not in the cache.

        Args:
            texts: Texts to embed (non-strings are treated as "").

        Returns:
            ``float32`` array of shape ``(len(texts), dim)``; ``(0, 0)`` for empty input.
        """
        clean = [t if isinstance(t, str) else "" for t in texts]
        if not clean:
            return np.zeros((0, 0), dtype=np.float32)
        keys = [self.text_key(t) for t in clean]
        missing = {k: t for k, t in zip(keys, clean, strict=True) if self._read_cached(k) is None}
        if missing:
            if self._model is None:
                self._model = self._load_model()
            logger.info(
                "Encoding %d new texts (%d cached)", len(missing), len(clean) - len(missing)
            )
            vectors = self._model.encode(
                list(missing.values()),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            for key, vector in zip(missing, vectors, strict=True):
                self._write_cached(key, np.asarray(vector, dtype=np.float32))
        return np.vstack([self._memory[k] for k in keys]).astype(np.float32)


def reduce_embeddings(
    train_emb: np.ndarray,
    other_embs: Sequence[np.ndarray] = (),
    n_components: int = 32,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Reduce embeddings with a PCA fitted on the training rows only (no leakage).

    Args:
        train_emb: Training embeddings, shape ``(n, d)``.
        other_embs: Further matrices (calibration, test, …) with ``d`` columns.
        n_components: Target dimension; capped at ``min(n, d)``.
        seed: PCA random state.

    Returns:
        The reduced training matrix and the list of reduced other matrices.

    Raises:
        ValueError: If a matrix is not 2-D or the column counts differ.
    """
    if train_emb.ndim != 2 or train_emb.shape[0] == 0:
        raise ValueError("train_emb must be a non-empty 2-D array")
    for other in other_embs:
        if other.ndim != 2 or other.shape[1] != train_emb.shape[1]:
            raise ValueError(f"Expected shape (m, {train_emb.shape[1]}), got {other.shape}")
    k = min(n_components, *train_emb.shape)
    pca = PCA(n_components=k, random_state=seed).fit(train_emb)
    return pca.transform(train_emb), [pca.transform(other) for other in other_embs]
