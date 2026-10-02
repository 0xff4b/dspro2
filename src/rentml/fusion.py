"""Deep-learning fusion model for log rent (PyTorch): entity embeddings + numeric + text branch.

Categorical columns (e.g. ``re_canton``/``re_district``/``re_municipality``) get entity embeddings
(index 0 = unknown, rare or missing); weight decay shrinks rare municipalities towards zero like a
random effect. Train-only preprocessing, AdamW, Huber/L1/MSE loss on the log price, early stopping
on the validation MAE in CHF with restoration of the best weights.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import logging
import math
from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from rentml.config import RANDOM_STATE

try:  # torch is the optional "dl" extra: keep the rest of rentml importable without it.
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = "<unknown>"
_CHOICES = {"loss": ("huber", "l1", "mse"), "scheduler": ("plateau", "onecycle", "none")}
_CLIP_Z = 8.0  # standardised numerics are clipped, so extreme inputs cannot explode activations
_EPS = 1e-12
TextInput = np.ndarray | pd.DataFrame  # (n_rows, dim); a DataFrame must share X_num's index
type _Optim = tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]


def is_torch_available() -> bool:
    """Return True if PyTorch could be imported."""
    return torch is not None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for rentml.fusion: run `uv sync --extra dl`")


def embedding_dim(n_categories: int, max_dim: int = 16) -> int:
    """Embedding width rule of thumb ``min(max_dim, round(1.6 * n ** 0.56))`` (fast.ai).

    Args:
        n_categories: Number of known categories (without the unknown slot).
        max_dim: Upper bound on the width.

    Returns:
        The embedding dimension (at least 1).
    """
    return max(1, min(max_dim, round(1.6 * max(n_categories, 1) ** 0.56)))


@dataclasses.dataclass
class FusionConfig:
    """Hyperparameters of the network and training loop; invalid values raise ValueError."""

    hidden: tuple[int, ...] = (256, 128)  # widths of the fused MLP head
    dropout: float = 0.15  # after every block and on the concatenated embeddings
    lr: float = 2e-3  # AdamW start (plateau) or peak (onecycle) learning rate
    weight_decay: float = 1e-4  # on weights and embeddings, not on biases / LayerNorm scales
    epochs: int = 200  # maximum number of epochs
    batch_size: int = 256  # mini-batch size
    patience: int = 20  # early-stopping patience in epochs (validation MAE in CHF)
    emb_dim_max: int = 16  # upper bound of the embedding width, see embedding_dim()
    text_proj_dim: int = 32  # output width of the text projection
    loss: str = "huber"  # "huber" | "l1" | "mse", computed on the log price
    seed: int = RANDOM_STATE  # weight init, dropout, shuffling, unknown-category masking
    device: str | None = None  # "cpu", "cuda", "cuda:1"; None = CUDA if available else CPU
    huber_delta: float = 0.15  # Huber transition in log units (0.15 ~ 16 % rent error)
    scheduler: str = "plateau"  # "plateau" | "onecycle" | "none"
    numeric_width: int = 64  # output width of the numeric branch
    log1p_cols: tuple[str, ...] = ()  # skewed numeric columns log1p-transformed (clip at 0)
    min_category_count: int = 2  # rarer training categories map to the unknown index 0
    cat_unknown_rate: float = 0.03  # share of training categories masked as unknown per batch
    grad_clip: float = 5.0  # maximum gradient norm

    def __post_init__(self) -> None:
        """Normalise sequence fields and validate all values."""
        self.hidden = tuple(int(h) for h in self.hidden)
        cols = self.log1p_cols  # a bare string would otherwise be split into characters
        self.log1p_cols = (cols,) if isinstance(cols, str) else tuple(cols)
        positive = ["lr", "huber_delta", "epochs", "batch_size", "patience", "emb_dim_max"]
        positive += ["text_proj_dim", "numeric_width", "min_category_count", "grad_clip"]
        invalid = [name for name in positive if getattr(self, name) <= 0]
        invalid += [n for n in ("dropout", "cat_unknown_rate") if not 0 <= getattr(self, n) < 1]
        invalid += [name for name, ok in _CHOICES.items() if getattr(self, name) not in ok]
        invalid += ["weight_decay"] * (self.weight_decay < 0)
        invalid += ["hidden (>= 1 layer, widths >= 1)"] * (min(self.hidden, default=0) < 1)
        if invalid:
            raise ValueError(f"Invalid FusionConfig value(s) for: {', '.join(invalid)}")


def _block(n_in: int, n_out: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.SiLU(), nn.Dropout(p))


class FusedRentNet(nn.Module if nn is not None else object):  # object only without torch
    """Entity embeddings + numeric branch + optional text branch, fused by an MLP head.

    Args:
        n_numeric: Width of the preprocessed numeric input (0 disables the branch).
        cat_cardinalities: Embedding table size per categorical column (incl. unknown index 0).
        text_dim: Width of the preprocessed text input (0 disables the branch).
        config: Architecture settings (hidden, dropout, text_proj_dim, numeric_width).
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: Sequence[int],
        text_dim: int = 0,
        config: FusionConfig | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        cfg = config or FusionConfig()
        dims = [embedding_dim(card - 1, cfg.emb_dim_max) for card in cat_cardinalities]
        self.embeddings = nn.ModuleList(map(nn.Embedding, cat_cardinalities, dims))
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.05)  # start near the "average" category
        self.emb_dropout = nn.Dropout(cfg.dropout)
        num_w = cfg.numeric_width if n_numeric else 0
        text_w = cfg.text_proj_dim if text_dim else 0
        self.numeric = _block(n_numeric, num_w, cfg.dropout) if num_w else None
        self.text = _block(text_dim, text_w, cfg.dropout) if text_w else None
        widths = [sum(dims) + num_w + text_w, *cfg.hidden]
        blocks = [_block(n_in, n_out, cfg.dropout) for n_in, n_out in itertools.pairwise(widths)]
        self.head, self.out = nn.Sequential(*blocks), nn.Linear(widths[-1], 1)

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor, x_text: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the standardised log price, shape (n,)."""
        return self.out(self.representation(x_num, x_cat, x_text)).squeeze(-1)

    def representation(
        self, x_num: torch.Tensor, x_cat: torch.Tensor, x_text: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fused per-listing representation (last hidden layer), shape (n, hidden[-1])."""
        parts = [] if self.numeric is None else [self.numeric(x_num)]
        if len(self.embeddings) > 0:
            embedded = [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
            parts.append(self.emb_dropout(torch.cat(embedded, dim=1)))
        if self.text is not None:
            if x_text is None:
                raise ValueError("This network has a text branch; x_text is required")
            parts.append(self.text(x_text))
        return self.head(torch.cat(parts, dim=1))


class _Preprocessor:  # imputation, missing indicators, scaling, category vocabularies
    def __init__(
        self, X_num: pd.DataFrame, X_cat: pd.DataFrame, X_text: TextInput | None, cfg: FusionConfig
    ) -> None:
        self.num_cols, self.cat_cols = list(X_num.columns), list(X_cat.columns)
        if unknown := sorted(set(cfg.log1p_cols) - set(self.num_cols)):
            logger.warning("log1p_cols not in X_num (ignored): %s", unknown)
        self.log_mask = np.array([c in cfg.log1p_cols for c in self.num_cols], dtype=bool)
        raw = self._raw(X_num)
        missing = np.isnan(raw)
        self.medians = np.nan_to_num(np.nanmedian(raw, axis=0))  # all-NaN column (warns) -> 0
        filled = np.where(missing, self.medians, raw)
        self.means, std = filled.mean(axis=0), filled.std(axis=0)
        self.stds = np.where(std < _EPS, 1.0, std)
        self.indicator_pos = np.flatnonzero(missing.any(axis=0) & ~missing.all(axis=0))
        self.vocabs: list[pd.Index] = []  # known labels; position + 1 = embedding index
        self.counts: list[np.ndarray] = []  # training rows per embedding index (0 = unknown)
        for col in self.cat_cols:
            freq = _category_keys(X_cat[col]).value_counts(dropna=True)
            known = sorted(freq.index[freq >= cfg.min_category_count])
            self.vocabs.append(pd.Index(known, dtype=object))
            n_known = freq[self.vocabs[-1]].to_numpy()
            self.counts.append(np.r_[len(X_cat) - n_known.sum(), n_known])
        self.text_dim = None if X_text is None else np.shape(X_text)[-1]
        self.n_numeric = len(self.num_cols) + len(self.indicator_pos)
        text_width = 0 if self.text_dim is None else self.text_dim + 1  # + missing-text flag
        self.net_shape = (self.n_numeric, [len(v) + 1 for v in self.vocabs], text_width)

    def _raw(self, X_num: pd.DataFrame) -> np.ndarray:
        raw = X_num[self.num_cols].to_numpy(dtype=np.float64, na_value=np.nan)
        raw[~np.isfinite(raw)] = np.nan
        raw[:, self.log_mask] = np.log1p(np.clip(raw[:, self.log_mask], 0.0, None))
        return raw

    def transform(
        self, X_num: pd.DataFrame, X_cat: pd.DataFrame, X_text: TextInput | None
    ) -> list[np.ndarray | None]:
        raw = self._raw(X_num)
        missing = np.isnan(raw)
        filled = np.where(missing, self.medians, raw)
        z = np.clip((filled - self.means) / self.stds, -_CLIP_Z, _CLIP_Z)
        num = np.hstack([z, missing[:, self.indicator_pos]]).astype(np.float32)
        cats = X_cat[self.cat_cols]  # KeyError if a training column is missing
        cat = np.zeros((len(X_num), len(self.cat_cols)), dtype=np.int64)
        for j, vocab in enumerate(self.vocabs):
            cat[:, j] = vocab.get_indexer(_category_keys(cats.iloc[:, j])) + 1  # unseen -> 0
        return [num, cat, self._text(X_text)]

    def _text(self, X_text: TextInput | None) -> np.ndarray | None:
        if (self.text_dim is None) != (X_text is None):
            state = "with" if self.text_dim is not None else "without"
            raise ValueError(f"The model was fitted {state} X_text; pass X_text consistently")
        if X_text is None:
            return None
        arr = pd.DataFrame(X_text).to_numpy(dtype=np.float64, na_value=np.nan)  # also pd.NA
        if arr.shape[1] != self.text_dim:
            raise ValueError(f"X_text must have {self.text_dim} columns, got shape {arr.shape}")
        missing = ~np.isfinite(arr).all(axis=1, keepdims=True)  # NaN/inf: e.g. no description
        return np.hstack([np.where(missing, 0.0, arr), missing]).astype(np.float32)


def _category_keys(s: pd.Series) -> pd.Series:
    # Map 261, 261.0 and "261" to the key "261" for any dtype: joins NaN-upcast integer ids to
    # float, and ids also arrive as category/object/string dtype (e.g. notebook vs app).
    if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s.dtype):
        num = pd.to_numeric(s.astype(object), errors="coerce")
        s = num if num.notna().equals(s.notna()) else s  # only if every label is numeric
    if pd.api.types.is_float_dtype(s.dtype) and (s.dropna() % 1 == 0).all():
        s = s.astype("Int64")
    return s.astype("string")


def _validate(
    X_num: pd.DataFrame,
    X_cat: pd.DataFrame | None,
    X_text: TextInput | None,
    y_log: np.ndarray | pd.Series | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    X_cat = pd.DataFrame(index=X_num.index) if X_cat is None else X_cat
    if len(X_cat) != len(X_num) or (X_cat.shape[1] and not X_cat.index.equals(X_num.index)):
        raise ValueError("X_num and X_cat must have the same length and index")
    if X_text is not None and len(X_text) != len(X_num):
        raise ValueError(f"X_text has {len(X_text)} rows, X_num has {len(X_num)}")
    for name, obj in (("X_text", X_text), ("y_log", y_log)):
        if isinstance(obj, pd.Series | pd.DataFrame) and not obj.index.equals(X_num.index):
            raise ValueError(f"{name} index does not match X_num index")
    if y_log is None:
        return X_cat, None
    y = np.asarray(y_log, dtype=np.float64).ravel()
    if len(y) != len(X_num) or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("y_log must be finite (no NaN/inf), aligned with X_num and >= 2 rows")
    return X_cat, y


def _optimizer(model: FusedRentNet, cfg: FusionConfig, n_rows: int) -> _Optim:
    decay = [p for p in model.parameters() if p.ndim >= 2]  # weights and embeddings
    no_decay = [p for p in model.parameters() if p.ndim < 2]  # biases and LayerNorm scales
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay}]
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=0.0)
    sched, total_steps = torch.optim.lr_scheduler, cfg.epochs * math.ceil(n_rows / cfg.batch_size)
    if cfg.scheduler == "onecycle":
        return optimizer, sched.OneCycleLR(optimizer, cfg.lr, total_steps=total_steps)
    if cfg.scheduler == "plateau":
        patience = max(2, cfg.patience // 4)
        return optimizer, sched.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)
    return optimizer, None


@dataclasses.dataclass
class _Batch:
    num: torch.Tensor
    cat: torch.Tensor
    text: torch.Tensor | None
    y: torch.Tensor | None = None  # standardised log price
    y_log: np.ndarray | None = None


class FusedRentRegressor:
    """Sklearn-like wrapper that preprocesses, trains and applies :class:`FusedRentNet`.

    Args:
        config: Hyperparameters; defaults to ``FusionConfig()``.
        **overrides: Field overrides applied on top of ``config`` (e.g. ``epochs=50``).

    Attributes:
        history_: DataFrame ``epoch``, ``train_loss`` (log scale), ``val_mae`` (CHF), ``lr``.
        best_epoch_: Epoch whose weights were restored (last epoch without ``eval_set``).
        best_val_mae_: Best validation MAE in CHF (NaN without ``eval_set``).
    """

    def __init__(self, config: FusionConfig | None = None, **overrides: object) -> None:
        self.config = dataclasses.replace(config or FusionConfig(), **overrides)  # own copy

    def fit(
        self,
        X_num: pd.DataFrame,
        X_cat: pd.DataFrame | None,
        y_log: np.ndarray | pd.Series,
        *,
        X_text: TextInput | None = None,
        eval_set: tuple | None = None,
    ) -> Self:
        """Fit the preprocessing on the training rows and train the network.

        Args:
            X_num: Numeric features (NaN allowed; may have zero columns).
            X_cat: Categorical features (any dtype, NaN allowed) or None.
            y_log: Target ``log(price)`` aligned with ``X_num``.
            X_text: Optional text embeddings (n_rows, dim); rows with NaN = missing text.
            eval_set: ``(X_num, X_cat, y_log[, X_text])`` for early stopping on the MAE in CHF;
                without it all ``epochs`` are trained and the last weights are kept.

        Returns:
            The fitted regressor.

        Raises:
            ValueError: On misaligned inputs, a non-finite target or no input features.
        """
        _require_torch()
        cfg = self.config
        X_cat, y = _validate(X_num, X_cat, X_text, y_log)
        if eval_set is not None and len(eval_set) not in (3, 4):
            raise ValueError("eval_set must be (X_num, X_cat, y_log) or (..., X_text)")
        self._prep = _Preprocessor(X_num, X_cat, X_text, cfg)
        if self._prep.n_numeric + len(self._prep.cat_cols) + (X_text is not None) == 0:
            raise ValueError("No input features: pass numeric, categorical or text inputs")
        self.target_mean_, self.target_std_ = float(y.mean()), max(float(y.std()), _EPS)
        self.device_ = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device_.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"Device {cfg.device!r} requested but CUDA is not available")
        train, val = self._batch(X_num, X_cat, X_text, y), None
        if eval_set is not None:
            X_text_val = eval_set[3] if len(eval_set) == 4 else None
            val = self._batch(eval_set[0], eval_set[1], X_text_val, eval_set[2])
        gpus = [self.device_.index] if self.device_.type == "cuda" else []
        gpus = [torch.cuda.current_device() if i is None else i for i in gpus]
        with torch.random.fork_rng(devices=gpus):  # seed only the RNGs used, restore them after
            torch.default_generator.manual_seed(cfg.seed)  # torch.manual_seed seeds every GPU
            for i in gpus:
                torch.cuda.default_generators[i].manual_seed(cfg.seed)
            self.model_ = FusedRentNet(*self._prep.net_shape, cfg).to(self.device_)
            self._train(train, val)
        return self

    def _batch(
        self,
        X_num: pd.DataFrame,
        X_cat: pd.DataFrame | None,
        X_text: TextInput | None,
        y_log: np.ndarray | pd.Series | None = None,
    ) -> _Batch:
        X_cat, y = _validate(X_num, X_cat, X_text, y_log)
        arrays = self._prep.transform(X_num, X_cat, X_text)
        if y is not None:
            arrays.append(((y - self.target_mean_) / self.target_std_).astype(np.float32))
        tensors = [None if a is None else torch.from_numpy(a).to(self.device_) for a in arrays]
        return _Batch(*tensors, y_log=y)

    def _train(self, train: _Batch, val: _Batch | None) -> None:
        cfg = self.config
        optimizer, scheduler = optim = _optimizer(self.model_, cfg, train.num.shape[0])
        generator = torch.Generator().manual_seed(cfg.seed)
        best_mae, best_epoch, best_state, history = math.inf, 0, None, []
        for epoch in range(1, cfg.epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            train_loss = self._train_epoch(train, optim, generator)
            val_mae = math.nan if val is None else self._mae_chf(val)
            history.append((epoch, train_loss, val_mae, lr))
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss if val is None else val_mae)
            if val is None or val_mae < best_mae:  # without eval_set: keep the last epoch
                best_mae, best_epoch = val_mae, epoch
                best_state = None if val is None else copy.deepcopy(self.model_.state_dict())
            elif epoch - best_epoch >= cfg.patience:
                break  # early stopping
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.history_ = pd.DataFrame(history, columns=["epoch", "train_loss", "val_mae", "lr"])
        self.best_epoch_, self.best_val_mae_ = best_epoch, best_mae
        msg = "FusedRentNet on %s: %d epochs, best epoch %d, best val MAE %.1f CHF"
        logger.info(msg, self.device_, len(history), self.best_epoch_, self.best_val_mae_)

    def _train_epoch(self, train: _Batch, optim: _Optim, generator: torch.Generator) -> float:
        (optimizer, scheduler), cfg, scale = optim, self.config, self.target_std_
        losses = {"huber": nn.HuberLoss(delta=cfg.huber_delta), "l1": nn.L1Loss()}
        loss_fn = losses.get(cfg.loss, nn.MSELoss())
        self.model_.train()
        n_rows = train.num.shape[0]
        order = torch.randperm(n_rows, generator=generator).to(self.device_)
        total = torch.zeros((), device=self.device_)
        for start in range(0, n_rows, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            x_cat = train.cat[idx]
            if cfg.cat_unknown_rate > 0 and x_cat.shape[1] > 0:
                mask = torch.rand(x_cat.shape, generator=generator) < cfg.cat_unknown_rate
                x_cat = x_cat.masked_fill(mask.to(self.device_), 0)
            text = None if train.text is None else train.text[idx]
            pred = self.model_(train.num[idx], x_cat, text)
            loss = loss_fn(pred * scale, train.y[idx] * scale)  # residuals on the log scale
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model_.parameters(), cfg.grad_clip)
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            total += loss.detach() * idx.numel()
        return float(total.item() / n_rows)

    def _run(self, batch: _Batch, *, embed: bool = False) -> np.ndarray:
        self.model_.eval()
        net = self.model_.representation if embed else self.model_
        with torch.no_grad():  # one pass: the inputs are already on the device in full
            out = net(batch.num, batch.cat, batch.text).cpu().numpy().astype(np.float64)
        return out if embed else out * self.target_std_ + self.target_mean_

    def _mae_chf(self, batch: _Batch) -> float:
        return float(np.mean(np.abs(np.exp(self._run(batch)) - np.exp(batch.y_log))))

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise NotFittedError("FusedRentRegressor is not fitted yet; call fit() first")

    def predict(
        self, X_num: pd.DataFrame, X_cat: pd.DataFrame | None, X_text: TextInput | None = None
    ) -> np.ndarray:
        """Predict ``log(price)``; ``np.exp`` of the result is the CHF estimate (median scale).

        Args:
            X_num: Numeric features with the training columns.
            X_cat: Categorical features with the training columns (None if fitted without).
            X_text: Text embeddings, required if and only if the model was fitted with them.

        Returns:
            Log-price predictions of shape (n_rows,).

        Raises:
            NotFittedError: If called before ``fit``.
            KeyError: If training columns are missing.
            ValueError: If rows are misaligned or ``X_text`` is inconsistent with training.
        """
        self._check_fitted()
        return self._run(self._batch(X_num, X_cat, X_text))

    def embed(
        self, X_num: pd.DataFrame, X_cat: pd.DataFrame | None, X_text: TextInput | None = None
    ) -> np.ndarray:
        """Return the fused per-listing representation (last hidden layer, eval mode).

        The coordinates belong to this network (arbitrary rotation and scale) and are fitted to
        its training targets: as features, use them only for rows it was not trained on and
        never mix networks (e.g. CV folds). For stacking, out-of-fold ``predict`` is safer.

        Args:
            X_num: Numeric features, as for :meth:`predict`.
            X_cat: Categorical features, as for :meth:`predict`.
            X_text: Text embeddings, as for :meth:`predict`.

        Returns:
            Array of shape (n_rows, config.hidden[-1]); raises like :meth:`predict`.
        """
        self._check_fitted()
        return self._run(self._batch(X_num, X_cat, X_text), embed=True)

    def embedding_table(self, col: str) -> pd.DataFrame:
        """Return the learned entity embedding of a categorical column.

        For inspection (maps, similar municipalities), not as model features: coordinates differ
        between networks (e.g. CV folds), and joined onto the network's own training rows they
        leak the target (worst for municipalities with few listings). See :meth:`embed`.

        Args:
            col: Name of a categorical training column.

        Returns:
            DataFrame indexed by category label (first row ``"<unknown>"``) with ``n_train``
            (training rows mapped to that index) and ``emb_0`` ... ``emb_{d-1}``.

        Raises:
            NotFittedError: If called before ``fit``.
            KeyError: If ``col`` was not a categorical training column.
        """
        self._check_fitted()
        if col not in self._prep.cat_cols:
            raise KeyError(f"{col!r} is not a categorical column; known: {self._prep.cat_cols}")
        j = self._prep.cat_cols.index(col)
        weights = self.model_.embeddings[j].weight.detach().cpu().numpy().astype(np.float64)
        labels = pd.Index([UNKNOWN_LABEL, *self._prep.vocabs[j]], name=str(col))
        table = pd.DataFrame(weights, index=labels).add_prefix("emb_")
        table.insert(0, "n_train", self._prep.counts[j])
        return table

    def __getstate__(self) -> dict[str, object]:
        """Pickle with the network on CPU so the file also loads on machines without CUDA."""
        state = self.__dict__.copy()
        if "model_" in state:
            state.update(model_=copy.deepcopy(self.model_).cpu(), device_=torch.device("cpu"))
        return state
