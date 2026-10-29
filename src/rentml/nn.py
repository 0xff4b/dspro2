"""Fused neural network for log rent (PyTorch, optional ``dl`` extra).

Entity embeddings for the location keys (weight decay shrinks rare units towards zero, a neural
analogue of random intercepts), an MLP on the numeric features and an optional branch for text
embeddings. Huber loss on log rent, early stopping on the validation MAE in CHF.
"""

import copy
import itertools
import logging
from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rentml.config import RANDOM_STATE

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("rentml.nn needs PyTorch: run `uv sync --extra dl`") from exc

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = "<unknown>"
MIN_CATEGORY_COUNT = 2  # rarer training units share the unknown index 0
HIDDEN = (256, 128)
DROPOUT = 0.15
HUBER_DELTA = 0.15  # log units, about 16 % rent error


def _block(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.SiLU(), nn.Dropout(DROPOUT)
    )


class _FusedNet(nn.Module):
    def __init__(self, n_numeric: int, cardinalities: Sequence[int], text_dim: int) -> None:
        super().__init__()
        dims = [min(16, round(1.6 * n**0.56)) for n in cardinalities]  # fast.ai rule of thumb
        self.embeddings = nn.ModuleList(map(nn.Embedding, cardinalities, dims))
        self.numeric = _block(n_numeric, 64)
        self.text = _block(text_dim, 32) if text_dim else None
        widths = [sum(dims) + 64 + (32 if text_dim else 0), *HIDDEN]
        self.head = nn.Sequential(*(_block(a, b) for a, b in itertools.pairwise(widths)))
        self.out = nn.Linear(widths[-1], 1)

    def forward(
        self, num: torch.Tensor, cat: torch.Tensor, text: torch.Tensor | None
    ) -> torch.Tensor:
        parts = [self.numeric(num), *(emb(cat[:, j]) for j, emb in enumerate(self.embeddings))]
        if self.text is not None:
            parts.append(self.text(text))
        return self.out(self.head(torch.cat(parts, dim=1))).squeeze(-1)


class FusedRentRegressor:
    """Entity embeddings + numeric MLP + optional text branch, trained on log rent.

    Args:
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        lr: AdamW learning rate (halved when the validation MAE plateaus).
        weight_decay: AdamW weight decay, also on the embeddings.
        patience: Early-stopping patience in epochs.
        log1p_cols: Skewed numeric columns to ``log1p`` (clipped at 0).
        seed: Seed of the initialisation and the shuffling.
        device: Torch device; None picks CUDA if available.

    Attributes:
        history_: ``epoch``, ``train_loss`` (Huber, log scale), ``val_mae`` (CHF).
        best_epoch_, best_val_mae_: Epoch whose weights were restored and its validation MAE.
    """

    def __init__(
        self,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        log1p_cols: Sequence[str] = (),
        seed: int = RANDOM_STATE,
        device: str | None = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.log1p_cols = log1p_cols
        self.seed = seed
        self.device = device

    def _numeric(self, X_num: pd.DataFrame) -> pd.DataFrame:
        X = X_num.astype(float)
        cols = [c for c in self.log1p_cols if c in X.columns]
        X[cols] = np.log1p(X[cols].clip(lower=0))
        return X

    def _tensors(
        self, X_num: pd.DataFrame, X_cat: pd.DataFrame, X_text: np.ndarray | None
    ) -> list[torch.Tensor | None]:
        num = self.scaler_.transform(self._numeric(X_num))
        cat = np.column_stack(
            [v.get_indexer(X_cat[c].astype("string")) + 1 for c, v in self.vocabs_.items()]
        )
        text = None
        if X_text is not None:  # NaN row = no description: zeros plus a missing flag
            missing = np.isnan(X_text).any(axis=1, keepdims=True)
            text = np.hstack([np.where(missing, 0.0, X_text), missing])
        arrays = [num.astype(np.float32), cat.astype(np.int64), text]
        dtypes = [torch.float32, torch.long, torch.float32]
        return [
            None if a is None else torch.as_tensor(a, dtype=d, device=self.device_)
            for a, d in zip(arrays, dtypes, strict=True)
        ]

    def fit(
        self,
        X_num: pd.DataFrame,
        X_cat: pd.DataFrame,
        y_log: pd.Series | np.ndarray,
        X_text: np.ndarray | None = None,
        eval_set: tuple | None = None,
    ) -> Self:
        """Fit preprocessing and network; ``eval_set`` is ``(X_num, X_cat, y_log[, X_text])``.

        Without ``eval_set`` all epochs run and the last weights are kept.
        """
        torch.manual_seed(self.seed)
        self.device_ = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler_ = make_pipeline(imputer, StandardScaler()).set_output(transform="default")
        self.scaler_.fit(self._numeric(X_num))
        self.vocabs_: dict[str, pd.Index] = {}
        self.counts_: dict[str, np.ndarray] = {}
        for col in X_cat.columns:
            freq = X_cat[col].astype("string").value_counts()
            known = freq[freq >= MIN_CATEGORY_COUNT]
            self.vocabs_[col] = pd.Index(known.index)
            self.counts_[col] = np.r_[len(X_cat) - known.sum(), known.to_numpy()]
        train = self._tensors(X_num, X_cat, X_text)
        y = torch.as_tensor(np.asarray(y_log, float), dtype=torch.float32, device=self.device_)
        text_dim = 0 if X_text is None else X_text.shape[1] + 1
        cards = [len(v) + 1 for v in self.vocabs_.values()]
        self.model_ = _FusedNet(train[0].shape[1], cards, text_dim).to(self.device_)
        with torch.no_grad():  # start at the mean log rent
            self.model_.out.bias.fill_(float(y.mean()))
        self._train(train, y, eval_set)
        return self

    def _train(self, train: list, y: torch.Tensor, eval_set: tuple | None) -> None:
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
        generator = torch.Generator().manual_seed(self.seed)
        history, best_state = [], None
        self.best_epoch_, self.best_val_mae_ = 0, np.inf
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            order = torch.randperm(len(y), generator=generator).to(self.device_)
            total = 0.0
            for idx in order.split(self.batch_size):
                text = None if train[2] is None else train[2][idx]
                loss = loss_fn(self.model_(train[0][idx], train[1][idx], text), y[idx])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                optimizer.step()
                total += loss.item() * len(idx)
            val_mae = np.nan
            if eval_set is not None:
                pred = np.exp(self.predict(*eval_set[:2], *eval_set[3:]))
                val_mae = float(np.mean(np.abs(pred - np.exp(np.asarray(eval_set[2], float)))))
                scheduler.step(val_mae)
            history.append((epoch, total / len(y), val_mae))
            if eval_set is None or val_mae < self.best_val_mae_:
                self.best_epoch_, self.best_val_mae_ = epoch, val_mae
                best_state = copy.deepcopy(self.model_.state_dict())
            elif epoch - self.best_epoch_ >= self.patience:
                break
        self.model_.load_state_dict(best_state)
        self.history_ = pd.DataFrame(history, columns=["epoch", "train_loss", "val_mae"])
        logger.info("Fused net: best epoch %d of %d", self.best_epoch_, len(history))

    def predict(
        self, X_num: pd.DataFrame, X_cat: pd.DataFrame, X_text: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict log rent."""
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(*self._tensors(X_num, X_cat, X_text))
        return out.cpu().numpy().astype(float)

    def embedding_table(self, col: str) -> pd.DataFrame:
        """Learned embedding of a key column: ``n_train`` and ``emb_*`` per unit (inspection only).

        The first row, ``UNKNOWN_LABEL``, holds the units with too few training objects.
        """
        j = list(self.vocabs_).index(col)
        weights = self.model_.embeddings[j].weight.detach().cpu().numpy()
        index = pd.Index([UNKNOWN_LABEL, *self.vocabs_[col]], name=col)
        table = pd.DataFrame(weights, index=index).add_prefix("emb_")
        table.insert(0, "n_train", self.counts_[col])
        return table
