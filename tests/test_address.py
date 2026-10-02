"""Tests for rentml.address (address string parsing; moved from test_geo.py)."""

import numpy as np
import pandas as pd
import pytest

from rentml.address import parse_address


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        (
            "8044, Gockhausen-Zürich, Rütistrasse, 1",
            (8044, "Gockhausen-Zürich", "Rütistrasse", "1"),
        ),
        ("1030, Bussigny, Rue de Lausanne 52 D, 52D", (1030, "Bussigny", "Rue de Lausanne", "52D")),
        ("1170, Aubonne, **, rue des Granges, 6", (1170, "Aubonne", "rue des Granges", "6")),
        ("Chemin du Grand Praz 3, 1012, Lausanne", (1012, "Lausanne", "Chemin du Grand Praz", "3")),
        ("Wintergasse **, 4056, Basel", (4056, "Basel", "Wintergasse", None)),
        ("Gellertstrasse 9-13a, 4052 Basel", (4052, "Basel", "Gellertstrasse", "9-13a")),
        ("Maulbeerstrasse 1 4058 Basel", (4058, "Basel", "Maulbeerstrasse", "1")),
        ("Chemin du Devin 47C, CH-1012, Lausanne", (1012, "Lausanne", "Chemin du Devin", "47C")),
        ("Prattelerstrasse **, Muttenz 4132, Schweiz", (4132, "Muttenz", "Prattelerstrasse", None)),
        ("Lehenmattstr, 201, 4052 Basel", (4052, "Basel", "Lehenmattstr", "201")),
        ("1260, Nyon, 8 Chem. du Chêne, 8", (1260, "Nyon", "Chem. du Chêne", "8")),
        ("Rieterstrasse **, Rieterstrasse 36, Zurich", (None, "Zurich", "Rieterstrasse", "36")),
        ("** Steinackerstrasse, Dietikon, 8953", (8953, "Dietikon", "Steinackerstrasse", None)),
        ("6403 Küssnacht am Rigi", (6403, "Küssnacht am Rigi", None, None)),
        ("Carouge", (None, "Carouge", None, None)),
        ("Gärtnerstrasse 69", (None, None, "Gärtnerstrasse", "69")),
    ],
)
def test_parse_address_variants(address: str, expected: tuple) -> None:
    row = parse_address(pd.Series([address])).iloc[0]
    postcode = None if pd.isna(row["postcode"]) else int(row["postcode"])
    assert (postcode, row["locality"], row["street"], row["house_number"]) == expected


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("1955, Chamoson, Route de Vayeplane 14, 2A", (1955, "Route de Vayeplane", "2A")),
        ("1723, Marly, Rue des Frères-Lumière 23, 27", (1723, "Rue des Frères-Lumière", "27")),
        ("Rue de Lausanne 25 bis, 1422 Grandson", (1422, "Rue de Lausanne", "25 bis")),
        ("1400, Yverdon, Rue des Philosophes 20bis", (1400, "Rue des Philosophes", "20bis")),
        ("Carl-Spitteler-Str. 20/22/24, 8053 Zürich", (8053, "Carl-Spitteler-Str.", "20/22/24")),
        ("2800, Delémont, Rue Du 23-Juin, 48", (2800, "Rue Du 23-Juin", "48")),
    ],
)
def test_parse_address_strips_embedded_house_number(address: str, expected: tuple) -> None:
    row = parse_address(pd.Series([address])).iloc[0]
    assert (int(row["postcode"]), row["street"], row["house_number"]) == expected


def test_parse_address_handles_missing_and_malformed() -> None:
    raw = pd.Series([None, np.nan, 123, "", "**", " , ,"], index=[10, 11, 12, 13, 14, 15])
    out = parse_address(raw)
    assert list(out.index) == list(raw.index)
    assert list(out.columns) == ["postcode", "locality", "street", "house_number"]
    assert out["postcode"].dtype == "Int64"
    assert out.isna().all().all()
