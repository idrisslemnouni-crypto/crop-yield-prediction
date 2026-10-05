import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs" / "default.json").read_text())


@pytest.fixture
def weather():
    return pd.DataFrame(
        [
            {
                "COUNTY_ID": "IA_TEST",
                "FYEAR": 2000,
                "DEKAD": d,
                "TAVG": 20.0,
                "TMIN": 10.0,
                "TMAX": 30.0,
                "PREC": 10.0,
                "ET0": 15.0,
            }
            for d in range(1, 37)
        ]
    )
