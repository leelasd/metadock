# tests/conftest.py
from pathlib import Path
import pytest
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_poses_sdf():
    return FIXTURES / "mini_poses.sdf"


@pytest.fixture
def mini_sar_csv():
    return FIXTURES / "mini_sar.csv"


@pytest.fixture
def mini_sar_df():
    return pd.read_csv(FIXTURES / "mini_sar.csv", parse_dates=["assay_date"])
