import pytest
from src.utils import *
import pandas as pd 

@pytest.fixture()
def sample_input_data():
    data = pd.read_csv(Inputs.datafile)
    data = data.head(20)
    return data