from sys import getwindowsversion
import pytest
from src.utils import *
import pandas as pd 

@pytest.fixture()
def sample_input_data():
    data = pd.read_csv(Inputs.datafile)
    data = data.head(20)
    return data

def test_status_unique_values(sample_input_data):
    # given 
    data = sample_input_data
    unique_values = data['status'].unique()
    assert len(unique_values) == 2
    