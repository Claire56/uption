import pytest
import pandas as pd 

from sklearn.model_selection import train_test_split

from packages.src.config.core import config
from packages.src.processing.data_management import load_dataset
from packages.src.utils import *
# from src.utils import *




@pytest.fixture()
def sample_input_data():
    data = pd.read_csv(DataDetails.datafile)
    data = data.head(20)
    return data

# @pytest.mark.parameterize(
#     'size' , [4,5,7])
