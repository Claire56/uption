# import sys , os
import pytest
from src.utils import *
import pandas as pd 
from sklearn.model_selection import train_test_split


def test_status_unique_values(sample_input_data):
    # given 
    data = sample_input_data
    unique_values = data['status'].unique()
    assert len(unique_values) == 2

class TestTraindata:
    @pytest.fixture
    def get_train(sample_input_data):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =.3, random_state = 0)
        train = X_train[:30]
    def test_train(train):
        assert train.shape[0] == 30
        

        