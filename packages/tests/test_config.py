import pytest
from packages.src.utils import DataDetails , Utilities
import pandas as pd 


from pathlib import Path

from packages.src.config.core import (
    create_and_validate_config,
    fetch_config_from_yaml,
)

import pytest
from pydantic import ValidationError


TEST_CONFIG_TEXT = """
package_name: src
training_data_file: startup_data.csv
test_data_file: test.csv
drop_features:
    - name
pipeline_name: startup_classifier
pipeline_save_file: startup_classifier_output_v
target: labels
variables_to_rename:
    has_roundA: has_roundA
test_size: 0.1
features:
  - relationships
numerical_vars:
  - relationships
categorical_vars:
  - category_code
one_hot_encoding_vars:
  - category_code
numerical_vars_with_na:
  - age_first_milestone_year
numerical_na_not_allowed:
  - is_CA
random_state: 0
n_estimators: 50
rare_label_tol: 0.01
rare_label_n_categories: 5
loss: log_loss
allowed_loss_functions:
  - log_loss
  
"""

INVALID_TEST_CONFIG_TEXT = """
package_name: src
training_data_file: startup_data.csv
test_data_file: test.csv
drop_features: 
    - name
pipeline_name: startup_classifier
pipeline_save_file: startup_classifier_output_v
target: labels
features:
  - relationships
numerical_vars:
  - relationships
categorical_vars:
  - category_codeal
# temporal_vars: YearRemodAdd
numerical_vars_with_na:
  - age_first_milestone_year
numerical_na_not_allowed:
  - is_CA
one_hot_encoding_vars:
  - category_code
random_state: 0
n_estimators: 50
rare_label_tol: 0.01
rare_label_n_categories: 5
loss: ls
allowed_loss_functions:
  - huber
"""


def test_fetch_config_structure(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    config = create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert config.model_config
    assert config.app_config


def test_config_validation_raises_error_for_invalid_config(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"

    # invalid config attempts to set a prohibited loss
    # function which we validate against an allowed set of
    # loss function parameters.
    config_1.write_text(INVALID_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "not in the allowed set" in str(excinfo.value)


def test_missing_config_field_raises_validation_error(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    TEST_CONFIG_TEXT = """package_name: src"""
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "field required" in str(excinfo.value)
    assert "pipeline_name" in str(excinfo.value)
