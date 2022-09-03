from pathlib import Path
import typing as t

from pydantic import BaseModel, validator
from strictyaml import load, YAML
import packages

from packages import  get_project_root , ROOT_DIR
import os 


# Project Directories

# PACKAGE_ROOT = Path(packages.src.__file__).resolve().parent
PACKAGE_ROOT = get_project_root().resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "packages/src/config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "packages/src/trained_models"
DATASET_DIR = PACKAGE_ROOT / "packages/src/datasets"



class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    training_data_file: str
    test_data_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    # drop_features: str
    drop_features:t.Sequence[str]
    target: str
    variables_to_rename: t.Dict
    features: t.Sequence[str]
    numerical_vars: t.Sequence[str]
    one_hot_encoding_vars:t.Sequence[str]
    categorical_vars: t.Sequence[str]
    # temporal_vars: str
    numerical_vars_with_na: t.Sequence[str]
    numerical_na_not_allowed: t.Sequence[str]
    test_size: float
    random_state: int
    n_estimators: int
    rare_label_n_categories: int
    rare_label_tol: float

    # the order is necessary for validation
    allowed_loss_functions: t.Tuple[str, ...]
    loss: str

    @validator("loss")
    def allowed_loss_function(cls, value, values):
        """
        Loss function to be optimized.

        ‘log_loss’ refers to binomial and multinomial deviance, the same 
        as used in logistic regression. It is a good choice for classification 
        with probabilistic outputs. For loss ‘exponential’, 
        gradient boosting recovers the AdaBoost algorithm.

        Following the research phase, loss is restricted to
        `log_loss`  for this model.
        """

        allowed_loss_functions = values.get("allowed_loss_functions")
        print(allowed_loss_functions)
        if value in allowed_loss_functions:
            return value
        raise ValueError(
            f"the loss parameter specified: {value}, "
            f"is not in the allowed set: {allowed_loss_functions}"
        )


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
