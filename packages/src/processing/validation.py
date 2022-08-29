import typing as t

from src.config.core import config

import numpy as np
import pandas as pd
from marshmallow import fields, Schema, ValidationError


class StartupDataInputSchema(Schema):
    age_first_funding_year = fields.Float(allow_none=True)
    age_last_funding_year = fields.Float(allow_none=True)
    age_first_milestone_year = fields.Float(allow_none=True)
    age_last_milestone_year = fields.Float(allow_none=True)
    relationships = fields.Integer(allow_none=True)
    funding_rounds = fields.Integer(allow_none=True)
    funding_total_usd = fields.Float(allow_none=True)
    milestones = fields.Float(allow_none=True)
    is_CA = fields.Integer(allow_none=True)
    is_NY = fields.Integer(allow_none=True)
    is_MA = fields.Integer(allow_none=True)
    is_TX = fields.Integer(allow_none=True)
    is_otherstate = fields.Integer(allow_none=True)
    is_software = fields.Integer(allow_none=True)
    is_web = fields.Integer(allow_none=True)
    is_mobile = fields.Integer(allow_none=True)
    is_enterprise = fields.Integer(allow_none=True)
    is_advertising = fields.Integer(allow_none=True)
    is_gamesvideo = fields.Integer(allow_none=True)
    is_ecommerce = fields.Integer(allow_none=True)
    is_biotech = fields.Integer(allow_none=True)
    is_consulting = fields.Integer(allow_none=True)
    is_othercategory = fields.Integer(allow_none=True)
    has_VC = fields.Integer(allow_none=True)
    has_angel = fields.Integer(allow_none=True)
    has_roundA = fields.Integer(allow_none=True)
    has_roundB = fields.Integer()
    has_roundC = fields.Integer()
    has_roundD = fields.Integer()
    avg_participants = fields.Integer()
    is_top500 = fields.Integer()
    category_code = fields.Str()
       


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    if input_data[config.model_config.numerical_na_not_allowed].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.model_config.numerical_na_not_allowed
        )

    return validated_data


def validate_inputs(
    *, input_data: pd.DataFrame
) -> t.Tuple[pd.DataFrame, t.Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    validated_data = drop_na_inputs(input_data=input_data)

    # set many=True to allow passing in a list
    schema = StartupDataInputSchema(many=True)
    errors = None

    try:
        # replace numpy nans so that Marshmallow can validate
        schema.load(validated_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as exc:
        errors = exc.messages

    return validated_data, errors
