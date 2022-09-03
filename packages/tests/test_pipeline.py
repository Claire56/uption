from packages.src import pipeline
from packages.src.config.core import config
from packages.src.processing.validation import validate_inputs


def test_pipeline_drops_unnecessary_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    print(X_train.columns)
    print('##########################################')
    print(config.model_config.drop_features)
    assert config.model_config.drop_features[0] in X_train.columns
    pipeline.startup_pipe.fit(X_train, y_train)

    # When
    # We access the transformed inputs with slicing
    transformed_inputs = pipeline.startup_pipe[:-1].transform(X_train)
    print(f'\n #####\n {transformed_inputs.columns}')

    # Then
    assert config.model_config.drop_features[0] in X_train.columns
    assert config.model_config.drop_features[0] not in transformed_inputs.columns



# def test_pipeline_predict_takes_validated_input(pipeline_inputs, sample_input_data):
#     # Given
#     X_train, X_test, y_train, y_test = pipeline_inputs
#     pipeline.startup_pipe.fit(X_train, y_train)

#     # When
#     validated_inputs, errors = validate_inputs(input_data=sample_input_data)
#     predictions = pipeline.startup_pipe.predict(
#         validated_inputs[config.model_config.features]
#     )

#     # Then
#     assert predictions is not None
#     assert errors is None
