# Package imports
import pandas as pd
from feature_engine import imputation as fti
from feature_engine import encoding as fte
from feature_engine import scaling as fts
from feature_engine import selection as ftsl
from feature_engine.pipeline import make_pipeline

# Building a preprocessing pipeline function
def simple_preprocessing_pipeline(
                   X ,
                   y = None,
                   features_to_drop: list = None,
                   features_to_impute: list = None,
                   imputation_method: str = 'median',
                   features_to_encode: list = None,
                   drop_last: bool = True,
                   features_to_scale: list = None):
    """
    A simple preprocessing pipeline that includes feature dropping, imputation, encoding, and scaling.
    """

    pipeline = make_pipeline(
        ftsl.DropFeatures(features_to_drop=features_to_drop),
        fti.MeanMedianImputer(variables=features_to_impute, imputation_method=imputation_method),
        fte.OneHotEncoder(variables=features_to_encode, drop_last=drop_last),
        fts.MeanNormalizationScaler(variables=features_to_scale)
    )
    if y is not None:
        pipeline.fit(X, y)
    else:
        pipeline.fit(X)
    return pipeline

# Building a preprocessing pipeline function
def complex_preprocessing_pipeline(
                   X ,
                   y = None,
                   features_to_drop: list = None,
                   features_to_impute: list = None,
                   imputation_method: str = 'median',
                   features_to_encode: list = None,
                   features_to_scale: list = None):
    """
    A simple preprocessing pipeline that includes feature dropping, imputation, encoding, and scaling.
    """

    pipeline = make_pipeline(
        ftsl.DropFeatures(features_to_drop=features_to_drop),
        fti.MeanMedianImputer(variables=features_to_impute, imputation_method=imputation_method),
        fte.OrdinalEncoder(variables=features_to_encode),
        fts.MeanNormalizationScaler(variables=features_to_scale)
    )
    if y is not None:
        pipeline.fit(X, y)
    else:
        pipeline.fit(X)
    return pipeline