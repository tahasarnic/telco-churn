# Package imports
from feature_engine import imputation as fti
from feature_engine import encoding as fte
from feature_engine import scaling as fts
from feature_engine import selection as ftsl
from feature_engine.pipeline import make_pipeline

class TelcoPreprocessor:
    def __init__(self,
                 features_to_drop=None,
                 features_to_impute=None,
                 features_to_encode=None,
                 features_to_scale=None,
                 imputation_method='median',
                 drop_last=True) -> None:
        self.features_to_drop = features_to_drop
        self.features_to_impute = features_to_impute
        self.features_to_encode = features_to_encode
        self.features_to_scale = features_to_scale
        self.imputation_method = imputation_method
        self.drop_last = drop_last

    def build_pipeline(self, X, y=None):
        self.pipeline = make_pipeline(
            ftsl.DropFeatures(features_to_drop=self.features_to_drop),
            fti.MeanMedianImputer(variables=self.features_to_impute, imputation_method=self.imputation_method),
            fte.OneHotEncoder(variables=self.features_to_encode, drop_last=self.drop_last),
            fts.MeanNormalizationScaler(variables=self.features_to_scale)
        )
        if y is not None:
            self.pipeline.fit(X, y)
        else:
            self.pipeline.fit(X)
        return self.pipeline.transform(X)
