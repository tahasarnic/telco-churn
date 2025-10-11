# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine.pipeline import make_pipeline
from feature_engine.selection import DropFeatures
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import RobustScaler
import hydra
from omegaconf import DictConfig
import joblib

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Read data
    df = pd.read_csv(cfg.data.data_path)

    # Data preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})
    df.to_csv(cfg.data.cleaned_data_path, index=False)

    # Split data
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state, stratify=y)

    # Preprocessing pipeline
    preprocessing_pipeline = make_pipeline(
        DropFeatures(features_to_drop=['customerID']),
        MeanMedianImputer(imputation_method='median', variables=['TotalCharges']),
        OrdinalEncoder(encoding_method='ordered'),
        SklearnTransformerWrapper(transformer=RobustScaler(), variables=['tenure', 'MonthlyCharges', 'TotalCharges'])
    )

    X_train_scaled = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_test_scaled = preprocessing_pipeline.transform(X_test)

    # Save preprocessing pipeline
    joblib.dump(preprocessing_pipeline, cfg.data.preprocessing_pipeline_path)

    # Save processed data
    X_train_scaled.to_csv(cfg.data.train_path, index=False)
    y_train.to_csv(cfg.data.y_train_path, index=False)
    X_test_scaled.to_csv(cfg.data.test_path, index=False)
    y_test.to_csv(cfg.data.y_test_path, index=False)

if __name__ == "__main__":
    main()