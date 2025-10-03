from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import MISSING


@dataclass
class SimplePreprocessingConfig:
    """Configuration schema for simple preprocessing pipeline (OneHotEncoder)."""

    _target_: str = "preprocessing.simple_preprocessing_pipeline"

    # Feature dropping
    features_to_drop: Optional[List[str]] = None

    # Imputation
    features_to_impute: Optional[List[str]] = None
    imputation_method: str = "median"

    # Encoding
    features_to_encode: Optional[List[str]] = None
    drop_last: bool = True

    # Scaling
    features_to_scale: Optional[List[str]] = None


@dataclass
class ComplexPreprocessingConfig:
    """Configuration schema for complex preprocessing pipeline (OrdinalEncoder)."""

    _target_: str = "preprocessing.complex_preprocessing_pipeline"

    # Feature dropping
    features_to_drop: Optional[List[str]] = None

    # Imputation
    features_to_impute: Optional[List[str]] = None
    imputation_method: str = "median"

    # Encoding
    features_to_encode: Optional[List[str]] = None

    # Scaling
    features_to_scale: Optional[List[str]] = None
