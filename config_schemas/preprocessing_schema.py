from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from dataclasses import dataclass

@dataclass
class PreprocessorConfig:
    _target_: str = "src.preprocessing.TelcoPreprocessor"
    features_to_drop: list[str] = MISSING
    features_to_impute: list[str] = MISSING
    features_to_encode: list[str] = MISSING
    features_to_scale: list[str] = MISSING
    imputation_method: str = 'median'
    drop_last: bool = True
    _convert_: str = "all"

def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="preprocessing", name="preprocessing_schema", node=PreprocessorConfig)
