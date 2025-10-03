from dataclasses import dataclass
from typing import Any
from hydra.core.config_store import ConfigStore
from config_schemas.preprocessing_schema import SimplePreprocessingConfig, ComplexPreprocessingConfig


@dataclass
class Config:
    """Main configuration schema."""
    preprocessing: Any


def register_configs():
    """Register all configuration schemas with Hydra."""
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
    cs.store(group="preprocessing", name="simple_preprocessor_schema", node=SimplePreprocessingConfig)
    cs.store(group="preprocessing", name="complex_preprocessor_schema", node=ComplexPreprocessingConfig)
