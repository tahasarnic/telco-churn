from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING

from config_schemas import preprocessing_schema

@dataclass
class Config:
    preprocessing: preprocessing_schema.PreprocessorConfig = MISSING


def setup_config() -> None:
    preprocessing_schema.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
