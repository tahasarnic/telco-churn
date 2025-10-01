# Packages
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from config_schemas.config_schema import setup_config

# Setup the config
setup_config()

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(config: DictConfig) -> None:
    # Data
    df = pd.read_csv('data/cleaned_data.csv')
    
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Instantiate the preprocessor
    preprocessor = instantiate(config.preprocessing)
    
    # Fit and transform the data
    X_processed = preprocessor.build_pipeline(X, y)
    
    print(X_processed.head())

if __name__ == "__main__":
    train()