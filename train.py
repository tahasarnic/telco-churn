# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from config_schemas.config_schema import register_configs

# Register configuration schemas
register_configs()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    df = pd.read_csv("data/cleaned_data.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Access preprocessing parameters
    # Convert OmegaConf objects to regular Python objects for instantiate
    pipeline = instantiate(
        cfg.preprocessing,
        X=X_train,
        y=y_train,  # optional
        _convert_="all"  # Convert all OmegaConf objects to Python primitives
    )
    
    # Use the pipeline
    transformed_data = pipeline.transform(X_test)
    print(transformed_data)

if __name__ == "__main__":
    main()