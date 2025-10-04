# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from config_schemas.config_schema import register_configs
import joblib
import mlflow
import mlflow.sklearn

# Register configuration schemas
register_configs()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set MLflow experiment
    mlflow.set_experiment("telco-churn-training")

    # Start MLflow run
    with mlflow.start_run(run_name="final_training"):
        # Load data
        df = pd.read_csv("data/cleaned_data.csv")
        mlflow.log_param("data_file", "data/cleaned_data.csv")
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Log preprocessing config
        preprocessing_params = OmegaConf.to_container(cfg.preprocessing, resolve=True)
        preprocessing_params["preprocessing_pipeline"] = cfg.preprocessing._target_
        # Prefix preprocessing params to avoid conflicts
        preprocessing_params_prefixed = {f"preprocessing_{k}": v for k, v in preprocessing_params.items()}
        mlflow.log_params(preprocessing_params_prefixed)

        # Instantiate preprocessing pipeline
        pipeline = instantiate(
            cfg.preprocessing,
            X=X_train,
            y=y_train,  # optional
            _convert_="all"  # Convert all OmegaConf objects to Python primitives
        )

        # Use the pipeline
        X_train_scaled = pipeline.transform(X_train)
        X_test_scaled = pipeline.transform(X_test)

        # Log preprocessing pipeline as artifact
        mlflow.sklearn.log_model(pipeline, "preprocessing_pipeline")

        # Load best model config and instantiate the model
        best_model_cfg = OmegaConf.load("configs/best_model.yaml")
        model = instantiate(best_model_cfg.model)

        # Log model info
        mlflow.log_param("model_name", best_model_cfg.model_name)
        mlflow.log_param("model_target", best_model_cfg.model._target_)

        # Prefix model params to avoid conflicts
        model_params = {f"model_{k}": v for k, v in model.get_params().items()}
        mlflow.log_params(model_params)

        print(f"\nTraining {best_model_cfg.model_name}...")

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)

        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")

        # Save the trained model
        joblib.dump(model, "trained_model.joblib")
        print(f"\nTrained model saved to trained_model.joblib")

        # Log model and artifacts to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("configs/best_model.yaml")


if __name__ == "__main__":
    main()