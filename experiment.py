# Packages
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import mlflow
import mlflow.sklearn


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set MLflow experiment
    mlflow.set_experiment("telco-churn-experiments")

    # Start MLflow run
    with mlflow.start_run(run_name="model_comparison"):
        # Log Hydra config with prefix to avoid conflicts
        preprocessing_params = OmegaConf.to_container(cfg.preprocessing, resolve=True)
        preprocessing_params["preprocessing_pipeline"] = cfg.preprocessing._target_
        preprocessing_params_prefixed = {f"preprocessing_{k}": v for k, v in preprocessing_params.items()}
        mlflow.log_params(preprocessing_params_prefixed)

        # Data
        df = pd.read_csv("data/cleaned_data.csv")
        mlflow.log_param("data_file", "data/cleaned_data.csv")
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocess data using the preprocessing pipeline from Hydra config
        preprocessing_pipeline = instantiate(cfg.preprocessing,
                                             X=X_train,
                                             y=y_train,
                                             _convert_="all")
        X_train = preprocessing_pipeline.transform(X_train)
        X_test = preprocessing_pipeline.transform(X_test)

        # Log preprocessing pipeline
        mlflow.sklearn.log_model(preprocessing_pipeline, "preprocessing_pipeline")

        # Models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(),
            # For random forest we need to set some hyperparameters to not overfit
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }

        # Evaluate models
        results = {}
        for model_name, model in models.items():
            # Create nested run for each model
            with mlflow.start_run(run_name=model_name, nested=True):
                # Log model parameters with prefix
                model_params = {f"model_{k}": v for k, v in model.get_params().items()}
                mlflow.log_params(model_params)

                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='balanced_accuracy')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_score = balanced_accuracy_score(y_test, y_pred)

                # Log metrics
                mlflow.log_metric("cv_mean_balanced_accuracy", cv_scores.mean())
                mlflow.log_metric("cv_std_balanced_accuracy", cv_scores.std())
                mlflow.log_metric("test_balanced_accuracy", test_score)

                # Log model
                mlflow.sklearn.log_model(model, f"model_{model_name.replace(' ', '_')}")

                results[model_name] = {
                    "CV Mean Balanced Accuracy": cv_scores.mean(),
                    "Test Balanced Accuracy": test_score,
                    "Classification Report": classification_report(y_test, y_pred)
                }

        # Print results
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  CV Mean Balanced Accuracy: {metrics['CV Mean Balanced Accuracy']:.4f}")
            print(f"  Test Balanced Accuracy: {metrics['Test Balanced Accuracy']:.4f}")
            print(f"\n{metrics['Classification Report']}")

        # Compare CV and test scores and save the best model
        best_model_name = max(results, key=lambda name: results[name]["Test Balanced Accuracy"])
        best_model = models[best_model_name]
        print(f"\nBest Model: {best_model_name} with Test Balanced Accuracy: {results[best_model_name]['Test Balanced Accuracy']:.4f}")

        # Log best model info
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_test_balanced_accuracy", results[best_model_name]['Test Balanced Accuracy'])

        # Map model names to their sklearn target paths
        model_target_mapping = {
            "Logistic Regression": "sklearn.linear_model.LogisticRegression",
            "K-Nearest Neighbors": "sklearn.neighbors.KNeighborsClassifier",
            "Support Vector Machine": "sklearn.svm.SVC",
            "Random Forest": "sklearn.ensemble.RandomForestClassifier"
        }

        # Create model config with _target_ and parameters
        best_model_config = {
            "model": {
                "_target_": model_target_mapping[best_model_name]
            },
            "model_name": best_model_name
        }

        # Add model-specific parameters
        if best_model_name == "Logistic Regression":
            best_model_config["model"]["max_iter"] = 1000
        elif best_model_name == "Random Forest":
            best_model_config["model"]["n_estimators"] = 100
            best_model_config["model"]["max_depth"] = 10
            best_model_config["model"]["random_state"] = 42

        # Save to YAML file
        with open("configs/best_model.yaml", "w") as f:
            OmegaConf.save(best_model_config, f)

        print(f"\nBest model config saved to configs/best_model.yaml")

        # Log best model config and joblib file as artifacts
        mlflow.log_artifact("configs/best_model.yaml")

if __name__ == "__main__":
    main()