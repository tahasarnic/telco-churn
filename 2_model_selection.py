# Packages
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.sklearn

# Scikit-learn modules
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



# Function to get model based on name
def get_model(model_name: str, params: dict):
    """Factory function for models"""
    models = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'svm': SVC,
        'knn': KNeighborsClassifier
    }
    return models[model_name](**params)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set up MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Read cleaned data
    X_train = pd.read_csv(cfg.data.train_path)
    y_train = pd.read_csv(cfg.data.y_train_path)

    # Start parent MLflow run for the entire model selection experiment
    with mlflow.start_run(run_name="model_selection_experiment") as parent_run:
        # Log dataset metadata
        mlflow.log_param("n_samples", X_train.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("data_train_path", cfg.data.train_path)
        mlflow.log_param("data_test_path", cfg.data.test_path)

        # Model training and evaluation
        experiment_results = {
            "Model": [],
            "Mean CV Score": []
        }

        for model_name in cfg.selection.models:
            model_cfg = OmegaConf.load(f'configs/model/{model_name}.yaml')
            model = get_model(model_name, model_cfg.params)

            # Start MLflow run for this model (nested under parent run)
            with mlflow.start_run(run_name=model_name, nested=True):
                # Log parameters
                mlflow.log_params(model_cfg.params)
                mlflow.log_param("cv_folds", cfg.selection.cv_folds)
                mlflow.log_param("scoring_metric", cfg.selection.scoring)

                # Perform cross-validation
                scores = cross_val_score(model, X_train, y_train, cv=cfg.selection.cv_folds, scoring=cfg.selection.scoring)

                # Log metrics
                mlflow.log_metric("mean_cv_score", scores.mean())
                mlflow.log_metric("std_cv_score", scores.std())

                # Store results
                experiment_results["Model"].append(model_name)
                experiment_results["Mean CV Score"].append(scores.mean())

        # Determine best model
        best_model_name = max(experiment_results["Model"], key=lambda k: experiment_results["Mean CV Score"][experiment_results["Model"].index(k)])
        best_score = experiment_results["Mean CV Score"][experiment_results["Model"].index(best_model_name)]

        # Log best model info to parent run
        mlflow.log_metric("best_model_score", best_score)
        mlflow.log_param("best_model_name", best_model_name)

        # Save results using OmegaConf
        output = OmegaConf.create({
            'best_model': best_model_name,
            'best_score': float(best_score),  # Convert numpy float to Python float
            'metric': cfg.selection.scoring,
            'results': {
                'models': experiment_results["Model"],
                'cv_scores': [float(score) for score in experiment_results["Mean CV Score"]]
            }
        })
        OmegaConf.save(output, 'model_selection_results.yaml')

        # Log the results file as an MLflow artifact
        mlflow.log_artifact('model_selection_results.yaml')

if __name__ == "__main__":
    main()