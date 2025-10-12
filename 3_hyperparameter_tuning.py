# Packages
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
# MLflow
import mlflow
import mlflow.sklearn
# Scikit-learn modules
from sklearn.model_selection import GridSearchCV


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Hyperparameter tuning for the best model from model selection"""

    # Set up MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Load model selection results with OmegaConf
    selection_results = OmegaConf.load("model_selection_results.yaml")
    best_model_name = selection_results.best_model

    # Load the corresponding model config
    model_config = OmegaConf.load(f"configs/model/{best_model_name}.yaml")

    # Get the sklearn module path and instantiate the model
    model_class_path = model_config.model

    # Instantiate model with base parameters using Hydra's instantiate
    model = instantiate(model_config.params, _target_=model_class_path)

    # Data
    X_train = pd.read_csv(cfg.data.train_path)
    y_train = pd.read_csv(cfg.data.y_train_path)
    X_test = pd.read_csv(cfg.data.test_path)
    y_test = pd.read_csv(cfg.data.y_test_path)

    # Start MLflow run for hyperparameter tuning
    with mlflow.start_run(run_name=f"hyperparameter_tuning_{best_model_name}"):
        # Log dataset information
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("model_name", best_model_name)
        mlflow.log_param("model_class", model_class_path)

        # Log tuning configuration
        mlflow.log_param("cv_folds", cfg.tuning.cv_folds)
        mlflow.log_param("scoring_metric", cfg.tuning.scoring)
        mlflow.log_param("n_jobs", cfg.tuning.n_jobs)

        # Log the parameter grid being searched
        mlflow.log_dict(dict(model_config.param_grid), "param_grid.json")

        # Log base model parameters
        mlflow.log_params({f"base_{k}": v for k, v in model_config.params.items()})

        # Log model selection score for comparison
        mlflow.log_metric("model_selection_cv_score", float(selection_results.best_score))

        # Hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=dict(model_config.param_grid),
            cv=cfg.tuning.cv_folds,
            scoring=cfg.tuning.scoring,
            n_jobs=cfg.tuning.n_jobs,
        )

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Log best hyperparameters
        mlflow.log_params({f"best_{k}": v for k, v in grid_search.best_params_.items()})

        # Log performance metrics
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        mlflow.log_metric("best_cv_score_std", grid_search.cv_results_['std_test_score'][grid_search.best_index_])

        # Evaluate on test set
        test_score = grid_search.score(X_test, y_test)
        mlflow.log_metric("test_score", test_score)

        # Log number of combinations tried
        mlflow.log_metric("n_combinations_tried", len(grid_search.cv_results_['mean_test_score']))

        # Log the trained model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")

        # Log the tuning results file
        output = OmegaConf.create({
            'model_name': best_model_name,
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'test_score': float(test_score),
            'metric': cfg.tuning.scoring,
        })
        
        OmegaConf.save(output, 'tuning_results.yaml')
        mlflow.log_artifact('tuning_results.yaml')

        print(f"Best hyperparameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(f"Test score: {test_score:.4f}")
        print(f"Number of combinations tried: {len(grid_search.cv_results_['mean_test_score'])}")

if __name__ == "__main__":
    main()
