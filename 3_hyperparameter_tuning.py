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

    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=dict(model_config.param_grid),
        cv=cfg.tuning.cv_folds,
        scoring=cfg.tuning.scoring,
        n_jobs=-cfg.tuning.n_jobs,
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Evaluate on test set
    test_score = grid_search.score(X_test, y_test)
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")
    print(f"Test score: {test_score}")

    # Save results using OmegaConf
    output = OmegaConf.create({
        'model_name': best_model_name,
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),  # Convert numpy float to Python float
        'test_score': float(test_score),  # Convert numpy float to Python float
        'metric': cfg.tuning.scoring,
    })
    OmegaConf.save(output, 'tuning_results.yaml')

if __name__ == "__main__":
    main()
