# Packages
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

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
    # Read cleaned data
    X_train = pd.read_csv(cfg.data.train_path)
    y_train = pd.read_csv(cfg.data.y_train_path)

    # Model training and evaluation
    experiment_results = {
        "Model": [],
        "Mean CV Score": []
    }

    for model_name in cfg.selection.models:
        model_cfg = OmegaConf.load(f'configs/model/{model_name}.yaml')
        model = get_model(model_name, model_cfg.params)
        score = cross_val_score(model, X_train, y_train, cv=5)
        experiment_results["Model"].append(model_name)
        experiment_results["Mean CV Score"].append(score.mean())

    # Determine best model
    best_model_name = max(experiment_results["Model"], key=lambda k: experiment_results["Mean CV Score"][experiment_results["Model"].index(k)])
    best_score = experiment_results["Mean CV Score"][experiment_results["Model"].index(best_model_name)]

    # Save results using OmegaConf
    output = OmegaConf.create({
        'best_model': best_model_name,
        'best_score': float(best_score),  # Convert numpy float to Python float
        'results': {
            'models': experiment_results["Model"],
            'cv_scores': [float(score) for score in experiment_results["Mean CV Score"]]
        }
    })
    OmegaConf.save(output, 'model_selection_results.yaml')

if __name__ == "__main__":
    main()