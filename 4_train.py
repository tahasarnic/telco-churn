# Packages
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
from pathlib import Path

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Scikit-learn modules
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Train final model using best hyperparameters from MLflow"""

    # Set up MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    # Get the best run from hyperparameter tuning experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{cfg.mlflow.experiment_name}' not found. Run hyperparameter tuning first.")

    # Search for the best hyperparameter tuning run
    # Filter by run name pattern and order by the scoring metric
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'hyperparameter_tuning_%'",
        order_by=[f"metrics.best_cv_score DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No hyperparameter tuning runs found. Run hyperparameter tuning first.")

    best_tuning_run = runs[0]
    best_run_id = best_tuning_run.info.run_id

    print(f"Found best hyperparameter tuning run: {best_tuning_run.info.run_id}")
    print(f"Run name: {best_tuning_run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"Best CV score: {best_tuning_run.data.metrics.get('best_cv_score', 'N/A'):.4f}")
    print(f"Test score: {best_tuning_run.data.metrics.get('test_score', 'N/A'):.4f}")

    # Extract model information
    model_name = best_tuning_run.data.params.get('model_name')
    model_class_path = best_tuning_run.data.params.get('model_class')

    print(f"\nModel: {model_name}")
    print(f"Loading best model from MLflow run: {best_run_id}")

    # Load the best model directly from MLflow (no need to extract and convert parameters!)
    model_uri = f"runs:/{best_run_id}/best_model"
    model = mlflow.sklearn.load_model(model_uri)

    # Get the model's parameters for logging
    best_params = model.get_params()

    print(f"Best hyperparameters: {best_params}")

    # Load data
    X_train = pd.read_csv(cfg.data.train_path)
    y_train = pd.read_csv(cfg.data.y_train_path).values.ravel()
    X_test = pd.read_csv(cfg.data.test_path)
    y_test = pd.read_csv(cfg.data.y_test_path).values.ravel()

    # Create a new experiment for final model training
    final_experiment_name = "final_model"
    mlflow.set_experiment(final_experiment_name)

    # Start MLflow run for final model training
    with mlflow.start_run(run_name=f"final_model_{model_name}") as run:
        # Log parent run information
        mlflow.set_tag("parent_run_id", best_run_id)
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("mlflow.note.content",
                      f"Final model trained with best hyperparameters from run {best_run_id}")

        # Log dataset information
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("model_class", model_class_path)

        # Log the best hyperparameters
        mlflow.log_params(best_params)

        print(f"\nTraining final model on full training set...")
        # Train the model on the full training set
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'train_f1': f1_score(y_train, y_train_pred),
        }

        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba),
        }

        # Log all metrics
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        mlflow.log_dict({
            'confusion_matrix': cm.tolist(),
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }, "confusion_matrix.json")

        # Log classification report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        # Create input example and signature for model logging
        input_example = X_train.head(5)
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

        # Log the trained model with signature and input example
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example,
            registered_model_name=f"telco_churn_{model_name}"
        )

        # Save model locally
        model_dir = Path("trained_models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "final_model.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        mlflow.log_artifact(str(model_path))

        # Save training results
        results = OmegaConf.create({
            'model_name': model_name,
            'model_class': model_class_path,
            'hyperparameters': best_params,
            'parent_run_id': best_run_id,
            'final_run_id': run.info.run_id,
            'metrics': {
                'train': {k: float(v) for k, v in train_metrics.items()},
                'test': {k: float(v) for k, v in test_metrics.items()}
            },
            'confusion_matrix': {
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1])
            }
        })

        results_path = model_dir / "final_model_results.yaml"
        OmegaConf.save(results, str(results_path))
        mlflow.log_artifact(str(results_path))

        # Print results
        print("\n" + "="*50)
        print("FINAL MODEL TRAINING RESULTS")
        print("="*50)
        print(f"\nModel: {model_name}")
        print(f"Model class: {model_class_path}")
        print(f"\nHyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        print(f"\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0, 0]:4d}  FP: {cm[0, 1]:4d}")
        print(f"  FN: {cm[1, 0]:4d}  TP: {cm[1, 1]:4d}")

        print(f"\nModel saved to: {model_path}")
        print(f"Results saved to: {results_path}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("="*50)


if __name__ == "__main__":
    main()
