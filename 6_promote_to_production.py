# Packages
import hydra
from omegaconf import DictConfig
from datetime import datetime

# MLflow
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Promote the best final model to production using MLflow Model Registry aliases"""

    # Set up MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()

    print("="*60)
    print("MODEL PROMOTION TO PRODUCTION")
    print("="*60)

    # Get the final_model experiment
    final_experiment = client.get_experiment_by_name("final_model")

    if final_experiment is None:
        raise ValueError("Experiment 'final_model' not found. Run training (4_train.py) first.")

    # Search for the best final model run
    # Order by test_roc_auc score (you can change this to other metrics)
    runs = client.search_runs(
        experiment_ids=[final_experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'final_model_%'",
        order_by=["metrics.test_roc_auc DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No final model runs found. Run training (4_train.py) first.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_name_tag = best_run.data.tags.get('model_name', 'unknown')

    # Extract metrics safely (convert from Metric objects to floats)
    test_accuracy = best_run.data.metrics.get('test_accuracy', 0.0)
    test_precision = best_run.data.metrics.get('test_precision', 0.0)
    test_recall = best_run.data.metrics.get('test_recall', 0.0)
    test_f1 = best_run.data.metrics.get('test_f1', 0.0)
    test_roc_auc = best_run.data.metrics.get('test_roc_auc', 0.0)

    # Display run information
    print(f"\nBest Final Model Run Found:")
    print(f"  Run ID: {run_id}")
    print(f"  Run Name: {best_run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"  Model Name: {model_name_tag}")
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  ROC-AUC:   {test_roc_auc:.4f}")

    # Get the registered model name
    registered_model_name = f"telco_churn_{model_name_tag}"

    # Check if model is already registered
    try:
        registered_model = client.get_registered_model(registered_model_name)
        print(f"\nRegistered model '{registered_model_name}' found.")
        print(f"  Description: {registered_model.description or 'No description'}")

        # Get all versions
        model_versions = client.search_model_versions(f"name='{registered_model_name}'")
        print(f"  Total versions: {len(model_versions)}")

        # Find the version corresponding to this run_id
        target_version = None
        for mv in model_versions:
            if mv.run_id == run_id:
                target_version = mv
                break

        if target_version is None:
            raise ValueError(f"Model version for run_id {run_id} not found in registry. "
                           f"This shouldn't happen - check your training script.")

        version_number = target_version.version

        print(f"\nModel Version Details:")
        print(f"  Version: {version_number}")
        print(f"  Status: {target_version.status}")

        # Check current aliases
        current_aliases = target_version.aliases if hasattr(target_version, 'aliases') else []
        if current_aliases:
            print(f"  Current Aliases: {', '.join(current_aliases)}")

    except MlflowException as e:
        raise ValueError(f"Registered model '{registered_model_name}' not found. "
                        f"This shouldn't happen - check your training script.") from e

    # Check if already has 'production' alias
    if 'production' in current_aliases:
        print(f"\n✓ Model version {version_number} already has 'production' alias.")
        print("  No action needed.")
        return

    # Check for existing production alias on other versions
    production_versions = []
    for mv in model_versions:
        mv_aliases = mv.aliases if hasattr(mv, 'aliases') else []
        if 'production' in mv_aliases:
            production_versions.append(mv)

    if production_versions:
        print(f"\n⚠ Found {len(production_versions)} model(s) with 'production' alias:")
        for pv in production_versions:
            print(f"  - Version {pv.version} (Run ID: {pv.run_id})")
        print(f"\nThe 'production' alias will be moved to the new version.")

    # Prompt for confirmation summary
    print(f"\n{'='*60}")
    print("PROMOTION SUMMARY")
    print("="*60)
    print(f"Registered Model: {registered_model_name}")
    print(f"Version to Promote: {version_number}")
    print(f"Alias: production")
    print(f"Run ID: {run_id}")
    print("="*60)

    # Uncomment the following lines if you want manual confirmation
    # confirmation = input("\nProceed with promotion? (yes/no): ")
    # if confirmation.lower() != 'yes':
    #     print("Promotion cancelled.")
    #     return

    # Set the 'production' alias (this automatically moves it from other versions)
    print(f"\n→ Setting 'production' alias on model version {version_number}...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add tags first (before any operations that might trigger YAML serialization issues)
    client.set_model_version_tag(
        name=registered_model_name,
        version=version_number,
        key="promoted_at",
        value=timestamp
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=version_number,
        key="alias",
        value="production"
    )

    # Add metrics as tags for easy reference
    client.set_model_version_tag(
        name=registered_model_name,
        version=version_number,
        key="test_roc_auc",
        value=str(test_roc_auc)
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=version_number,
        key="test_f1",
        value=str(test_f1)
    )

    # Now set the alias
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="production",
        version=version_number
    )

    print(f"✓ Model version {version_number} successfully promoted to production!")

    # Display final summary
    print(f"\n{'='*60}")
    print("PRODUCTION MODEL DETAILS")
    print("="*60)
    print(f"Model URI (alias): models:/{registered_model_name}@production")
    print(f"Model URI (versioned): models:/{registered_model_name}/{version_number}")
    print(f"Run ID: {run_id}")
    print(f"Promoted at: {timestamp}")
    print("="*60)

    print("\nYou can now use this model for inference:")
    print(f"  model = mlflow.pyfunc.load_model('models:/{registered_model_name}@production')")
    print("\nOr view in MLflow UI at: http://localhost:5000")


if __name__ == "__main__":
    main()
