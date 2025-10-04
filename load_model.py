# Load and use a trained model from MLflow
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
import sys


def load_and_predict(run_id: str):
    """
    Load a model from MLflow and make predictions.

    Args:
        run_id: MLflow run ID to load the model from
    """
    print(f"Loading model from run: {run_id}")

    # Load the preprocessing pipeline
    preprocessing_pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/preprocessing_pipeline")
    print("✓ Preprocessing pipeline loaded")

    # Load the trained model
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    print("✓ Model loaded")

    # Get run information
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    print(f"\nRun Information:")
    print(f"  Experiment: {run.info.experiment_id}")
    print(f"  Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"  Model: {run.data.params.get('model_name', 'N/A')}")
    print(f"  Start Time: {run.info.start_time}")

    # Load test data
    df = pd.read_csv("data/cleaned_data.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Use the same split as training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform data using the loaded preprocessing pipeline
    X_test_transformed = preprocessing_pipeline.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_transformed)

    # Evaluate
    accuracy = balanced_accuracy_score(y_test, y_pred)

    print(f"\nTest Results:")
    print(f"  Balanced Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, preprocessing_pipeline


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_model.py <RUN_ID>")
        print("\nExample: python load_model.py fdd665881da24a59ab8c593a40f1bf2c")
        print("\nTo find run IDs, use: mlflow ui")
        sys.exit(1)

    run_id = sys.argv[1]
    model, pipeline = load_and_predict(run_id)

    print(f"\n✓ Model and pipeline loaded successfully!")
    print(f"  You can now use 'model' and 'pipeline' for predictions")
