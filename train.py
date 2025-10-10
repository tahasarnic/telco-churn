# Packages
import pandas as pd

# Feature engineering
from feature_engine.pipeline import make_pipeline
from feature_engine.selection import DropFeatures
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import RobustScaler

# Scikit-learn modules
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Experiments with base models
def main():
    # Read cleaned data
    df = pd.read_csv("data/cleaned_data.csv")

    # Train-test split
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing pipeline
    preprocessing_pipeline = make_pipeline(
        DropFeatures(features_to_drop=['customerID']),
        MeanMedianImputer(imputation_method='median', variables=['TotalCharges']),
        OrdinalEncoder(encoding_method='ordered'),
        SklearnTransformerWrapper(transformer=RobustScaler(), variables=['tenure', 'MonthlyCharges', 'TotalCharges'])
    )
    X_train_scaled = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_test_scaled = preprocessing_pipeline.transform(X_test)
    print("Preprocessing completed.")
    print(f"Processed training data shape: {X_train_scaled.head(3)}")
    print(f"Processed test data shape: {X_test_scaled.head(3)}")

    # Model training and evaluation
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    experiment_results = {
        "Model": [],
        "Mean CV Score": []
    }

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = cross_val_score(model, X_test_scaled, y_test, cv=5)
        experiment_results["Model"].append(model_name)
        experiment_results["Mean CV Score"].append(score.mean())
        print(f"{model_name} - Mean CV Score: {score.mean():.4f}")
        print(f"{model_name} - CV Scores: {score}")
        print("-" * 30)
    
    # Feature selection methods will be added here in the future and will be used with Hydra for hyperparameter tuning.

    print("Experiment results:", experiment_results)
if __name__ == "__main__":
    main()