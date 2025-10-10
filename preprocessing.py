# Packages
import pandas as pd

def main():
    # Read data
    df = pd.read_csv("data/data.csv")

    # Data preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})
    df.to_csv("data/cleaned_data.csv", index=False)

if __name__ == "__main__":
    main()