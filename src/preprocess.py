import pandas as pd


INPUT_FILE = "data/dataset.csv"
OUTPUT_FILE = "data/processed.csv"


def preprocess_data():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original dataset shape: {df.shape}")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows containing missing values
    df = df.dropna()

    print(f"Processed dataset shape: {df.shape}")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_data()