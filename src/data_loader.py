import pandas as pd

def load_data():
    df = pd.read_csv("data/raw/airline_sentiment.csv")
    
    # Keep only needed columns
    df = df[['text', 'airline_sentiment']]
    
    # Rename for consistency
    df.columns = ['text', 'sentiment']
    
    # Drop missing values
    df.dropna(inplace=True)
    
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("\nDataset Shape:", df.shape)
    
df.to_csv("data/processed/cleaned_data.csv", index=False)
print(df['sentiment'].value_counts())