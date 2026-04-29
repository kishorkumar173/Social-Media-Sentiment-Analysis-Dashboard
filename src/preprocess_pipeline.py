import pandas as pd
from preprocessing import clean_text

# load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# apply preprocessing
df['cleaned_text'] = df['text'].apply(clean_text)

# save final dataset
df.to_csv("data/processed/final_data.csv", index=False)

print("Preprocessing completed!")
print(df[['text', 'cleaned_text']].head())
print(df['cleaned_text'].head())