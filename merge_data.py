import pandas as pd

true_df = pd.read_csv("data/true.csv")
fake_df = pd.read_csv("data/fake.csv")

true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

true_df = true_df[['text', 'label']]
fake_df = fake_df[['text', 'label']]

merged_df = pd.concat([true_df, fake_df], ignore_index=True)

merged_df = merged_df.sample(frac=1, random_state=42)

merged_df.to_csv("data/merged_news.csv", index=False)
print("Merged dataset saved to data/merged_news.csv")
