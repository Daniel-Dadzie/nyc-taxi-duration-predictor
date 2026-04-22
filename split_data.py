import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("train.csv")

# Split into train (80%) and test (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save new files
train_df.to_csv("train_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

print("Done!")
print("Train size:", len(train_df))
print("Test size:", len(test_df))