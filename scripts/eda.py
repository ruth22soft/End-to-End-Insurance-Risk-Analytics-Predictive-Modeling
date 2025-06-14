import pandas as pd

# Load data from a pipe-separated file
df = pd.read_csv('../data/MachineLearningRating_v3.csv', delimiter='|')

# 1. Number of records
num_records = len(df)
print(f"Number of records: {num_records}")

# 2. Total characters across all text columns
# Select only object (string) columns
text_columns = df.select_dtypes(include='object').columns

total_chars = df[text_columns].applymap(lambda x: len(str(x))).sum().sum()
print(f"Total characters in text columns: {total_chars}")

# 3. Total words across all text columns
total_words = df[text_columns].applymap(lambda x: len(str(x).split())).sum().sum()
print(f"Total words in text columns: {total_words}")

# 4. Vocabulary size (unique words)
all_text = ' '.join(df[text_columns].astype(str).values.flatten()).lower()
vocab = set(all_text.split())
print(f"Vocabulary size: {len(vocab)}")

# 5. File size on disk (if you have the file path)
import os
file_size_bytes = os.path.getsize('../data/MachineLearningRating_v3.csv')
print(f"File size on disk: {file_size_bytes / (1024*1024):.2f} MB")
