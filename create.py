import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a function to calculate embeddings
def calculate_embeddings(texts):
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

# Define the CSV file path and NPY file path
csv_file_path = './organizations.csv'
npy_file_path = 'embeddings.npy'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Handle missing or non-string values in the 'short_description' column
df['short_description'] = df['short_description'].fillna('')  # Replace NaN with empty string
df['short_description'] = df['short_description'].astype(str)  # Ensure all values are strings

# Split the data into chunks to save RAM
batch_size = 1000
num_chunks = len(df) // batch_size + 1

embeddings_list = []

# Iterate over chunks and calculate embeddings
for i in tqdm(range(num_chunks), desc="Calculating Embeddings"):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_texts = df['short_description'].iloc[start_idx:end_idx].tolist()
    batch_embeddings = calculate_embeddings(batch_texts)
    embeddings_list.extend(batch_embeddings)

# Convert embeddings list to a numpy array
embeddings_array = np.array(embeddings_list)

# Save the embeddings to an NPY file
np.save(npy_file_path, embeddings_array)

print(f"Embeddings saved to {npy_file_path}")
