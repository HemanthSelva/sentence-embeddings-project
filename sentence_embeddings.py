import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load Model
print("Loading model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define Sentences
sentences = [
    "Artificial intelligence is rapidly transforming the technology landscape.",
    "Machine learning models can analyze vast amounts of data quickly.",
    "A walk through the dense forest reveals the soothing beauty of nature.",
    "The snow-capped mountains provided a breathtaking view at sunrise.",
    "Basketball requires agility, teamwork, and quick decision-making.",
    "The local team won the championship after a thrilling overtime match.",
    "The chef prepared a spicy and flavorful five-course meal.",
    "Eating a balanced diet with fresh vegetables is crucial for health."
]

# Generate Embeddings
print("Generating embeddings...")
embeddings = model.encode(sentences)

print("\n--- Embeddings Details ---")
for i, sentence in enumerate(sentences):
    print(f"Sentence: '{sentence}'")
    print(f"Shape: {embeddings[i].shape}")
    print(f"First 5 values: {embeddings[i][:5]}")
    print("-" * 60)

# Similarity Matrix
print("\n--- Cosine Similarity Matrix ---")
similarity_matrix = cosine_similarity(embeddings)

# Formatted printing of the similarity matrix
header = "    " + "".join([f"{i:>7}" for i in range(len(sentences))])
print(header)
for i in range(len(sentences)):
    row_str = f"{i:>2}  " + "".join([f"{val:>7.2f}" for val in similarity_matrix[i]])
    print(row_str)

max_sim = -2.0
min_sim = 2.0
most_sim_pair = (0, 1)
least_sim_pair = (0, 1)

n = len(sentences)
for i in range(n):
    for j in range(i + 1, n):
        sim = similarity_matrix[i][j]
        if sim > max_sim:
            max_sim = sim
            most_sim_pair = (i, j)
        if sim < min_sim:
            min_sim = sim
            least_sim_pair = (i, j)

print("\n--- Similarity Extremes ---")
print("Most similar pair:")
print(f" 1: '{sentences[most_sim_pair[0]]}'")
print(f" 2: '{sentences[most_sim_pair[1]]}'")
print(f" Similarity: {max_sim:.4f}")

print("\nLeast similar pair:")
print(f" 1: '{sentences[least_sim_pair[0]]}'")
print(f" 2: '{sentences[least_sim_pair[1]]}'")
print(f" Similarity: {min_sim:.4f}")

# Save Output
print("\nSaving output to sentence_embeddings.json...")
output_data = {
    "sentences": sentences,
    "embeddings": embeddings.tolist()
}

with open('sentence_embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4)

print("Process completed successfully.")
