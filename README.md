# Sentence Embeddings Generator

This project generates sentence embeddings using the `sentence-transformers` library and computes cosine similarity between different sentences. It utilizes the `all-MiniLM-L6-v2` model from Hugging Face for efficient and powerful text representation.

## Features
- Generates high-quality sentence embeddings for diverse topics.
- Visualizes embedding shape and partial value arrays.
- Computes a full cosine similarity matrix across all sentences.
- Identifies and highlights the most similar and least similar sentence pairs.
- Automatically saves outputs consisting of the sentences and their exact embedded representations to `sentence_embeddings.json`.

## Requirements
Ensure you have Python 3 installed. Use `pip` to install the dependencies:
```bash
pip install sentence-transformers scikit-learn numpy
```

## Running the Code
1. Navigate to the project directory.
2. Run the Python script:
   ```bash
   python sentence_embeddings.py
   ```
3. Check the console for the output and the directory for the generated `sentence_embeddings.json` file.
