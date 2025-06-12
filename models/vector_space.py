from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def vector_space_model(documents, queries, top_k=5):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)  # shape: (num_docs, vocab_size)

    results = []
    for query in queries:
        query_vec = vectorizer.transform([query])  # shape: (1, vocab_size)

        scores = cosine_similarity(query_vec, tfidf_matrix)[0]  # 1D array

        top_k_indices = np.argsort(scores)[::-1][:top_k]
        top_k_scores = [(idx + 1, scores[idx]) for idx in top_k_indices]

        results.append(top_k_scores)
    return results
