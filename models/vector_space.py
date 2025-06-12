from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def vector_space_model(documents, queries, top_k=5):
    vectorizer = TfidfVectorizer()

    # Tính TF-IDF cho toàn bộ documents
    tfidf_matrix = vectorizer.fit_transform(documents)  # shape: (num_docs, vocab_size)

    results = []
    for query in queries:
        # Vector hoá truy vấn
        query_vec = vectorizer.transform([query])  # shape: (1, vocab_size)

        # Tính cosine similarity
        scores = cosine_similarity(query_vec, tfidf_matrix)[0]  # 1D array

        # Sắp xếp giảm dần và lấy top K
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        top_k_scores = [(idx + 1, scores[idx]) for idx in top_k_indices]

        results.append(top_k_scores)
    return results
