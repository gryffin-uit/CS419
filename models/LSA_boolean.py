# models/lsa_boolean.py
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from models.boolean import boolean_retrieval
from sklearn.feature_extraction.text import TfidfVectorizer



def LSA_with_boolean(query, documents, inverted_index, number_of_selected,n_components=100):
    # Bước 1: Lọc tài liệu liên quan bằng mô hình Boolean
    filtered_ids = boolean_retrieval(query, inverted_index)
    filtered_docs = [documents[i - 1] for i in filtered_ids]  # -1 vì index bắt đầu từ 0

    if not filtered_docs:
        return []

    # Bước 2: Vector hóa tập tài liệu đã lọc
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(filtered_docs) # shape (n_docs, n_terms)


    # Bước 3: Phân tích SVD
    svd = TruncatedSVD(n_components=min(n_components, X.shape[1]-1))
    X_reduced = svd.fit_transform(X)  # shape (n_docs, n_components)
    X_reduced = normalize(X_reduced)  # chuẩn hóa vector tài liệu

    # Bước 4: Vector hóa truy vấn
    q_vec = vectorizer.transform([query])
    q_reduced = svd.transform(q_vec)
    q_reduced = normalize(q_reduced)


    similarities = np.dot(X_reduced, q_reduced.T).flatten()
    top_indices = np.argsort(similarities)[-number_of_selected:][::-1]


    # Bước 5: Lấy các tài liệu có độ tương đồng cao nhất
    top_docs = [filtered_ids[i] for i in top_indices]
    return top_docs 