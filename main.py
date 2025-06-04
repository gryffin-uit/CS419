from config import *
from data.reader import read_documents, read_queries, read_relevance
from data.preprocessor import create_term_freq,create_inverted_index,create_vocab_list
from models.boolean import boolean_retrieval
from models.vector_space import prepare_tfidf, vectorSpaceModel
from models.LSA_boolean import LSA_with_boolean
from evaluation.metrics import eval_model
from evaluation.plot import plot_precision_recall
import os

# Read data
documents = read_documents(DOCUMENTS_PATH)
queries = read_queries(QUERIES_PATH)
relevance = read_relevance(RELEVANCE_PATH)

# Word-level stats
# Build term frequency dictionary
word_lst = {}
for idx, doc in enumerate(documents, 1):
    word_lst = create_term_freq(idx, doc, word_lst)

#print(word_lst)  # ✅ term frequency dictionary

# Tạo word_info_lst từ word_lst để phục vụ TF-IDF
word_info_lst = create_vocab_list(word_lst)  # ✅ doc count + total freq

#print(word_info_lst)  # ✅ word_info_lst chứa thông tin về từ vựng

# Build inverted index (posting list)
inverted_index = {}
for idx, doc in enumerate(documents, 1):
    inverted_index = create_inverted_index(idx, doc, inverted_index)


# Boolean Evaluation
eval_model("Boolean", boolean_retrieval, queries, relevance, inverted_index)

# # # Vector Space Evaluation
tf_idf, word_info_lst = prepare_tfidf(documents, word_info_lst)
eval_model("Vector Space", lambda q,: vectorSpaceModel(q, documents, tf_idf, word_info_lst, TOP_N), queries, relevance)
# # # LSI Evaluation
eval_model("LSA boolean", lambda q,: LSA_with_boolean(q, documents,inverted_index, TOP_N,n_components=100), queries, relevance)

# Optional: Plotting
# plot_precision_recall(...)
