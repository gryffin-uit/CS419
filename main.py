from config import *
from data.reader import read_documents, read_queries, read_relevance
from data.preprocessor import create_term_freq,create_inverted_index,create_vocab_list
from models.boolean import boolean_retrieval
from models.vector_space import vector_space_model
from models.LSA_boolean import LSA_with_boolean
from evaluation.metrics import eval_model
from evaluation.plot import plot_precision_recall
import os

# Read data
documents = read_documents(DOCUMENTS_PATH)
queries = read_queries(QUERIES_PATH)
relevance = read_relevance(RELEVANCE_PATH)


# Build inverted index (posting list)
inverted_index = {}
for idx, doc in enumerate(documents, 1):
    inverted_index = create_inverted_index(idx, doc, inverted_index)

#print(inverted_index)  # ✅ inverted index

#Boolean Evaluation
eval_model("Boolean", boolean_retrieval, queries, relevance, inverted_index)

#Vector Space Evaluation
eval_model(
    "Vector Space",
    lambda q: [doc_id for doc_id, _ in vector_space_model(documents, [q], TOP_N)[0]],
    queries,
    relevance
)
#LSI Evaluation
eval_model("LSA boolean", lambda q,: LSA_with_boolean(q, documents,inverted_index, TOP_N,n_components=100), queries, relevance)

# Optional: Plotting
# plot_precision_recall(...)
