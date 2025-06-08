import numpy as np
from collections import OrderedDict
from data.preprocessor import create_words
import math

def termFrequencyInDoc(word_info_lst, documents):
    tf_docs = {}
    for term in word_info_lst.keys():
        tf_docs[term] = {}
        for docs_id in range(len(documents)):
            doc = documents[docs_id]
            tf_docs[term][docs_id] = np.log2(1 + doc.count(term))
    return tf_docs

def inverseFrequencyInDoc(word_info_lst, documents):
    idf_docs = {}
    length = len(documents)
    for term in word_info_lst.keys():
        doc_count = word_info_lst[term][0]
        idf_docs[term] = np.log2(length / (1 + doc_count))
    return idf_docs

def tf_idf_score(tf_weight, idf_weight, word_info_lst, documents):
    tf_idf = {}
    for term in word_info_lst.keys():
        tf_idf[term] = {}
        for docs_id in range(len(documents)):
            tf_idf[term][docs_id] = tf_weight[term][docs_id] * idf_weight[term]
    return tf_idf

def prepare_tfidf(documents, word_info_lst):
    tf_score = termFrequencyInDoc(word_info_lst, documents)
    idf_score = inverseFrequencyInDoc(word_info_lst, documents)
    tf_idf = tf_idf_score(tf_score, idf_score, word_info_lst, documents)
    return tf_idf, word_info_lst

def vectorSpaceModel(query_input, documents, tf_idf, word_info_lst, number_of_selected):
    query = list(create_words(query_input))
    total_terms = len(query)
    
    query_vocab = list({word for word in query if word in word_info_lst})

    query_tf_idf = {}
    N = len(documents)  # Tổng số document
    for word in query_vocab:
        tf = query.count(word) / total_terms
        doc_count = word_info_lst[word][0]  
        idf = math.log2(N / doc_count) if doc_count != 0 else 0
        query_tf_idf[word] = tf * idf

    relevance_scores = {}
    for docs_id in range(len(documents)):
        score = sum(query_tf_idf[word] * tf_idf[word].get(docs_id, 0) for word in query_vocab)
        relevance_scores[docs_id + 1] = score

    sorted_value = OrderedDict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
    top_values = {k: sorted_value[k] for k in list(sorted_value)[:number_of_selected]}
    return top_values