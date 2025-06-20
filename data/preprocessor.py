import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import numpy as np
import math

stemmer = SnowballStemmer("english")
stopword_lst = set(stopwords.words('english'))

def preprocess_text(sentence):
    word_lst = re.split(r"[^\w]+", sentence)
    
    terms = []
    for word in filter(None, word_lst):
        word_lower = word.lower()
        if word_lower in stopword_lst:
            continue
        if word_lower.isdigit():  
            continue
        stem_word = stemmer.stem(word_lower)
        if len(stem_word) < 3:
            continue
        terms.append(stem_word)
    return terms


def preprocess_documents(documents):
    processed_docs = []
    vocab_dict = {}  

    for doc in documents:
        terms = preprocess_text(doc)
        processed_docs.append(terms)
        for term in terms:
            if term not in vocab_dict:
                vocab_dict[term] = None  

    vocab_lst = list(sorted(vocab_dict.keys()))
    return processed_docs, vocab_lst



def create_inverted_index(processed_docs):
    inverted_index = defaultdict(lambda: defaultdict(int))  

    for doc_id, terms in enumerate(processed_docs, start=1): 
        for term in terms:
            inverted_index[term][doc_id] += 1

    return inverted_index


def vectorize_documents(processed_docs, vocab_lst, inverted_index):
    D = len(vocab_lst)
    N = len(processed_docs)
    vocab_index = {term: i for i, term in enumerate(vocab_lst)}

    doc_vectors = np.zeros((N, D), dtype=int)

    for term, postings in inverted_index.items():
        term_idx = vocab_index[term]
        for doc_id, tf in postings.items():
            doc_vectors[doc_id][term_idx] = tf

    return doc_vectors


def compute_idf_vector(vocab_lst, inverted_index, total_docs):
    idf_dict = {}
    for term in vocab_lst:
        df = len(inverted_index.get(term, {}))
        if df == 0:
            idf = 0
        else:
            idf = math.log10(total_docs / df)
        idf_dict[term] = idf
    return idf_dict

def compute_tfidf_matrix(tf_matrix, idf_vector):
    
    tfidf_matrix = tf_matrix * idf_vector 
    return tfidf_matrix