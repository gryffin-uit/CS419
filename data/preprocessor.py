import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
stopword_lst = set(stopwords.words('english'))

def create_words(sentence):
    words = re.split(r"[^\w]+", sentence)
    return list(filter(None, words))  # loại bỏ chuỗi rỗng

def create_terms(sentence):
    # Tách câu thành từ, loại bỏ stopword và áp dụng stemming
    word_lst = re.split(r"[^\w]+", sentence)
    terms = []
    for word in filter(None, word_lst):
        word_lower = word.lower()
        if word_lower in stopword_lst:
            continue
        stem_word = stemmer.stem(word_lower)
        if len(stem_word) < 3:
            continue
        terms.append(stem_word)
    return terms


def update_term_freq(doc_id, term, tf_dict):
    if term in tf_dict:
        tf_dict[term][doc_id] = tf_dict[term].get(doc_id, 0) + 1
    else:
        tf_dict[term] = {doc_id: 1}
    return tf_dict

def create_term_freq(doc_id, sentence, tf_dict):
    for term in create_terms(sentence):
        tf_dict = update_term_freq(doc_id, term, tf_dict)
    return tf_dict

def update_inverted_index(doc_id, term, inverted_index):
    if term in inverted_index:
        inverted_index[term].add(doc_id)
    else:
        inverted_index[term] = {doc_id}
    return inverted_index

def create_inverted_index(doc_id, sentence, inverted_index):
    for term in create_terms(sentence):
        inverted_index = update_inverted_index(doc_id, term, inverted_index)
    return inverted_index


def create_vocab_list(term_freq_dict):
    vocab_list = {}
    for term in term_freq_dict:
        doc_count = len(term_freq_dict[term])
        total_freq = sum(term_freq_dict[term].values())
        vocab_list[term] = [doc_count, total_freq]
    return vocab_list

