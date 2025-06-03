import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
stopword_lst = set(stopwords.words('english'))

def create_words(sentence):
    regex = r"[ .,()0123456789=:+-/']\s*"
    words = re.split(regex, sentence)
    return list(filter(None, words))

def create_terms(sentence):
    regex = r"[ .,()0123456789=:+-/']\s*"
    word_lst = re.split(regex, sentence)
    terms = []
    for word in filter(None, word_lst):
        if word in stopword_lst:
            continue
        stem_word = stemmer.stem(word)
        if len(stem_word) < 3:
            continue
        terms.append(stem_word)
    return terms

def update_word_lst(id, word, word_lst):
    if word in word_lst:
        word_lst[word][id] = word_lst[word].get(id, 0) + 1
    else:
        word_lst[word] = {id: 1}
    return word_lst

def create_word_lst(id, sentence, word_lst):
    for word in create_words(sentence):
        word_lst = update_word_lst(id, word, word_lst)
    return word_lst

def update_inverted_index(id, term, inverted_index):
    if term in inverted_index:
        inverted_index[term][id] = inverted_index[term].get(id, 0) + 1
    else:
        inverted_index[term] = {id: 1}
    return inverted_index

def create_inverted_index(id, sentence, inverted_index):
    for term in create_terms(sentence):
        inverted_index = update_inverted_index(id, term, inverted_index)
    return inverted_index

def create_vocab_lst(inverted_index):
    vocab_lst = {}
    for term in inverted_index:
        doc_count = len(inverted_index[term])
        total_freq = sum(inverted_index[term].values())
        vocab_lst[term] = [doc_count, total_freq]
    return vocab_lst
