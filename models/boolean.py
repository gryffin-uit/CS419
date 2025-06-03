from data.preprocessor import create_terms

def boolean_retrieval(query, inverted_index):
    retrieved_indexes = set()
    query_terms = create_terms(query)
    for term in query_terms:
        if term in inverted_index:
            retrieved_indexes.update(inverted_index[term])
    return list(retrieved_indexes)
