# run_query.py
import argparse
from models.boolean import *
from models.vector_space import *
from models.LSA_boolean import *
from models import boolean
from models import vector_space
from models import LSA_boolean
from data.reader import *
from data.preprocessor import *
from config import *

def run_model(query_id, model_name, top_n, queries, documents, inverted_index=None, tf_idf=None, word_info_lst=None,show_result=False):
    query_text = queries[query_id - 1] 
    
    if model_name == "Boolean":
        if inverted_index is None:
            raise ValueError("inverted_index phải được truyền cho Boolean model")
        retrieved_docs = boolean_retrieval(query_text, inverted_index)
        retrieved_docs = retrieved_docs[:top_n]
    elif model_name == "VectorSpace":
        if tf_idf is None or word_info_lst is None:
            raise ValueError("tf_idf và word_info_lst phải được truyền cho VectorSpace model")
        retrieved_docs = vectorSpaceModel(query_text, documents, tf_idf, word_info_lst, top_n)
        retrieved_docs = list(retrieved_docs.keys())[:top_n]
    elif model_name == "LSA_Boolean":
        if inverted_index is None:
            raise ValueError("inverted_index phải được truyền cho LSA_Boolean model")
        retrieved_docs = LSA_with_boolean(query_text, documents, inverted_index, top_n)
    else:
        raise ValueError(f"Model {model_name} chưa được hỗ trợ.")

    print(f"Query: \"{query_text}\"")
    print("Results:")
    print(retrieved_docs)

    if show_result:
        print("\nDetailed results:")
        for doc_id in retrieved_docs:
            print(f"{doc_id}: \"{documents[doc_id - 1]}\"")

def main():
    parser = argparse.ArgumentParser(description="Run IR model on a query")
    parser.add_argument('-qid', type=int, required=True, help="Query ID (1-based)")
    parser.add_argument('--model', type=str, required=True, choices=["Boolean", "VectorSpace", "LSA_Boolean"], help="Model name")
    parser.add_argument('-n', '--top_n', type=int, default=10, help="Number of top documents to return")
    parser.add_argument('--prepare_tfidf', action='store_true', help="Chuẩn bị tf-idf (chỉ dùng cho VectorSpace)")
    parser.add_argument('--prepare_inverted_index', action='store_true', help="Chuẩn bị inverted index (dùng cho Boolean, LSA_Boolean và VectorSpace)")
    parser.add_argument('--show_result', action='store_true', help="Hiển thị nội dung chi tiết các văn bản trả về")

    args = parser.parse_args()

    documents = read_documents(DOCUMENTS_PATH)
    queries = read_queries(QUERIES_PATH)
    relevance = read_relevance(RELEVANCE_PATH)

    word_lst = {}
    for idx, doc in enumerate(documents, 1):
        word_lst = create_word_lst(idx, doc, word_lst)
    word_info_lst = create_vocab_lst(word_lst)

    tf_idf = None
    inverted_index = None

    if args.prepare_inverted_index or args.model in ["Boolean", "LSA_Boolean", "VectorSpace"]:
        inverted_index = {}
        for idx, doc in enumerate(documents, 1):
            inverted_index = create_inverted_index(idx, doc, inverted_index)

    if args.prepare_tfidf or args.model == "VectorSpace":
        tf_idf, word_info_lst = prepare_tfidf(documents, word_info_lst)
    
    run_model(args.qid, args.model, args.top_n, queries, documents, inverted_index, tf_idf, word_info_lst, args.show_result)

if __name__ == "__main__":
    main()
