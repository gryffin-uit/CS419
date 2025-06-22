# data/reader.py
import os
import re
from natsort import natsorted

def read_documents(documents_path):
    print(f"Reading documents from: {documents_path}")
    documents = []
    for file in natsorted(os.listdir(documents_path)):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    return documents

def read_queries(queries_path):
    with open(queries_path) as file:
        queries = file.readlines()
    queries = [line.strip().split("\t")[1] for line in queries]
    return queries

def read_relevance(relevance_path):
    relevance = {}
    for idx, file in enumerate(natsorted(os.listdir(relevance_path)), 1):
        if file.endswith(".txt"):
            file_path = os.path.join(relevance_path, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                relevance[idx] = [int(re.split(r"\s+", li.strip())[1]) for li in lines if li.strip()]
    return relevance
