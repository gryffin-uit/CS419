# Information Retrieval Project

This project implements different Information Retrieval (IR) models including:
- Boolean Model
- Vector Space Model (TF-IDF)
- Latent Semantic Indexing (LSI) using Boolean filtering

## 🛠 Requirements

- Python 3.9
- Required packages in `requirements.txt`

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/gryffin-uit/CS419.git

2. Navigate to the project folder:

    ```bash
    cd CS419

3. Make sure you are using Python 3.9. You can check your version:

    ```bash
    python --version

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt

5. Setup nltk

    ```bash
    python setup_nltk.py

6. Run the main program to valid F1,Precision,ReCall and MAP score

    ```bash
    python main.py


# IR Query Runner (`run_query.py`)

This script runs different Information Retrieval (IR) models on a specific query and returns the top relevant documents.

## Supported Models

- **Boolean**
- **Vector Space**
- **LSA Boolean**

---

## Usage

    ```bash
        python run_query.py -qid <query_id> --model <model_name> [-n TOP_N] [--prepare_tfidf] [--prepare_inverted_index] [--show_result]


### Arguments

- `-qid`: Query ID (1-based index). **Required**
- `--model`: The IR model to use. Must be one of `"Boolean"`, `"VectorSpace"`, or `"LSA_Boolean"`. **Required**
- `-n`, `--top_n`: Number of top documents to return (default: 10)
- `--prepare_tfidf`: Prepare TF-IDF matrix (only used for VectorSpace model)
- `--prepare_inverted_index`: Prepare inverted index (used by Boolean, LSA_Boolean, and VectorSpace models)
- `--show_result`: Show detailed content of the returned documents

---

### Examples

1. Run Boolean model on query #5, return top 5 documents, show detailed results:

   ```bash
   python run_query.py -qid 5 --model Boolean -n 5 --prepare_inverted_index --show_result

2. Run Vector Space model on query #1, return top 10 documents, prepare TF-IDF and inverted index, show detailed results:

    ```bash
    python run_query.py -qid 1 --model VectorSpace --prepare_tfidf --prepare_inverted_index --show_result

3. Run LSA Boolean model on query #3, return top 8 documents, prepare inverted index:

    ```bash
    python run_query.py -qid 3 --model LSA_Boolean -n 8 --prepare_inverted_index


## 📁 Project Structure

- `main.py` – Main script to run evaluations.
- `models/` – Contains retrieval model implementations.
- `evaluation/` – Evaluation metrics and tools.
- `data/` – Preprocessing and input utilities.
- `in/` – Folder for input files (queries, relevance judgments, documents).

## 📌 Notes

- Ensure the input data (Cranfield dataset) is placed properly in the `in/` folder.
- The documents, queries, and relevance data files are loaded from paths specified in the config.py file.
- For Boolean and LSA Boolean models, it is recommended to use --prepare_inverted_index.
- The Vector Space model requires both TF-IDF and inverted index data, so use --prepare_tfidf and --prepare_inverted_index.
- Use --show_result to print the full content of the retrieved documents for better analysis.
## 👨‍🏫 Advisor

- TS. Nguyễn Trọng Chỉnh

## 👥 List of Group Members

| STT | MSSV     | Full Name                |
|-----|----------|--------------------------|
| 1   | 22520811 | Huỳnh Ngọc Bảo Long      |
| 2   | ........ | .........................|
| 3   | ........ | .........................|


