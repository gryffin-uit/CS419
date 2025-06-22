# Information Retrieval Project


Welcome to the **Information Retrieval (IR) System** developed by **Team ManChester City** — where precision meets power, and search meets style.

We don’t just retrieve documents —  
We **dominate** queries like we dominate the Premier League.

Built with Python, designed for clarity, and evaluated with passion, this project is our playbook for exploring Boolean, Vector Space, and LSA Boolean models.

> *"You may search the whole corpus, but only we retrieve like champions."*

---


## 🛠 Requirements

- Python 3.9
- Required packages in `requirements.txt`

## 🚀 How to Run
Open your VS Code and follow these step below.

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

6. Run the main notebook to valid Precision,ReCall and MAP score

    ```bash
    jupyter nbconvert --to notebook --execute --inplace --allow-errors VSM_and_LSI.ipynb

7. Open VSM_and_LSI.ipynb to see the output.
    ```bash
    code VSM_and_LSI.ipynb


## 📁 Project Structure

- `VSM_and_LSI.ipynb` – Main script to run evaluations.
- `models/` – Contains boolean model implementations.
- `evaluation/` – Evaluation metrics and plot.
- `data/` – Preprocessing and input utilities.
- `in/` – Folder for input files (queries, relevance judgments, documents).

## 📌 Notes

- Ensure the input data (Cranfield dataset) is placed properly in the `in/` folder.
- The documents, queries, and relevance data files are loaded from paths specified in the config.py file.
## 👨‍🏫 Advisor

- TS. Nguyễn Trọng Chỉnh

## 👥 List of Group Members

| STT | MSSV     | Full Name                |
|-----|----------|--------------------------|
| 1   | 22520811 | Huỳnh Ngọc Bảo Long      |
| 2   | 22521465 | Huỳnh Dương Tiến         |
| 3   | 22521007 | Trần Thành Nhân          |

## 👥 Contributor

[![Contributors](https://contrib.rocks/image?repo=gryffin-uit/CS419)](https://github.com/gryffin-uit/CS419/graphs/contributors)




