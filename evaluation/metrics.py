def eval_model(model_name, retrieval_fn, queries, relevance, *args):
    """
    Đánh giá mô hình IR bằng Precision, Recall và F1-score trung bình.

    Args:
        model_name (str): Tên mô hình (ví dụ: "Boolean")
        retrieval_fn (function): Hàm truy xuất tài liệu
        queries (list): Danh sách các query string
        relevance (dict): dict chứa nhãn relevance {query_id: [doc_ids]}
        *args: các đối số bổ sung cho retrieval_fn (ví dụ inverted_index)

    Prints:
        Trung bình Precision, Recall, F1-score
    """
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_queries = len(queries)

    for qid, query_text in enumerate(queries, start=1):
        retrieved_docs = retrieval_fn(query_text, *args)
        relevant_docs = relevance.get(qid, [])

        # Tính precision, recall, f1 cho từng truy vấn
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        true_positives = len(retrieved_set & relevant_set)
        precision_val = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall_val = true_positives / len(relevant_set) if relevant_set else 0.0
        f1_val = (2 * precision_val * recall_val / (precision_val + recall_val)
                  if (precision_val + recall_val) else 0.0)

        total_precision += precision_val
        total_recall += recall_val
        total_f1 += f1_val

    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_f1 = total_f1 / num_queries

    print(f"\n[{model_name} Model Evaluation]")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1-score:  {avg_f1:.4f}")
