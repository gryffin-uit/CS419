import numpy as np

def precision_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = list(retrieved_docs)[:k]
    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_k)
    true_positives = len(retrieved_set & relevant_set)
    return true_positives / k if k > 0 else 0.0

def recall_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = list(retrieved_docs)[:k]
    relevant_set = set(relevant_docs)
    true_positives = len(set(retrieved_k) & relevant_set)
    return true_positives / len(relevant_set) if relevant_set else 0.0

def average_precision(relevant_docs, retrieved_docs):
    """
    Tính Average Precision (AP) cho một truy vấn.
    """
    relevant_set = set(relevant_docs)
    if not relevant_set:
        return 0.0

    ap = 0.0
    hit_count = 0
    for i, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_set:
            hit_count += 1
            ap += hit_count / i
    return ap / len(relevant_set)

def interpolated_precision(recall_points, precisions):
    """
    Nội suy precision tại 11 điểm recall (0.0, 0.1, ..., 1.0)
    theo cách của TREC (11-point interpolated average precision)
    """
    recall_levels = np.linspace(0, 1, 11)
    interpolated = []
    for r in recall_levels:
        precs = [p for rec, p in zip(recall_points, precisions) if rec >= r]
        interpolated.append(max(precs) if precs else 0.0)
    return np.mean(interpolated)

def eval_model(model_name, retrieval_fn, queries, relevance, *args):
    """
    Đánh giá mô hình IR bằng Precision, Recall, F1 trung bình và MAP nội suy 11 điểm.

    Args:
        model_name (str): Tên mô hình
        retrieval_fn (function): Hàm truy xuất tài liệu
        queries (list): Danh sách các query string
        relevance (dict): dict chứa nhãn relevance {query_id: [doc_ids]}
        *args: các đối số bổ sung cho retrieval_fn

    Prints:
        Trung bình Precision, Recall, F1-score, MAP nội suy 11 điểm
    """     
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_map = 0
    num_queries = len(queries)

    for qid, query_text in enumerate(queries, start=1):
        retrieved_docs = retrieval_fn(query_text, *args)
        relevant_docs = relevance.get(qid, [])

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

        # Tính precision và recall tại từng điểm cắt k để nội suy
        precisions = []
        recalls = []
        for k in range(1, len(retrieved_docs) + 1):
            p_k = precision_at_k(relevant_docs, retrieved_docs, k)
            r_k = recall_at_k(relevant_docs, retrieved_docs, k)
            precisions.append(p_k)
            recalls.append(r_k)

        map_11_point = interpolated_precision(recalls, precisions)
        total_map += map_11_point

    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_f1 = total_f1 / num_queries
    avg_map = total_map / num_queries

    print(f"\n[{model_name} Model Evaluation]")
    print(f"Precision (avg): {avg_precision:.4f}")
    print(f"Recall    (avg): {avg_recall:.4f}")
    print(f"F1-score  (avg): {avg_f1:.4f}")
    print(f"MAP 11-point (avg): {avg_map:.4f}")

    
