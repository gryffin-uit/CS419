# evaluation/plot.py
import matplotlib.pyplot as plt

def evaluate_precision_recall(retrieval_results, reference):
    precisions = []
    recalls = []

    for qid, retrieved_docs in retrieval_results.items():
        relevant_docs = set(reference.get(qid, []))  
        retrieved_set = set(retrieved_docs)

        true_positives = len(retrieved_set & relevant_docs)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    print(f"\nAverage Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    return avg_precision, avg_recall

def plot_precision_recall(precision_vals, recall_vals, title="Precision-Recall"):
    """
    precision_vals, recall_vals là các list giá trị precision và recall tương ứng theo threshold hoặc k
    
    Ví dụ: precision_vals = [1.0, 0.8, 0.75, ...]
           recall_vals = [0.1, 0.2, 0.3, ...]
    """

    plt.figure(figsize=(8,6))
    plt.plot(recall_vals, precision_vals, marker='o', color='b', label='Precision-Recall')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def compute_interpolated_precisions(retrieved_docs_dict, relevant_docs_dict):
    """
    Trả về dict các giá trị precision trung bình tại mỗi mức recall từ 0.0 đến 1.0 (11 điểm).
    
    Returns:
        - avg_precisions: dict {recall_level: avg_precision}
    """
    recall_levels = [i / 10 for i in range(11)]
    all_precisions_at_recall = {r: [] for r in recall_levels}

    for qid, retrieved_docs in retrieved_docs_dict.items():
        relevant_docs = relevant_docs_dict.get(qid, set())
        if not relevant_docs:
            continue

        precision_recall_points = []
        num_relevant = 0

        for i, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision = num_relevant / i
                recall = num_relevant / len(relevant_docs)
                precision_recall_points.append((recall, precision))

        for recall_level in recall_levels:
            precisions = [p for r, p in precision_recall_points if r >= recall_level]
            max_p = max(precisions) if precisions else 0.0
            all_precisions_at_recall[recall_level].append(max_p)

    avg_precisions = {
        r: sum(all_precisions_at_recall[r]) / len(all_precisions_at_recall[r])
        if all_precisions_at_recall[r] else 0.0
        for r in recall_levels
    }
    return avg_precisions



def plot_interpolated_map(avg_precisions, title="📈 11-point Interpolated MAP"):
    """
    Vẽ biểu đồ MAP nội suy 11 điểm từ dict {recall_level: precision}
    """
    recall_levels = sorted(avg_precisions.keys())
    precision_vals = [avg_precisions[r] for r in recall_levels]

    plt.figure(figsize=(7, 5))
    plt.plot(recall_levels, precision_vals, marker='o', color='green', label='Mean Interpolated Precision')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 1.05)
    plt.xticks(recall_levels)
    plt.grid(True)
    plt.legend()
    plt.show()

    map_11pt = sum(precision_vals) / 11
    print(f" MAP nội suy 11 điểm: {map_11pt:.4f}")


