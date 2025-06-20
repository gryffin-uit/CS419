# evaluation/plot.py
import matplotlib.pyplot as plt

def plot_precision_recall(precision_vals, recall_vals, title="Precision-Recall Curve"):
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


def plot_avg_11pt_interpolated_map(retrieved_docs_dict, relevant_docs_dict):
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

    avg_precisions = [sum(all_precisions_at_recall[r]) / len(all_precisions_at_recall[r])
                      if all_precisions_at_recall[r] else 0.0
                      for r in recall_levels]

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(recall_levels, avg_precisions, marker='o', color='green', label='Mean Interpolated Precision')
    plt.title('📈 11-point Interpolated Mean Average Precision (MAP)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 1.05)
    plt.xticks(recall_levels)
    plt.grid(True)
    plt.legend()
    plt.show()

    map_11pt = sum(avg_precisions) / 11
    print(f" MAP nội suy 11 điểm: {map_11pt:.4f}")
