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
