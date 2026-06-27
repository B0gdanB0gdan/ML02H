import numpy as np
import matplotlib.pyplot as plt

def auc_roc(scores, y_true):

    thresholds = sorted(scores, reverse=True)
    thresholds = [1.1] + thresholds + [0.0]

    roc_points = []

    P = sum(y_true == 1)
    N = sum(y_true == 0)

    for t in thresholds:
        predictions = scores >= t

        TP = np.sum((predictions == 1) & (y_true == 1))
        FP = np.sum((predictions == 1) & (y_true == 0))

        TPR = TP / P
        FPR = FP / N

        roc_points.append((FPR, TPR))

    roc_points = sorted(roc_points)

    auc = 0

    for i in range(len(roc_points)-1):
        x1, y1 = roc_points[i]
        x2, y2 = roc_points[i+1]

        width = x2 - x1
        height = (y1 + y2) / 2

        auc += width * height

    return roc_points, auc


y_true = np.array([1, 1, 0, 1, 0, 0])
scores = np.array([0.95, 0.90, 0.85, 0.70, 0.40, 0.10])

roc_points, auc = auc_roc(scores, y_true)

plt.figure(figsize=(6, 6))

fpr, tpr = zip(*(roc_points))
plt.plot(fpr, tpr, marker='o', label=f"ROC curve (AUC = {auc:.3f})")

# random baseline
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random classifier")

# formatting
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)

plt.show()