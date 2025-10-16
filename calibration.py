import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# -------------------------------
# Threshold
# -------------------------------

def threshold_learning(scores: np.ndarray, labels: np.ndarray):
    """
    Learn an optimal similarity threshold based on F1 score.

    Args:
        scores: similarity scores (cosine similarities)
        labels: true binary labels (0/1)

    Returns:
        best_thr, best_f1
    """
    scores = np.asarray(scores)
    best_thr, best_f1 = 0.0, 0.0
    for thr in np.linspace(-1, 1, 200): # 200 is number of thresholds to try
        preds = (scores > thr).astype(np.int32)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    print(f"Learned threshold = {best_thr:.3f} (F1 = {best_f1:.3f})")
    return best_thr, best_f1


# -------------------------------
# Classifier
# -------------------------------

def classifier_learning(X: np.ndarray, labels: np.ndarray):
    """
    Train a simple classifier on cosine similarity.

    Args:
        X : element-wise difference of embeddings
        labels: true binary labels

    Returns:
        trained model
    """
    clf = LogisticRegression()
    clf.fit(X, labels)
    print("Trained logistic regression classifier for calibration.")
    return clf
