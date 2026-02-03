from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, fbeta_score, accuracy_score
import numpy as np

def compute_roc_auc(y_true, y_prob):
    """
    Computes Area Under the Receiver Operating Characteristic Curve.
    """
    return roc_auc_score(y_true, y_prob)

def compute_brier(y_true, y_prob):
    """
    Computes Brier Score (mean squared error of probabilities).
    Lower is better.
    """
    return brier_score_loss(y_true, y_prob)

def compute_f1(y_true, y_pred):
    """
    Computes balanced F1 score.
    """
    return f1_score(y_true, y_pred)

def compute_c_at_1(y_true, y_pred, y_prob=None):
    """
    Computes C@1 metric.
    C@1 = (1/n) * (nc + nu * (nc/n))
    Where nc = number of correct answers, nu = number of unanswered.
    For this baseline, we assume zero unanswered unless y_prob is exactly 0.5.
    """
    n = len(y_true)
    if y_prob is not None:
        # Identify "unanswered" as predictions with probability exactly 0.5
        unanswered_mask = (y_prob == 0.5)
        n_u = np.sum(unanswered_mask)
        n_c = np.sum((y_pred == y_true) & (~unanswered_mask))
    else:
        n_u = 0
        n_c = np.sum(y_pred == y_true)
    
    if n == 0:
        return 0
    
    return (1/n) * (n_c + n_u * (n_c / n))

def compute_f05u(y_true, y_pred, y_prob=None):
    """
    Computes F0.5u metric.
    F0.5 score where "unanswered" (prob=0.5) are treated as false negatives.
    """
    if y_prob is not None:
        # Treat prob=0.5 as "unanswered" and map to a prediction that is never correct
        # Actually, the definition says treat as False Negatives.
        # So we can just modify y_pred to be 0 for these cases if they were 1?
        # Or just treat them as incorrect if they were 1?
        # Let's follow: "specifically treats 'non-answers' (predictions of 0.5) as false negatives"
        # Since 1 is AI, a non-answer (0.5) should have been 1 but was "missed".
        modified_y_pred = y_pred.copy()
        unanswered_mask = (y_prob == 0.5)
        modified_y_pred[unanswered_mask] = 0 # Map unanswered to "Human" (negative class)
        return fbeta_score(y_true, modified_y_pred, beta=0.5)
    else:
        return fbeta_score(y_true, y_pred, beta=0.5)

def get_all_metrics(y_true, y_pred, y_prob):
    """
    Returns a dictionary of all required PAN2025 metrics.
    """
    return {
        "ROC-AUC": compute_roc_auc(y_true, y_prob),
        "Brier": compute_brier(y_true, y_prob),
        "F1": compute_f1(y_true, y_pred),
        "C@1": compute_c_at_1(y_true, y_pred, y_prob),
        "F0.5u": compute_f05u(y_true, y_pred, y_prob)
    }


def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    """
    Plots and optionally saves a confusion matrix.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()
    return cm


def plot_roc_curve(y_true, y_prob, save_path=None, title="ROC Curve"):
    """
    Plots and optionally saves an ROC curve.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()
    return roc_auc

