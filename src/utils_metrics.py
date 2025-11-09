
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

POKER_LABELS_SHORT = ["HC","OP","TP","3K","ST","FL","FH","4K","SF","RF"]

def evaluate_and_report(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average='macro')
    weighted = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, digits=3)
    return acc, macro, weighted, report

def plot_confusion(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=POKER_LABELS_SHORT, yticklabels=POKER_LABELS_SHORT,
                cbar_kws={"label":"Count"})
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()

def plot_class_f1(model_name, y_true, y_pred):
    class_f1 = f1_score(y_true, y_pred, average=None)
    plt.figure(figsize=(8,4))
    sns.barplot(x=POKER_LABELS_SHORT, y=class_f1, palette="viridis")
    plt.title(f"Per-Class F1 — {model_name}")
    plt.ylim(0,1); plt.xlabel("Class"); plt.ylabel("F1")
    plt.tight_layout(); plt.show()

def plot_true_vs_pred(model_name, y_true, y_pred):
    plt.figure(figsize=(6,5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0,9],[0,9],'k--',label='Perfect')
    plt.title(f"True vs Predicted — {model_name}")
    plt.xlabel("True Class"); plt.ylabel("Predicted Class")
    plt.xticks(range(10), POKER_LABELS_SHORT); plt.yticks(range(10), POKER_LABELS_SHORT)
    plt.legend(); plt.tight_layout(); plt.show()
