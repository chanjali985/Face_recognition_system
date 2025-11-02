# eval_metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_face_recognition(y_true, y_pred):
    """
    Evaluate recognition predictions for current frame.
    Expects y_true and y_pred as lists (one or more items).
    """
    if not y_true or not y_pred:
        print("[⚠️] Skipping evaluation (empty inputs)")
        return

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n[📊 Face Recognition Evaluation]")
    print(f"   ✅ Accuracy:  {acc:.2f}")
    print(f"   🎯 Precision: {prec:.2f}")
    print(f"   🔁 Recall:    {rec:.2f}")
    print(f"   🧮 F1 Score:  {f1:.2f}")
    print("   Confusion Matrix:")
    print(cm)
