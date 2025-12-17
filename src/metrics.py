import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(all_labels, all_preds, class_names, save_dir="./confusion_matrix"):
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(np.array(all_labels), np.array(all_preds))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    file_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved to {file_path}")
