"""
Evaluation Metrics & Confusion Matrix Visualizer
"""
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CLASS_NAMES = ['Caries', 'Impacted Molar', 'Periapical Lesion', 'Bone Loss', 'Healthy']

def plot_confusion_matrix(y_true, y_pred, output_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth')
    plt.title('Dental Disease Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f'[✓] Confusion matrix saved to {output_path}')