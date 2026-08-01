import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Confusion Matrix from your image
cm = np.array([[33, 2, 1],
               [0, 45, 3],
               [0, 3, 24]])

classes = ['Dog_growl', 'Dog_grunt', 'Dog_whining']

# Convert confusion matrix to y_true and y_pred
y_true = []
y_pred = []

for i in range(len(cm)):
    for j in range(len(cm)):
        y_true += [i] * cm[i, j]
        y_pred += [j] * cm[i, j]

# Generate classification report
report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
accuracy = accuracy_score(y_true, y_pred)

report_df = pd.DataFrame(report).transpose()

# Create figure
fig = plt.figure(figsize=(12,5), dpi=300)

# ---------------- Confusion Matrix ----------------
plt.subplot(1,2,1)

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# ---------------- Classification Report ----------------
plt.subplot(1,2,2)

metrics = report_df.iloc[:-1, :3]

sns.heatmap(metrics,
            annot=True,
            cmap='Blues',
            fmt=".3f",
            cbar=False)

plt.title(f"Classification Report\nAccuracy = {accuracy:.3f}")
plt.xlabel("Metrics")
plt.ylabel("Class")

plt.tight_layout()

plt.savefig("model_evaluation.png", dpi=300)

plt.show()