import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

# Confusion matrix from your figure
cm = np.array([[33, 2, 1],
               [0, 45, 3],
               [0, 3, 24]])

class_names = ['Dog_growl', 'Dog_grunt', 'Dog_whining']

# Convert confusion matrix to y_true and y_pred
y_true = []
y_pred = []

for i in range(len(cm)):
    for j in range(len(cm)):
        y_true += [i] * cm[i, j]
        y_pred += [j] * cm[i, j]

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Convert report to table
import pandas as pd
report_df = pd.DataFrame(report).transpose()

# Plot high quality figure
plt.figure(figsize=(8,4), dpi=600)
sns.heatmap(report_df.iloc[:-1, :3],
            annot=True,
            annot_kws={"weight": "bold", "size":12},
            cmap="Blues",
            fmt=".3f",
            cbar=False)

plt.title("Classification Report",fontweight='bold')
plt.ylabel("Class",fontweight='bold')
plt.xlabel("Metrics",fontweight='bold')

plt.tight_layout()
plt.savefig("classification_report.png", dpi=600)
plt.show()