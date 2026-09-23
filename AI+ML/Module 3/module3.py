from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # no display in Codespaces, so save plots to files instead
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

folder = Path(__file__).resolve().parent
csv_path = folder / "data.csv"


# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------
data = pd.read_csv(csv_path)

# "id" is just a patient number, and "Unnamed: 32" is an empty column
# created by a trailing comma in the CSV. Neither helps the model.
data = data.drop(columns=["id", "Unnamed: 32"])

print(data.head())
print(f"\nShape: {data.shape}")
print("\nDiagnosis counts:")
print(data["diagnosis"].value_counts())


# ---------------------------------------------------------------
# 2. Preprocessing: encode target and split
# ---------------------------------------------------------------
# Malignant = 1 (the "positive" class we care about detecting), Benign = 0
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"].map({"B": 0, "M": 1})

# Split BEFORE scaling so the scaler never sees the test data (no leakage).
# stratify=y keeps the same benign/malignant ratio in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")


# ---------------------------------------------------------------
# 3. Feature scaling
# ---------------------------------------------------------------
# fit_transform on train learns mean/std from training rows only;
# transform on test reuses those same numbers.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------------
# 4. Train Logistic Regression
# ---------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of Malignant


# ---------------------------------------------------------------
# 5a. Confusion matrix
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion matrix:")
print(f"  True Negatives  (benign, predicted benign):       {tn}")
print(f"  False Positives (benign, predicted malignant):    {fp}")
print(f"  False Negatives (malignant, predicted benign):    {fn}")
print(f"  True Positives  (malignant, predicted malignant): {tp}")

class_names = ["Benign", "Malignant"]
cell_labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#3987e5", "#0d366b"])

fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(cm, cmap=blues)
fig.colorbar(image, ax=ax, label="Number of patients")

ax.set_xticks([0, 1], labels=class_names)
ax.set_yticks([0, 1], labels=class_names)
ax.set_xlabel("Predicted diagnosis")
ax.set_ylabel("Actual diagnosis")
ax.set_title("Confusion matrix: Logistic Regression")

# Write the count and its name in each cell; white text on dark cells
for i in range(2):
    for j in range(2):
        text_color = "white" if cm[i, j] > cm.max() / 2 else "#0b0b0b"
        ax.text(j, i, f"{cm[i, j]}\n{cell_labels[i][j]}",
                ha="center", va="center", color=text_color, fontsize=12)

fig.tight_layout()
fig.savefig(folder / "confusion_matrix.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------
# 5b. Classification report
# ---------------------------------------------------------------
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

# Malignant is class 1, so pos_label=1 gives the metrics for Malignant
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("Malignant class:")
print(f"  Precision: {precision:.4f}  -> of the tumors flagged malignant, "
      f"{precision:.1%} really were")
print(f"  Recall:    {recall:.4f}  -> of the truly malignant tumors, "
      f"{recall:.1%} were caught ({fn} missed)")
print(f"  F1-score:  {f1:.4f}  -> balance of precision and recall")


# ---------------------------------------------------------------
# 5c. ROC curve and AUC
# ---------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
print(f"\nAUC: {auc:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#2a78d6", linewidth=2, zorder=3, clip_on=False)
ax.plot([0, 1], [0, 1], color="#52514e", linewidth=1, linestyle="--")
ax.text(0.55, 0.45, "Random guessing (AUC = 0.5)", color="#52514e", rotation=38)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("False Positive Rate (benign flagged as malignant)")
ax.set_ylabel("True Positive Rate (malignant caught)")
ax.set_title(f"ROC curve: Logistic Regression (AUC = {auc:.4f})")
ax.grid(color="#e6e5e0", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(folder / "roc_curve.png", dpi=150)
plt.close(fig)

print(f"\nPlots saved to {folder / 'confusion_matrix.png'} and {folder / 'roc_curve.png'}")
