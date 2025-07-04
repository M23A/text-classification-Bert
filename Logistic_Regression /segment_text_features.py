import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from google.colab import drive


# Mount Google Drive
drive.mount('/content/drive')


# Load the CSV file
data = pd.read_csv('/content/drive/MyDrive/Maram/Text/Features_Extraction_from_text/Features_chunk_Text/Bert/chtxt_bert_features_labels.csv')

# Separate features (X) and labels (y)
X_combined = data.drop(['PID', 'CAI State'], axis=1).values
y_combined = data['CAI State'].values  # Labels column
# Define person_ids
person_ids = data['PID']

# Initialize Logistic Regression model
LR_model = LogisticRegression(max_iter=1000)

gkf = GroupKFold(n_splits=10)

# Store chunk-level predictions and true labels
accuracies = []
precisions = []
recalls = []
f1s = []
confusion_matrices = []

for train_index, test_index in gkf.split(X_combined, y_combined, groups=person_ids):
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = y_combined[train_index], y_combined[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    LR_model.fit(X_train_scaled, y_train)
    y_pred = LR_model.predict(X_test_scaled)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1s.append(f1_score(y_test, y_pred, average='weighted'))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

print("Evaluation Results:")
print(f"Accuracy:  {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
print(f"Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
print(f"Recall:    {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
print(f"F1 Score:  {np.mean(f1s):.2f} ± {np.std(f1s):.2f}")
