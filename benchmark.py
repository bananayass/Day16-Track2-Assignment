import json
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# 1. Load dataset
start = time.perf_counter()

df = pd.read_csv("creditcard.csv")

load_time = time.perf_counter() - start

print(f"Dataset shape: {df.shape}")
print(f"Load data time: {load_time:.4f} seconds")


# Features / target
X = df.drop(columns=["Class"])
y = df["Class"]


# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 2. Train LightGBM
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


start = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start


# 3. Predict test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# 4. Evaluation metrics
auc_roc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

best_iteration = getattr(model, "best_iteration_", None)

if not best_iteration:
    best_iteration = model.n_estimators


# 5. Inference latency - 1 row
one_row = X_test.iloc[[0]]

start = time.perf_counter()
model.predict_proba(one_row)
latency_seconds = time.perf_counter() - start

latency_ms = latency_seconds * 1000


# Inference throughput - 1000 rows
batch = X_test.iloc[:1000]

start = time.perf_counter()
model.predict_proba(batch)
batch_time = time.perf_counter() - start

throughput = len(batch) / batch_time


# 6. Save results
results = {
    "load_data_time_seconds": load_time,
    "training_time_seconds": training_time,
    "best_iteration": int(best_iteration),
    "auc_roc": auc_roc,
    "accuracy": accuracy,
    "f1_score": f1,
    "precision": precision,
    "recall": recall,
    "inference_latency_1_row_ms": latency_ms,
    "inference_throughput_rows_per_second": throughput,
}


with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=4)


print("\n===== BENCHMARK RESULTS =====")
print(f"Load data time        : {load_time:.4f} s")
print(f"Training time         : {training_time:.4f} s")
print(f"Best iteration        : {best_iteration}")
print(f"AUC-ROC               : {auc_roc:.6f}")
print(f"Accuracy              : {accuracy:.6f}")
print(f"F1-Score              : {f1:.6f}")
print(f"Precision             : {precision:.6f}")
print(f"Recall                : {recall:.6f}")
print(f"Inference latency     : {latency_ms:.4f} ms / row")
print(f"Inference throughput  : {throughput:.2f} rows/s")
print("\nSaved to benchmark_result.json")

