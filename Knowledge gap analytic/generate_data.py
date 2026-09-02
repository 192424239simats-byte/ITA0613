"""
Generates a synthetic dataset of student performance records used to train
the Random Forest (competency) and XGBoost (next-score) models.

Feature order matches utils/preprocess.py build_feature_vector():
    [accuracy, previous_score, avg_time, difficulty_level, topic_accuracy]

Target columns:
    competency   - Beginner / Intermediate / Advanced   (Random Forest target)
    next_score   - predicted score % on the next test    (XGBoost target)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

accuracy = np.clip(np.random.normal(60, 20, N), 5, 99)
previous_score = np.clip(accuracy + np.random.normal(0, 8, N), 0, 100)
avg_time = np.clip(np.random.normal(55, 20, N), 10, 150)          # seconds/question
difficulty_level = np.random.choice([1, 2, 3], size=N, p=[0.35, 0.4, 0.25])
topic_accuracy = np.clip(accuracy + np.random.normal(0, 12, N), 0, 100)

competency_score = (
    0.5 * accuracy + 0.25 * topic_accuracy + 0.15 * (difficulty_level * 20) - 0.1 * avg_time
)
competency = pd.cut(
    competency_score,
    bins=[-np.inf, 45, 70, np.inf],
    labels=["Beginner", "Intermediate", "Advanced"],
)

improvement = (100 - accuracy) * 0.08 + (difficulty_level - 2) * 2
next_score = np.clip(
    accuracy + improvement + np.random.normal(0, 6, N) - (avg_time > 100) * 3,
    0,
    100,
)

df = pd.DataFrame({
    "accuracy": accuracy.round(2),
    "previous_score": previous_score.round(2),
    "avg_time": avg_time.round(1),
    "difficulty_level": difficulty_level,
    "topic_accuracy": topic_accuracy.round(2),
    "next_score": next_score.round(2),
    "competency": competency.astype(str),
})

df.to_csv("dataset/student_data.csv", index=False)
print("Saved dataset/student_data.csv with", len(df), "rows")
print(df.head())
