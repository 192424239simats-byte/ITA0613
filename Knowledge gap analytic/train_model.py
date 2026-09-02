"""
Trains:
  1. RandomForestClassifier -> predicts Competency (Beginner/Intermediate/Advanced)
  2. XGBRegressor (or GradientBoostingRegressor fallback) -> predicts next test score %

Run from the project root:
    python models/train_model.py
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBRegressor
    USING_XGBOOST = True
except ImportError:
    # xgboost isn't installed in this sandbox (no network access to pip
    # install it). Falls back to sklearn's GradientBoostingRegressor, which
    # has an identical fit/predict interface. Install xgboost and re-run
    # this script on a machine with internet access to use real XGBoost.
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    USING_XGBOOST = False

FEATURES = ["accuracy", "previous_score", "avg_time", "difficulty_level", "topic_accuracy"]

df = pd.read_csv("dataset/student_data.csv")
X = df[FEATURES]
y_comp = df["competency"]
y_score = df["next_score"]

le = LabelEncoder()
y_comp_enc = le.fit_transform(y_comp)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_comp_enc, test_size=0.2, random_state=42, stratify=y_comp_enc
)
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
print(f"[Random Forest] Competency accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}")

Xs_train, Xs_test, ys_train, ys_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, subsample=0.9)
xgb.fit(Xs_train, ys_train)
model_name = "XGBoost" if USING_XGBOOST else "XGBoost-fallback (GradientBoostingRegressor)"
print(f"[{model_name}] Next-score MAE: {mean_absolute_error(ys_test, xgb.predict(Xs_test)):.2f} points")

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(xgb, "models/xgboost.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(FEATURES, "models/feature_order.pkl")
print("Saved models/random_forest.pkl, models/xgboost.pkl, models/label_encoder.pkl")
